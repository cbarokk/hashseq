--[[

This file trains a event-based multi-layer RNN on timestamped data

Code is based on char-rnn from karpathy based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'

require 'util.misc'
local ExternalMinibatchLoader_NextEvent = require 'util.ExternalMinibatchLoader_NextEvent'
local ExternalMinibatchLoader_past = require 'util.ExternalMinibatchLoader_past'

local model_utils = require 'util.model_utils'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a timestamped events model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-num_events', 500, 'size of events vocabulary')
cmd:option('-num_weekly_slots', 10080, 'size of time vocabulary')
cmd:option('-model', 'next_event', 'next_event or hashing')
cmd:option('-rnn_unit', 'lstm', 'lstm,gru or rnn')
cmd:option('-encoder_shape', '256,32', 'size of the hash codes')
cmd:option('-theta_weight',1.0,'weight for loss function')
cmd:option('-event_weight',1.0,'weight for loss function')

-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',50,'number of epochs, before considering decaying the learning rate')
cmd:option('-learning_rate_decay_threshold',-1e-8,'maximum slope of 2nd derivative ')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',5000000,'number of full passes through the training data')
cmd:option('-iterations_per_epoch',100,'number of iterations per epoch')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-save_every',500,'how many steps/minibatches between dumping a checkpoint')

cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','model','filename to autosave the checkpoint to. Will be inside checkpoint_dir/')
cmd:option('-redis_queue', '', 'name of the redis queue to read from')
cmd:option('-info', '', 'small string, just to simplify viewing several plotting windows at once')


-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class  
if opt.model == 'next_event' then
  opt.loader = ExternalMinibatchLoader_NextEvent.create()
elseif opt.model == 'hashing' then
  opt.loader = ExternalMinibatchLoader_past.create()
end

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end
local start_epoch = 0

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos

    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    start_epoch = checkpoint.epoch
    do_random_init = false
else
    protos = opt.loader.create_rnn_units_and_criterion()
    
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if opt.rnn_unit == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.rnn_unit == 'lstm' then
  for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
      if node.data.annotations.name == "i2h_" .. layer_idx then
        print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
        -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
        node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
      end
    end
  end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end


-- do fwd/bwd and return loss, grad_params
init_state_global = clone_list(init_state)


-- start optimization here
local train_losses = {}
local monitor_losses = {}
local learning_rates={}
local slopes={}

local loss0 = nil
local accum_train_loss

local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local decay_learning_rate = false
local current_slope = 0

for epoch = start_epoch, opt.max_epochs do
  accum_train_loss = 0
  for i = 1, opt.iterations_per_epoch do
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(opt.loader.feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    accum_train_loss = accum_train_loss + train_loss
  
    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs, slope = %2.2e", i, opt.iterations_per_epoch, epoch, train_loss, grad_params:norm() / params:norm(), time, current_slope))
    end
  
    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
      print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
      break -- halt
    end
  end
  
  accum_train_loss = accum_train_loss / opt.iterations_per_epoch
  train_losses[#train_losses+1] = accum_train_loss
  gnuplot.figure('train losses')
  gnuplot.title('train losses ' .. opt.info)
  gnuplot.plot({torch.Tensor(train_losses),'-'})
  
  learning_rates[#learning_rates+1] = optim_state.learningRate
  gnuplot.figure('learning rate')
  gnuplot.title('learning rate ' .. opt.info)
  gnuplot.plot({torch.Tensor(learning_rates),'-'})
  
  monitor_losses[#monitor_losses+1] = accum_train_loss
  
  if #monitor_losses > 3 then
    local y = torch.Tensor(monitor_losses)
    local x = torch.Tensor(#monitor_losses,4):fill(1)
    for i=1, x:size()[1] do
      x[{i,1}] = i*i*i
      x[{i,2}] = i*i
      x[{i,3}] = i
    end
    
    local theta = normal_equations(x, y)
    local p_x = torch.linspace(1,#monitor_losses, #monitor_losses)
    local p_y = p_x:clone():cmul(p_x):cmul(p_x):mul(theta[1]):add(p_x:clone():cmul(p_x):mul(theta[2])):add(p_x:clone():mul(theta[3])):add(theta[4])
    
    --local first_derivative = p_x:clone():cmul(p_x):mul(3*theta[1]):add(p_x:clone():mul(2*theta[2])):add(theta[3])
    --local second_derivative = p_x:clone():mul(6*theta[1]):add(2*theta[2])

    current_slope = 6*theta[1]
    slopes[#slopes+1] = current_slope
    gnuplot.figure('slopes')
    gnuplot.title('slopes ' .. opt.info)
    gnuplot.plot({torch.Tensor(slopes),'-'})
    
    gnuplot.figure('monitor losses')
    gnuplot.title('monitor losses ' .. opt.info)
    gnuplot.plot({"loss", torch.Tensor(monitor_losses),'-'},  {"theta", p_x, p_y,'-'})
    
    --[[
    gnuplot.figure('derivatives')
    gnuplot.title('derivatives ' .. opt.info)
    gnuplot.plot({"1st", p_x, first_derivative,'-'}, {"2nd", p_x, second_derivative,'-'})
      ]]--
  
    if #monitor_losses > opt.learning_rate_decay_after then
      if current_slope > opt.learning_rate_decay_threshold then
        decay_learning_rate = true
        monitor_losses = {}
      end
    end
  end
  
  if loss0 == nil then loss0 = accum_train_loss end
  if accum_train_loss > loss0 * 3 then
    print('loss is exploding, aborting.')
    break -- halt
  end
  
  if decay_learning_rate then
    local decay_factor = opt.learning_rate_decay
    optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
    print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    decay_learning_rate = false
  end
  
  -- every now and then or on last iteration
  if epoch % opt.save_every == 0 then
    local savefile = string.format('%s/model_past_%s_%s_%s_%s_%s_%s_epoch%.2f_%.8f.t7', opt.checkpoint_dir, opt.rnn_unit, opt.num_layers, opt.rnn_size, opt.encoder_shape, opt.theta_weight, opt.event_weight, epoch, accum_train_loss)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.epoch = epoch
    checkpoint.learning_rate = optim_state.learningRate
    torch.save(savefile, checkpoint)
  end
  
end


