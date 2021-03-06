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
require 'util.QuadraticPenalty'

require 'util.misc'
local ExternalMinibatchLoader_past = require 'util.ExternalMinibatchLoader_past'
local model_utils = require 'util.model_utils'
local LSTM_theta_past = require 'model.LSTM_theta_past'
local GRU_theta_past = require 'model.GRU_theta_past'
local RNN_theta_past = require 'model.RNN_theta_past'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a timestamped events model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-rnn_size', 1024, 'size of LSTM internal state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-num_events', 0, 'size of events vocabulary')
cmd:option('-num_time_slots', 10080, 'size of time vocabulary')
cmd:option('-encoder_shape', '256,32', 'size of the hash codes')


cmd:option('-model', 'lstm', 'lstm,gru or rnn')
-- optimization
cmd:option('-theta_weight',1.0,'weight for loss function')
cmd:option('-event_weight',1.0,'weight for loss function')


cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-learning_rate_decay_freq', 10, 'decay learning rate every X epochs')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',5000000,'max number of epochs through the training data')
cmd:option('-iterations_per_epoch',100,'number of iterations per epoch')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-lambda', 0, 'Quadratic Penalty for regularization, used at last RNN hidden layer. 0 = no regularization')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-save_every',1,'how many steps/minibatches between dumping a checkpoint')
cmd:option('-start_lambda_after',1500,'number of epochs to start lambda, i.e. fine tuning')


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

opt.encoder = {opt.rnn_size, unpack(opt.encoder_shape:split(","))}
for i =1, #opt.encoder do
  opt.encoder[i] = tonumber(opt.encoder[i])
end



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
local loader = ExternalMinibatchLoader_past.create()
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
    --opt.learning_rate = checkpoint.learning_rate
    opt.encoder = checkpoint.opt.encoder
    do_random_init = false
    start_epoch = checkpoint.epoch
    print ("watch out!, last learning rate:", checkpoint.learning_rate)
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
       protos.rnn = LSTM_theta_past.lstm()
    elseif opt.model == 'gru' then
       protos.rnn = GRU_theta_past.gru()
    elseif opt.model == 'rnn' then
       protos.rnn = RNN_theta_past.rnn()
    end
    
    local crit1 = nn.DistKLDivCriterion()
    local crit2 = nn.DistKLDivCriterion()
    protos.criterion = nn.ParallelCriterion():add(crit1, opt.theta_weight):add(crit2, opt.event_weight)
    
end


-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
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
if opt.model == 'lstm' then
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



function set_lambda()
  for t=1,opt.seq_length do
    for _,node in ipairs(clones.rnn[t].forwardnodes) do
      if node.data.annotations.name == "hashcode" then
        node.data.module.lambda = opt.lambda
      end
    end
  end 
end


-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, e_x, e_y = loader:next_batch(opt.redis_queue)
    local hashcodes = torch.DoubleTensor(opt.batch_size, opt.seq_length, opt.encoder[#opt.encoder]):fill(0)
    local zero_codes = torch.DoubleTensor(opt.batch_size, opt.encoder[#opt.encoder]):fill(0)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
      -- have to convert to float because integers can't be cuda()'d
      x = x:float():cuda()
      y = y:float():cuda()
      e_x = e_x:float():cuda()
      e_y = e_y:float():cuda()
      hashcodes = hashcodes:float():cuda()
      zero_codes = zero_codes:float():cuda()
    end
     ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0
    --set_lambda()

    for t=1,opt.seq_length do
       clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
       local lst = clones.rnn[t]:forward{x[{{},t,{}}], e_x[{{},t,{}}], unpack(rnn_state[t-1])}
    
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      predictions[t] = { lst[#lst- 1], lst[#lst]} -- last elements is the prediction
      hashcodes[{{}, t, {}}] = lst[#lst- 2] -- get the hash codes
      
      loss = loss + clones.criterion[t]:forward(predictions[t], { y[{{}, t, {}}], e_y[{{}, t, {}}] })
      
      --gnuplot.figure("predictions")
      --gnuplot.title("predictions")
      --gnuplot.plot({"y", y[{{}, t, {}}][1], '|'}, {"x", lst[#lst- 1][1]:exp():double(),'|'})
      --gnuplot.plot({"x", lst[#lst- 1][1]:exp():double(),'-'})
      --gnuplot.plot({"e_y", e_y[{{}, t, {}}][1], '|'}, {"e_x", lst[#lst][1]:exp():double(),'|'})
      
    end
    loss = loss / opt.seq_length
    
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    
    
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    
    for t=opt.seq_length,1,-1 do
      -- backprop through loss, and softmax/linear
      --local doutput_t = clones.criterion[t]:backward(predictions[t], { y[{{}, t, {}}]:clone():view(opt.batch_size*60*24*7), e_y[{{}, t, {}}]:clone():view(opt.batch_size*opt.num_events)})
      local doutput_t = clones.criterion[t]:backward(predictions[t], { y[{{}, t, {}}], e_y[{{}, t, {}}]})
      table.insert(drnn_state[t], zero_codes)
      table.insert(drnn_state[t], doutput_t[1])
      table.insert(drnn_state[t], doutput_t[2])
    
      local dlst = clones.rnn[t]:backward({x[{{},t,{}}], e_x[{{},t,{}}], unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k > 2 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          drnn_state[t-1][k-2] = v
        end
      end
    --
    end
    ---------------------- misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    
    
    
    --gnuplot.figure("hashcodes ")
    --gnuplot.title("hashcodes ")
    --gnuplot.hist(hashcodes , 100)
      
  
    --sparsity = hashcodes:abs():lt(0.1):sum()/hashcodes:nElement()
    sparsity = hashcodes:abs():mean()
    return loss, grad_params
end


-- start optimization here
sparsity = 0
sparse_loss = 0
lambda = opt.lambda
set_lambda()
local train_losses = {}
local monitor_losses = {}
local learning_rates={}
local loss0 = nil
local accum_train_loss = 0


local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local decay_learning_rate = false
for epoch = start_epoch, opt.max_epochs do
  for i = 1, opt.iterations_per_epoch do
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    accum_train_loss = accum_train_loss + train_loss
  
    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %d), train_loss = %6.8f, grad/param norm = %6.4e, lambda = %2.5f, sparsity = %2.2f, time/batch = %.2fs", i, opt.iterations_per_epoch, epoch, accum_train_loss/i, grad_params:norm() / params:norm(), lambda, sparsity, time))
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
  if #monitor_losses > 10 then
    local y = torch.Tensor(monitor_losses)
    local x = torch.Tensor(#monitor_losses,3):fill(1)
    for i=1, x:size()[1] do
      x[{i,1}] = i
      x[{i,2}] = 1.0/i
    end
    
    local theta = normal_equations(x, y)
    local p_x = torch.linspace(1,#monitor_losses, #monitor_losses)
    local p_y = p_x:clone():mul(theta[1]):add(torch.ones(p_x:size()):cdiv(p_x):mul(theta[2])):add(theta[3])
    
    local theta_prime = torch.ones(p_x:size()):mul(-1*theta[2]):cdiv((p_x:clone():cmul(p_x))):add(theta[1])
    
    gnuplot.figure('monitor losses')
    gnuplot.title('monitor losses ' .. opt.info)
    gnuplot.plot({torch.Tensor(monitor_losses),'-'},  {"theta", p_x, p_y,'-'})
    
    theta_prime:div(theta_prime:min())
    print("theta_prime", theta_prime)
    gnuplot.figure('theta prime')
    gnuplot.title('theta prime ' .. opt.info)
    gnuplot.plot({"theta_prime", p_x, theta_prime,'-'})
      
    
    if theta_prime:min() < 0.05 then
      decay_learning_rate = true
      monitor_losses = {}
    end
  end
  
    
  if loss0 == nil then loss0 = accum_train_loss end
  if accum_train_loss > loss0 * 3 then
    print('loss is exploding, aborting.')
    break -- halt
  end
 
  -- exponential learning rate decay
  if decay_learning_rate then
    local decay_factor = opt.learning_rate_decay
    optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
    print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
    decay_learning_rate = false
  end

  -- every now and then or on last iteration
  if epoch % opt.save_every == 0 then
    local savefile = string.format('%s/model_past_%s_%s_%s_%s_%s_%s_epoch%.2f_%.8f.t7', opt.checkpoint_dir, opt.model, opt.num_layers, opt.rnn_size, opt.encoder_shape, opt.theta_weight, opt.event_weight, epoch, accum_train_loss)
    accum_train_loss = 0
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    checkpoint.train_losses = train_losses
    checkpoint.epoch = epoch
    checkpoint.vocab = loader.vocab_mapping
    checkpoint.learning_rate = optim_state.learningRate
    torch.save(savefile, checkpoint)
  end
end

