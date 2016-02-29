
--[[

This file runs a battery of tests on validation data

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'util.QuadraticPenalty'
require 'sys'

require 'util.misc'
local ExternalMinibatchLoader = require 'util.ExternalMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM_theta = require 'model.LSTM_theta'
local GRU_theta = require 'model.GRU_theta'
local RNN_theta = require 'model.RNN_theta'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Validates timestamped events model')
cmd:text()
cmd:text('Options')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',30,'number of sequences to validate on in parallel')
cmd:option('-num_batches',1000,'number of batches to use for validation')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'random number generator\'s seed')
-- bookkeeping
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')

cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

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
local loader = ExternalMinibatchLoader.create(opt.batch_size, opt.seq_length)

print('loading the model from checkpoint ' .. opt.init_from)
local checkpoint = torch.load(opt.init_from)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly
    
protos.criterion = nn.ClassNLLCriterion()

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.init_from .. '...')

init_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(opt.batch_size, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end


-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

--local init_state_global = clone_list(init_state)

-- evaluate the loss over an entire split
function eval_split()
    print('evaluating classification accuracy over validation data ')
    local loss = 0
    local rnn_state = {[0] = init_state}
    local top_k = 3
    local classification_accuracy = torch.Tensor(top_k, opt.seq_length-20):fill(0)
    
    for i = 1, opt.num_batches do -- iterate over batches in the split
      -- fetch a batch
      local x, e_x, e_y = loader:next_batch('validate')
      if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        e_x = e_x:float():cuda()
        e_y = e_y:float():cuda()
      end
      
      local predictions = {}
      -- forward pass
      for t=1,opt.seq_length-20 do

        clones.rnn[t]:evaluate() -- for dropout proper functioning
        local lst = clones.rnn[t]:forward{x[{{},t,{}}], e_x[{{},t,{}}], unpack(rnn_state[t-1])}
    
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
        predictions[t] = lst[#lst] -- last elements is the prediction
        
        local sorted_y, sorted_i = torch.sort(lst[#lst], true)
        for idx=1,top_k do
          if sorted_i[1][idx] == e_y[{{}, t, {}}][{1, 1}] then
            classification_accuracy:sub(idx,top_k, t, t):add(1) 
            break
          end
        end
     end
      -- carry over lstm state
      rnn_state[0] = rnn_state[#rnn_state]
      print(i .. '/' .. opt.num_batches .. '...')
      collectgarbage()
    end
    classification_accuracy:div(opt.num_batches)
    
    
    gnuplot.figure("classification accuracy")
    gnuplot.title("classification accuracy")
    gnuplot.grid(true)
    gnuplot.axis{0,40,0,1.0}
    local temp = {}
    for i=top_k, 1, -1 do
      table.insert(temp, {"top "..i, torch.Tensor(classification_accuracy[i])})
    end
    gnuplot.plot(unpack(temp))
    
    gnuplot.figure("classification accuracy avg by num guesses")
    gnuplot.title("classification accuracy avg by num guesses")
    gnuplot.plot({classification_accuracy:clone():mean(2)})


--    print ("classification_accuracy", torch.mean(classification_accuracy, 2))

    sys.sleep(30000)
end



-- start validation here

eval_split()



