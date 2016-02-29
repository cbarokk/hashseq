
--[[

This file dumps the embeddings learned by the model

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
local ExternalMinibatchLoader_past = require 'util.ExternalMinibatchLoader_past'
local model_utils = require 'util.model_utils'

local redis = require 'redis'
local client = redis.connect('127.0.0.1', 6379)


cmd = torch.CmdLine()
cmd:text()
cmd:text('Dump hash codes from sparse models')
cmd:text()
cmd:text('Options')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')

cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-redis_queue', '', 'name of the redis queue to read from')
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


print('loading the model from checkpoint ' .. opt.init_from)
local checkpoint = torch.load(opt.init_from)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

opt.num_events = get_size_softmax_layer(protos.rnn.forwardnodes, "decoder")
opt.num_time_slots = get_size_softmax_layer(protos.rnn.forwardnodes, 'theta_pred')

-- create the data loader class  
local loader = ExternalMinibatchLoader_past.create(opt.batch_size, opt.seq_length, num_events, opt.redis_queue)


-- initialize the rnn state to all zeros
function init_state()
   state={}
    for L = 1,checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(opt.batch_size, checkpoint.opt.rnn_size)
        if opt.gpuid >= 0 then h_init = h_init:cuda() end
        table.insert(state, h_init:clone())
        if checkpoint.opt.model == 'lstm' then
            table.insert(state, h_init:clone())
        end
    end
    return state
end
current_state = init_state()
state_size = #current_state

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

function dump_hidden_state(h_i)
  gnuplot.figure("hist top_h")
  gnuplot.title("top_h")
  gnuplot.hist(h_i, 100)
  local h = torch.histc(h_i,200,-1,1)
  -- print(h:div(h:sum()), h:sum())
  hash_codes_file:writeFloat(h:div(h:sum()):storage())
      
  h_i:cmax(0):ceil()
  --sys.sleep(0.5)
  for j=1,h_i:size(1) do
    local s = ""  
    h_i[j]:apply(function(x) s = s .. x end)
    --print ("dumping code" .. s, loader.batch[j])
    print ("dumping code: " .. s)
    client:sadd(s, loader.batch[j])
  end
  
  --sequence:apply(function(x) seq = seq .. ' ' .. x end)
  --client:hincrbyfloat(s, freq, sequence)
  --client:hmset(s, seq, freq)      
  --sys.sleep(1)
end

-- start dumping
hash_codes_file = torch.DiskFile('hash_codes.txt', 'w')

print ("starting demo")
while true do
  -- fetch a batch
   local x, y, e_x, e_y = loader:next_batch()
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
     -- have to convert to float because integers can't be cuda()'d
     x = x:float():cuda()
     y = y:float():cuda()
    e_x = e_x:float():cuda()
    e_y = e_y:float():cuda()
  end
  current_state = init_state()
  -- forward pass
  local lst
  for t=1,opt.seq_length do
     lst = protos.rnn:forward{x[{{},t,{}}], e_x[{{},t,{}}], unpack(current_state)}
     current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
  end
  dump_hidden_state(lst[#lst- 2]:float())
  collectgarbage()
end






