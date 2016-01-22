
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
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'random number generator\'s seed')
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
local loader = ExternalMinibatchLoader.create(1, 1)

print('loading the model from checkpoint ' .. opt.init_from)
local checkpoint = torch.load(opt.init_from)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.init_from .. '...')

current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'lstm' then
        table.insert(current_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

function dump_embeddings(index, embed_annot, file)
  print ("dumping", index)
  for _,node in ipairs(protos.rnn.forwardnodes) do
    if node.data.annotations.name == embed_annot then
      local x= node.data.module.output:float()
      file:writeString(index ..": ")
      file:writeFloat(x:storage())
    
    end
  end
  --sequence:apply(function(x) seq = seq .. ' ' .. x end)
  --client:hincrbyfloat(s, freq, sequence)
  --client:hmset(s, seq, freq)      
  --sys.sleep(1)
end

-- Scan and dump embeddings
local x = torch.DoubleTensor(1, 2*theta_size)
local e_x = torch.IntTensor(1,1):fill(1) 

embed_theta_file = torch.DiskFile('embed_theta.txt', 'w')
local start_timestamp = 1452241177
for i = 1, 60*24*7 do -- iterate over batches in the split
  local theta, date = ExternalMinibatchLoader.timestamp2theta(start_timestamp+i*60)
  x:sub(1,1, 1,theta_size):copy(theta)
  x:sub(1,1, theta_size+1, -1):copy(theta)
  
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    e_x = e_x:float():cuda()
  end
  
  protos.rnn:forward{x, e_x, unpack(current_state)}
  local week_mins = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1) 
  dump_embeddings(week_mins, 'emb_theta_sigm', embed_theta_file)
end
embed_theta_file:close() -- make sure the data is written

local n = get_size_softmax_layer(protos.rnn.forwardnodes, "decoder")
embed_event_file = torch.DiskFile('embed_event.txt', 'w')
for i = 1, n do -- iterate over batches in the split
  e_x:fill(i)    
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    e_x = e_x:float():cuda()
  end
  protos.rnn:forward{x, e_x, unpack(current_state)}
  dump_embeddings(i, 'emb_e', embed_event_file)
end
embed_event_file:close()




