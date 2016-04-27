
require 'torch'
require 'nn'
require 'nngraph'
require 'gnuplot'
require 'sys'

require 'util.misc'
require 'util.SequenceFactory'

local ExternalMinibatchLoader_NextEvent = require 'util.ExternalMinibatchLoader_NextEvent'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict the next event in a stream. Waits for file paths to be pushed on <redis_prefix>-predictions')
cmd:text()
cmd:text('Options')
cmd:option('-seq_len',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-redis_prefix', '', 'redis key name prefix, where to read train/validation/events data')
cmd:option('-window_size',60,'width of window for weighted averaging of time predictions')

cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sequence_provider', 3,'which sequence provider to use, 1 for redis, 2 for synthetic, 3 from disc')

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

factory_initialization()

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
checkpoint = torch.load(opt.init_from)
protos = checkpoint.protos

opt.num_layers = checkpoint.opt.num_layers

-- create the data loader class  
local loader = ExternalMinibatchLoader_NextEvent.create()

-- initialize the rnn state to all zeros
function init_state()
   local state={}
    for L = 1,checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(opt.batch_size, checkpoint.opt.rnn_size)
        if opt.gpuid >= 0 then h_init = h_init:cuda() end
        table.insert(state, h_init:clone())
        if checkpoint.opt.rnn_unit == 'lstm' then
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

print ("starting predictor")
protos.rnn:evaluate() -- put in eval mode so that BatchNormalization and DropOut works properly
--protos.rnn:training() -- put in eval mode so that BatchNormalization and DropOut works properly

batch_type = "testing" -- only look at test data
while true do
  -- fetch a batch
  local x, y, e_x, e_y = loader:next_batch()
  current_state = init_state()
  -- forward pass
  local lst, last_pred
  for t=1, opt.seq_len do
    if t > 1 then
      print ("e_x[t]", opt.event_inv_mapping[e_x[t][1]], opt.event_inv_mapping[last_pred])
    else 
      print ("e_x[t]", opt.event_inv_mapping[e_x[t][1]])
    end
    
    lst = protos.rnn:forward{x[t], e_x[t], unpack(current_state)}
   
    local e_y_probs = lst[#lst]:clone():exp():double()
    local e_y_probs_sorted, e_y_probs_idx = torch.sort(e_y_probs, 2, true) -- sort along 2nd dim, true = in descending order
    
    local y_probs = lst[#lst-1]:clone():exp():double()
    smooth_probs(y_probs, opt.window_size)
    
    -- The predictions for the events are in e_y_probs_sorted (the indices are in e_y_probs_idx) 
    -- The predictions for the time slots are in y_probs (not sorted)
    
    
    last_pred = e_y_probs_idx[1][1]
    if e_x[t][1] > 1 then
      display_uncertainty(y_probs, y[t], e_y_probs, e_y[t])
    end
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
  end
  collectgarbage()
end






