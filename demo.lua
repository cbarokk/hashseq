
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
local colors = require 'ansicolors'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Validates timestamped events model')
cmd:text()
cmd:text('Options')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-mapping','','mapping between event Ids and internal Ids')

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
local loader = ExternalMinibatchLoader.create(1, opt.seq_length)

print('loading the model from checkpoint ' .. opt.init_from)
local checkpoint = torch.load(opt.init_from)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly
    
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
state_size = #current_state

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end


local file = io.open (opt.mapping, "r")
mapping = {}
for line in file:lines() do 
  local words = string.split(line, ":")
  mapping[tonumber(words[2])] = words[1]
end

-- start demo
local top_k = 2
local all_observations = 0
local intervals = torch.Tensor(60*24*7):fill(0)

print ("starting demo")
while true do
  -- fetch a batch
  
  local classification_accuracy = torch.Tensor(top_k):fill(0)
  local observations = 0

  local x, e_x, e_y = loader:next_batch('validate')
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    e_x = e_x:float():cuda()
    e_y = e_y:float():cuda()
  end
    
  -- forward pass
 
  print ("\nNEW RANDOM CUSTOMER (50 events)")
  print ("------------------------------------------------------------------------------------------")
  print("DATE                        ", "| SITE |", "| 1st GUESS", "| 2nd GUESS", "| Accuracy")
  print ("------------------------------------------------------------------------------------------")
    
  
    
  for t=1,opt.seq_length do
    local lst = protos.rnn:forward{x[{{},t,{}}], e_x[{{},t,{}}], unpack(current_state)}

    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    
    local ey_truth = e_y[{{}, t, {}}][{1,1}]
    
    if t == 1 then
      print (os.date ("%c", loader.dates[t]), "| ".. mapping[ey_truth] .."|", "|***************|***************|*********")
    end

    
    if t > 1 then 
    
      all_observations = all_observations + 1
      
      local event_probs = lst[#lst]:clone():exp():double()
      local sorted_ey, sorted_ei = torch.sort(event_probs, true)
    
      
    
      for idx=1,top_k do
        if sorted_ei[1][idx] == e_y[{{}, t, {}}][{1, 1}] then
          classification_accuracy:sub(idx,top_k):add(1)
          break
        end 
      end
      observations = observations + 1
     
      
    local printtop_k = {}
    local hit = 0
    for i =1, top_k do
      if sorted_ei[1][i] == ey_truth then 
        printtop_k[i] = colors('%{green}' .. string.format("| %s [%.1f%%]", mapping[sorted_ei[1][i]],100*sorted_ey[1][i]))
        hit = 1
      else
        printtop_k[i] = string.format("| %s [%.1f%%]", mapping[sorted_ei[1][i]],100*sorted_ey[1][i])
      end
    end
    printtop_k[top_k+1] = string.format('| %d%%', classification_accuracy[top_k]/(observations)*100)
    
    local truth = colors('%{red}' .. mapping[ey_truth])
    
    
    if hit == 1 then truth = colors('%{green}' .. mapping[ey_truth]) end
    
    print (os.date ("%c", loader.dates[t+1]), "| " .. truth .. "|", unpack(printtop_k))
    
    
    --[[gnuplot.figure("event_probs")
    gnuplot.title("event_probs")
    gnuplot.plot({torch.Tensor(event_probs[1]),'-'},  {"e_y", ey_t:clone():mul(event_probs[1]:max()), '|'})
    ]]--
    --[[local y_t4 = torch.Tensor(60*24*7):fill(0)
    local width = 120
    for j=1,60*24*7 do
      local stop_x = math.min(60*24*7, j+width)
      y_t4[j] = probs[1]:sub(j, stop_x):sum()
      local start_x = math.max(1, j-width)
      y_t4[j] = probs[1]:sub(start_x, j):sum()
    end
    gnuplot.figure("guess")
    gnuplot.title("guess")
    gnuplot.raw('set xtics ("Sun" 1, "06:00" 360, "12:00" 720, "18:00" 1080, "Mon" 1440, "06:00" 1800, "12:00" 2160, "18:00" 2520, "Tue" 2880, "06:00" 3240, "12:00" 3600, "18:00" 3960,"Wed" 4320, "06:00" 4680, "12:00" 5040, "18:00" 5400, "Thu" 5760, "06:00" 6120, "12:00" 6480, "18:00" 6840, "Fri" 7200, "06:00" 7560, "12:00" 7920, "18:00" 8280, "Sat" 8640, "06:00" 9000, "12:00" 9360, "18:00" 9720)')

    gnuplot.plot({y_t4,'-'}, {"y", y_t:clone():mul(y_t4:max()), '|'})
    
    
    if t > 1 then
      local y_t3 = torch.Tensor(60*24*7):fill(0)
      local cumul_y = torch.Tensor(60*24*7):fill(0)
    
      y_t3[y_truth] = 1
      local sum=0
      local last_y = y[{{}, t-1, {}}][{1,1}]
      for j=last_y, 60*24*7 do
        sum = sum + probs[1][j]
        cumul_y[j] = sum
      end
      
      --for j=1, last_y-1 do
      --  sum = sum + probs[1][j]
      --  cumul_y[j] = sum
      --end
    
      local x_axis = torch.Tensor(60*24*7-last_y+1):fill(0)
      local j=0
      x_axis:apply(function(x)
          j=j+1
          return j+last_y
        end)
      gnuplot.figure("cumulative distribution")
      gnuplot.title("cumulative distribution")
      
      --gnuplot.raw('set xtics ("Jan" 1, "Feb" 2, "Mar" 3, "Apr" 4, "May" 5, "Jun" 6, "Jul" 7, "Aug" 8, "Sep" 9, "Oct" 10, "Nov" 11, "Dec" 12)')
      --gnuplot.axis{0,13,0,''}
      gnuplot.plot({"confidence", x_axis, torch.Tensor(cumul_y):sub(last_y,-1),'-'}, {"y", torch.Tensor(1):fill(y_truth), y_t3:sub(y_truth,y_truth), '|'})
      gnuplot.raw('set xtics ("Sun" 1, "06:00" 360, "12:00" 720, "18:00" 1080, "Mon" 1440, "06:00" 1800, "12:00" 2160, "18:00" 2520, "Tue" 2880, "06:00" 3240, "12:00" 3600, "18:00" 3960,"Wed" 4320, "06:00" 4680, "12:00" 5040, "18:00" 5400, "Thu" 5760, "06:00" 6120, "12:00" 6480, "18:00" 6840, "Fri" 7200, "06:00" 7560, "12:00" 7920, "18:00" 8280, "Sat" 8640, "06:00" 9000, "12:00" 9360, "18:00" 9720)')
    end
    ]]--
    
      collectgarbage()
      sys.sleep(1) 
    end
  end
      
end

