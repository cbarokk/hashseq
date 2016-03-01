
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local LSTM_theta_past = require 'model.LSTM_theta_past'
local GRU_theta_past = require 'model.GRU_theta_past'
local RNN_theta_past = require 'model.RNN_theta_past'
require 'util.QuadraticPenalty'

local redis = require 'redis'
local redis_client = redis.connect()

local ExternalMinibatchLoader_past = {}
ExternalMinibatchLoader_past.__index = ExternalMinibatchLoader_past

theta_size = 8

function ExternalMinibatchLoader_past.create()
    
    -- extract autoencoder_shape in opt
    opt.encoder = {opt.rnn_size, unpack(opt.encoder_shape:split(","))}
    for i =1, #opt.encoder do
      opt.encoder[i] = tonumber(opt.encoder[i])
    end
    
    
    local self = {}
    setmetatable(self, ExternalMinibatchLoader_past)
    
    print('reshaping tensor...')
    self.x = torch.DoubleTensor(opt.batch_size, opt.seq_length, theta_size) 
    self.e_x = torch.IntTensor(opt.batch_size, opt.seq_length, 1) 
    
    self.y = torch.DoubleTensor(opt.batch_size, opt.seq_length, opt.num_weekly_slots) 
    --self.y = torch.Doub leTensor(opt.batch_size, opt.seq_length, 1) 
    
    --self.e_y = torch.IntTensor(opt.batch_size, opt.seq_length, 1) 
    self.e_y = torch.DoubleTensor(opt.batch_size, opt.seq_length, opt.num_events) 
    
    assert(opt.redis_queue:len()>0, 'You forgot to specify the redis queue name.')
    print('Reading from redis queue: ' .. opt.redis_queue)
    collectgarbage()
    return self
end

function ExternalMinibatchLoader_past:create_rnn_units_and_criterion()
  print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
  local protos = {}
  if opt.rnn_model == 'lstm' then
    protos.rnn = LSTM_theta_past.lstm()
  elseif opt.rnn_unit == 'gru' then
    protos.rnn = GRU_theta_past.gru()
  elseif opt.rnn_unit == 'rnn' then
    protos.rnn = RNN_theta_past.rnn()
  end
  local crit1 = nn.DistKLDivCriterion()
  local crit2 = nn.DistKLDivCriterion()
  protos.criterion = nn.ParallelCriterion():add(crit1, opt.theta_weight):add(crit2, opt.event_weight)
   
  return protos
end

function ExternalMinibatchLoader_past.timestamp2theta(timestamp)
  local theta = torch.DoubleTensor(theta_size):fill(0)
  
  local date = os.date("*t", timestamp)
  
  local sec = date['sec']
  theta[1] = math.cos((2*math.pi)/60*sec) --cos_sec
  theta[2] = math.sin((2*math.pi)/60*sec) --sin_sec
  
  local min = date['min']
  theta[3] = math.cos((2*math.pi)/60*min) --cos_min
  theta[4] = math.sin((2*math.pi)/60*min) --sin_min
      
  local hour = date['hour']
  theta[5] = math.cos((2*math.pi)/24*hour) --cos_hour
  theta[6] = math.sin((2*math.pi)/24*hour) -- sinhour
      
  local day = date['wday']-1
  theta[7] = math.cos((2*math.pi)/7*day) --cos_day
  theta[8] = math.sin((2*math.pi)/7*day) --sin_day
  
  return theta, date
end

function ExternalMinibatchLoader_past:next_batch()
  collectgarbage()
  self.y:zero()
  self.x:zero()
  self.e_y:zero()
  self.e_x:zero()
  
  self.batch = {}
  
  for b=1, opt.batch_size do
    seq = redis_client:blpop(opt.redis_queue, 0)
    table.insert(self.batch, seq[2])
    local events = seq[2]:split(",")
    for t=1, #events do
      local words = events[t]:split("-")
      local e = tonumber(words[2])
      local timestamp = tonumber(words[1])
      local theta, date = ExternalMinibatchLoader_past.timestamp2theta(timestamp)
      
      self.x[b][t]:copy(theta)
      self.e_x[b][t] = e
      
      local min_of_the_week = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1) 
      local time_slot = math.floor(min_of_the_week/10080*opt.num_weekly_slots) +1 -- +1 bcs index starts at 1 in lua 
      
      self.y[b]:sub(t, -1, time_slot, time_slot):fill(t*t)
      self.y[b][t]:div(self.y[b][t]:sum())
      self.e_y[b]:sub(t, -1, e, e):add(1)
      self.e_y[b][t]:div(self.e_y[b][t]:sum())
    end 
  end
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    self.x = self.x:float():cuda()
    self.e_x = self.e_x:float():cuda()
    self.y = self.y:float():cuda()
    self.e_y = self.e_y:float():cuda()
  end
end

function ExternalMinibatchLoader_past:feval()
    grad_params:zero()
    opt.loader:next_batch(opt.redis_queue)
    local x = opt.loader.x
    local y = opt.loader.y
    local e_x = opt.loader.e_x
    local e_y = opt.loader.e_y
    
    local hashcodes = torch.DoubleTensor(opt.batch_size, opt.seq_length, opt.encoder[#opt.encoder]):fill(0)
    local zero_codes = torch.DoubleTensor(opt.batch_size, opt.encoder[#opt.encoder]):fill(0)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
      -- have to convert to float because integers can't be cuda()'d
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


return ExternalMinibatchLoader_past

