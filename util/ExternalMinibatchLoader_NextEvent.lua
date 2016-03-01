
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local LSTM_theta = require 'model.LSTM_theta'
local GRU_theta = require 'model.GRU_theta'
local RNN_theta = require 'model.RNN_theta'

local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)

local ExternalMinibatchLoader_NextEvent = {}
ExternalMinibatchLoader_NextEvent.__index = ExternalMinibatchLoader_NextEvent

theta_size = 8

function ExternalMinibatchLoader_NextEvent.create()
    local self = {}
    setmetatable(self, ExternalMinibatchLoader_NextEvent)

    print('reshaping tensor...')
    self.x = torch.DoubleTensor(opt.batch_size, opt.seq_length, theta_size) 
    self.e_x = torch.IntTensor(opt.batch_size, opt.seq_length, 1) 
    
    self.y = torch.DoubleTensor(opt.batch_size, opt.seq_length, 1) 
    self.w_y = torch.IntTensor(opt.batch_size, opt.seq_length, 1)
    self.e_y = torch.IntTensor(opt.batch_size, opt.seq_length, 1)
    
    collectgarbage()
    return self
end

function ExternalMinibatchLoader_NextEvent:create_rnn_units_and_criterion()
  print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
  local protos = {}
  if opt.rnn_unit == 'lstm' then
    protos.rnn = LSTM_theta.lstm()
  elseif opt.rnn_unit == 'gru' then
    protos.rnn = GRU_theta.gru()
  elseif opt.rnn_unit == 'rnn' then
    protos.rnn = RNN_theta.rnn()
  end
  local crit1 = nn.ClassNLLCriterion()
  local crit2 = nn.ClassNLLCriterion()  
  local crit3 = nn.ClassNLLCriterion()  
  protos.criterion = nn.ParallelCriterion():add(crit1, opt.theta_weight):add(crit2, opt.theta_weight):add(crit3, opt.event_weight)
  return protos
end

function ExternalMinibatchLoader_NextEvent.timestamp2theta(timestamp)
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
  
  return theta, tonumber(os.date( "%V", timestamp)), date
end

function ExternalMinibatchLoader_NextEvent:next_batch(queue)
  collectgarbage()
  self.x:zero()
  self.y:zero()
  self.e_y:zero()
  self.e_x:zero()
  self.w_y:zero()
  
  self.dates = {}
  self.batch = {}

  -- we check to see what are the events of interest. If none, default to plain old next event prediction.
  
  for b=1, opt.batch_size do
    seq = redis_client:blpop(queue, 0)
    table.insert(self.batch, seq[2])
    
    local events = seq[2]:split(",")
      
    for t=1, #events do
      local words = events[t]:split("-")
      local e = tonumber(words[2])

      local timestamp = tonumber(words[1])
      table.insert(self.dates, timestamp)

      local theta, weeknr, date = ExternalMinibatchLoader_NextEvent.timestamp2theta(timestamp)
      
      if t < #events then
        self.x[b][t]:sub(1,theta_size):copy(theta)
        self.e_x[b][t] = e
      end
      
      if t > 1 then
        local week_mins = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1) + 1 -- +1 bcs index starts at one
        self.y[b][t-1] = week_mins 
        self.w_y[b][t-1] = weeknr
        self.e_y[b][t-1] = e
      end
    end 
  end
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    self.x = self.x:float():cuda()
    self.e_x = self.e_x:float():cuda()
    self.y = self.y:float():cuda()
    self.e_y = self.e_y:float():cuda()
    self.w_y = self.w_y:float():cuda()

  end
end

function ExternalMinibatchLoader_NextEvent:feval()
    grad_params:zero()

    ------------------ get minibatch -------------------
    opt.loader:next_batch(opt.redis_queue)
    
    local x = opt.loader.x
    local y = opt.loader.y
    local e_x = opt.loader.e_x
    local e_y = opt.loader.e_y
    local w_y = opt.loader.w_y
    
    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0
    
    for t=1,opt.seq_length do
      clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones.rnn[t]:forward{x[{{},t,{}}], e_x[{{},t,{}}], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      predictions[t] = { lst[#lst- 2], lst[#lst- 1], lst[#lst]} --
      loss = loss + clones.criterion[t]:forward(predictions[t], 
        { y[{{}, t, {}}]:clone():view(opt.batch_size), w_y[{{}, t, {}}]:clone():view(opt.batch_size), e_y[{{}, t, {}}]:clone():view(opt.batch_size)})
    end
      
    loss = loss / opt.seq_length
    
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
      -- backprop through loss, and softmax/linear
      local doutput_t = clones.criterion[t]:backward(predictions[t], 
        { y[{{}, t, {}}]:clone():view(opt.batch_size), w_y[{{}, t, {}}]:clone():view(opt.batch_size), e_y[{{}, t, {}}]:clone():view(opt.batch_size)})
      table.insert(drnn_state[t], doutput_t[1])
      table.insert(drnn_state[t], doutput_t[2])
      table.insert(drnn_state[t], doutput_t[3])

      
      local dlst = clones.rnn[t]:backward({x[{{},t,{}}], e_x[{{},t,{}}], unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
        if k > 2 then -- k == 1 is gradient on x, which we dont need
          -- note we do k-1 because first item is dembeddings, and then follow the 
          -- derivatives of the state, starting at index 2. I know...
          drnn_state[t-1][k-2] = v
        end
      end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    
    return loss, grad_params
end


return ExternalMinibatchLoader_NextEvent

