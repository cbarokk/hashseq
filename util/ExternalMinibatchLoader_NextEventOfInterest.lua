-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local LSTM_theta = require 'model.LSTM_theta'
local GRU_theta = require 'model.GRU_theta'
local RNN_theta = require 'model.RNN_theta'

local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)

local ExternalMinibatchLoader_NextEventOfInterest = {}
ExternalMinibatchLoader_NextEventOfInterest.__index = ExternalMinibatchLoader_NextEventOfInterest

theta_size = 8

function ExternalMinibatchLoader_NextEventOfInterest.create()
    local self = {}
    setmetatable(self, ExternalMinibatchLoader_NextEventOfInterest)
    collectgarbage()
    return self
end

function ExternalMinibatchLoader_NextEventOfInterest:create_rnn_units_and_criterion()
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

function ExternalMinibatchLoader_NextEventOfInterest.timestamp2theta(timestamp)
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
  theta[6] = math.sin((2*math.pi)/24*hour) --sin_hour
      
  local day = date['wday']-1
  theta[7] = math.cos((2*math.pi)/7*day) --cos_day
  theta[8] = math.sin((2*math.pi)/7*day) --sin_day
  
  return theta, tonumber(os.date( "%V", timestamp)), date
end

function in_table(tbl, item)
   for key, value in pairs(tbl) do
      if value == item then return key end
   end
   return false
end

function ExternalMinibatchLoader_NextEventOfInterest:next_batch(queue, interesting_list)
   collectgarbage()
   local dates = {}

   local events_of_interest = redis_client:lrange(interesting_list, 0, -1)
   for i, value in pairs(events_of_interest) do
      events_of_interest[i] = tonumber(value)
   end

   local x = torch.DoubleTensor(opt.batch_size, opt.seq_length, theta_size)
   local e_x = torch.IntTensor(opt.batch_size, opt.seq_length, 1) 
   
   local y = torch.DoubleTensor(opt.batch_size, opt.seq_length, 1) 
   local w_y = torch.IntTensor(opt.batch_size, opt.seq_length, 1)
   local e_y = torch.IntTensor(opt.batch_size, opt.seq_length, 1)
   
   for b=1, opt.batch_size do
      local seq = redis_client:blpop(queue, 0)
      local events = seq[2]:split(",")
      
      local interesting_week_mins = false
      local interesting_weeknr = false
      local interesting_event = false

      assert(#events == opt.seq_length+1, 'Until dynamic seq_length is fixed, load redis with sequences of opt.seq_length+1')
      
      for t=#events,1,-1 do 
	 local words = events[t]:split("-")
	 local e = tonumber(words[2])

	 local timestamp = tonumber(words[1])
	 table.insert(dates, timestamp)

	 local theta, weeknr, date = ExternalMinibatchLoader_NextEventOfInterest.timestamp2theta(timestamp)
	 local week_mins = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1) + 1 -- +1 bcs index starts at one
	 
	 if t < #events then
	    x[b][t]:sub(1,theta_size):copy(theta)
	    e_x[b][t] = e
	 end

	 if in_table(events_of_interest, e) then
	    interesting_week_mins = week_mins
	    interesting_weeknr = weeknr
	    interesting_event = in_table(events_of_interest, e) + 1 -- 1 reserved for 'not interesting'
	 end
	 
	 if t > 1 then
	    if interesting_event then
	       y[b][t-1] = interesting_week_mins 
	       w_y[b][t-1] = interesting_weeknr
	       e_y[b][t-1] = interesting_event
	    else
	       y[b][t-1] = week_mins 
	       w_y[b][t-1] = weeknr
	       e_y[b][t-1] = 1 -- not interesting
	    end
	 end
      end 
   end
   if opt.gpuid >= 0 then -- ship the input arrays to GPU
      -- have to convert to float because integers can't be cuda()'d
      x = x:float():cuda()
      e_x = e_x:float():cuda()
      y = y:float():cuda()
      e_y = e_y:float():cuda()
      w_y = w_y:float():cuda()
   end

   return x, y, e_x, e_y, w_y, dates
end

function ExternalMinibatchLoader_NextEventOfInterest:feval()
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, e_x, e_y, w_y, _ = opt.loader:next_batch(opt.redis_queue, opt.redis_interest_list)

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


return ExternalMinibatchLoader_NextEventOfInterest

