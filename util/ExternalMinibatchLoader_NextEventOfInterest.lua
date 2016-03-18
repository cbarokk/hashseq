-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local LSTM_theta = require 'model.LSTM_theta'
local GRU_theta = require 'model.GRU_theta'
local RNN_theta = require 'model.RNN_theta'
require 'util.misc'


local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)

local ExternalMinibatchLoader_NextEventOfInterest = {}
ExternalMinibatchLoader_NextEventOfInterest.__index = ExternalMinibatchLoader_NextEventOfInterest

theta_size = 8

function ExternalMinibatchLoader_NextEventOfInterest.create()
  local self = {}
  setmetatable(self, ExternalMinibatchLoader_NextEventOfInterest)
  
  opt.num_eoi = redis_client:scard(opt.redis_prefix .. '-events-of-interest')
  print ("opt.num_eoi", opt.num_eoi)
  assert(opt.num_eoi > 0, 'You must specify interesting events.')
  opt.num_eoi = opt.num_eoi + 1 -- add "not interesting"

  opt.num_events = redis_client:hlen(opt.redis_prefix .. '-events') + 1
  
  local interesting_list = opt.redis_prefix .. '-events-of-interest'
  local tmp = redis_client:smembers(interesting_list)
   
  self.events_of_interest = {}
  for id, eoi in pairs(tmp) do
    self.events_of_interest[tonumber(eoi)] = id+1 -- 1 reserved to "not intersting"
  end
  
  print('There are ' .. opt.num_events-1 .. ' unique events in redis, of which ' .. opt.num_eoi-1 .. ' are interesting')
  opt.time_slot_size = opt.horizon/opt.num_time_slots
  
  
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
  protos.criterion = nn.ParallelCriterion():add(crit1, opt.theta_weight):add(crit2, opt.event_weight)
  return protos
end


function ExternalMinibatchLoader_NextEventOfInterest:next_batch()
  collectgarbage()

--  local seqs = fetch_next_seqs(sequence_providers[opt.sequence_provider], opt)
  local seqs = sequence_providers[opt.sequence_provider]()
    
  local num_events = #seqs[1]:split(",")
   
  local x = torch.DoubleTensor(#seqs, num_events-1, theta_size):zero()
  local e_x = torch.IntTensor(#seqs,  num_events-1, 1):zero()
  local y = torch.IntTensor(#seqs, num_events-1, 1):zero()
  local e_y = torch.IntTensor(#seqs, num_events-1, 1):fill(1)
  
  for s=1,#seqs do
    local events = seqs[s]:split(",")
  
    for t= 1, #events do
      local words = events[t]:split("-")
      local e = tonumber(words[2])

      local timestamp = tonumber(words[1])
      local theta = timestamp2theta(timestamp, theta_size)

      if t < #events then
        x[s][t]:sub(1,theta_size):copy(theta)
        e_x[s][t] = e
        y[s][t] = timestamp --remember what is current timestamp
      end
      
      if t > 1 then
        if self.events_of_interest[e] then
          e_y[s]:sub(1,t-1):apply(function(x)
            if x == 1 then --dont overwrite previous eoi
              return self.events_of_interest[e] 
            end 
          end)
        
          y[s]:sub(1,t):apply(function(x)
            if x > opt.num_time_slots then --dont overwrite previous eoi
              return math.min( math.ceil((timestamp - x)/opt.time_slot_size)+1, opt.num_time_slots) 
            end     
          end)
        end
      end
    end
    y[s]:apply(function(x)
      return math.min(x, opt.num_time_slots) -- overwrite initial timestamps when no eoi founds
    end)
    
  end
  
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
    x = x:float():cuda()
    e_x = e_x:float():cuda()
    y = y:float():cuda()
    e_y = e_y:float():cuda()
  end
  --print ("y", y)
  return x, y, e_x, e_y
end

function display_uncertainty(y_pred, y_fasit)
      local y_probs = y_pred:clone():exp():double()
      local y_truth = y_fasit:clone()
      
      local s=0
      local left_cumul = y_probs:clone():mean(1)
      left_cumul:apply(function(x)
          s = s + x
          return s
        end)
      
      local right_cumul = y_probs:clone():mean(1)
      s=0
      for i=opt.num_time_slots, 1, -1 do
        s = s + right_cumul[1][i]
        right_cumul[1][i] = s
      end
      local y_truth_dist = left_cumul:clone():zero()
      
      y_truth:apply(function(x)
          y_truth_dist[1][x] = y_truth_dist[1][x] + 1 
        end)
      
      y_truth_dist:div(y_truth_dist:sum())
      
      gnuplot.axis{1, opt.num_time_slots,0,1}
      gnuplot.figure('uncertainty')
      gnuplot.title('uncertainty ' .. opt.info)
      gnuplot.plot({"start", left_cumul[1],'-'}, {"y", y_truth_dist[1], '|'}, {"end", right_cumul[1],'-'})
      
    end

function ExternalMinibatchLoader_NextEventOfInterest:feval()
    grad_params:zero()

    ------------------ get minibatch -------------------
    opt.loader:next_batch()
    
    local x, y, e_x, e_y = opt.loader:next_batch(opt.loader.train_queue)
    
    local batch_size = x:size(1)
    local len_seq = x:size(2)
    
    local init_state = get_init_state(batch_size)
    local init_state_global = get_init_state_global(batch_size)
    local clones = get_clones(len_seq)

    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0

    for t=1,len_seq do
      clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones.rnn[t]:forward{x[{{},t,{}}], e_x[{{},t,{}}], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      predictions[t] = {lst[#lst- 1], lst[#lst]} --
      
      --display_uncertainty(lst[#lst-1], y[{{}, t, {}}])
      
      loss = loss + clones.criterion[t]:forward(predictions[t], 
        { y[{{}, t, {}}]:clone():view(batch_size), e_y[{{}, t, {}}]:clone():view(batch_size)})
    end
      
    loss = loss / len_seq
    
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[len_seq] = clone_list(init_state, true)} -- true also zeros the clones
    for t=len_seq,1,-1 do
      -- backprop through loss, and softmax/linear
      local doutput_t = clones.criterion[t]:backward(predictions[t], 
        {y[{{}, t, {}}]:clone():view(batch_size) , e_y[{{}, t, {}}]:clone():view(batch_size)})
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
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(len_seq) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  
    return loss, grad_params
end


return ExternalMinibatchLoader_NextEventOfInterest

