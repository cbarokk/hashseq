
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local LSTM_theta = require 'model.LSTM_theta'
local GRU_theta = require 'model.GRU_theta'
local RNN_theta = require 'model.RNN_theta'

local ExternalMinibatchLoader_NextEvent = {}
ExternalMinibatchLoader_NextEvent.__index = ExternalMinibatchLoader_NextEvent

function ExternalMinibatchLoader_NextEvent.create()
    local self = {}
    setmetatable(self, ExternalMinibatchLoader_NextEvent)
    
    fetch_events()
    print('There are ' .. opt.num_events-1 .. ' unique events in redis')
    
    
    collectgarbage()
    return self
end

function ExternalMinibatchLoader_NextEvent:create_rnn_units_and_criterion()
  if string.len(opt.init_from) == 0 then
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
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
  end
end


function ExternalMinibatchLoader_NextEvent:next_batch(queue)
  collectgarbage()
  
  local seqs = sequence_providers[opt.sequence_provider]()
  self.batch = seqs
  --print ("seqs", seqs)
  --local num_events = #seqs[1]:split(",")
  local num_events = #seqs[1]
  
  local x = torch.DoubleTensor(num_events-1, #seqs, theta_size):zero()
  local e_x = torch.IntTensor(num_events-1, #seqs):zero()
  
  local y = torch.IntTensor(num_events-1, #seqs):zero()
  local e_y = torch.IntTensor(num_events-1, #seqs):fill(1)

  for s=1,#seqs do
    --local events = seqs[s]:split(",")
    local events = seqs[s]
    
    for t= 1, #events do
      --print ("events[t]", events[t])
      local e = tonumber(events[t][2])

      local timestamp = tonumber(events[t][1])
      local theta, date = timestamp2theta(timestamp, theta_size)

      if t < #events then
        x[t][s]:sub(1,theta_size):copy(theta)
        e_x[t][s] = e
      end
      
      local min_of_the_week = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1)
      local time_slot = math.floor(min_of_the_week/10080*opt.num_time_slots) + 1 -- +1 bcs index starts at 1 in lua 
      
      if t < #events then
        x[t][s]:sub(1,theta_size):copy(theta)
        e_x[t][s] = e
      end
      
      if t > 1 then
        e_y[t-1][s] = e
        y[t-1][s] = time_slot

        if e > 1 then
          for i=t-2,1,-1 do
            if e_y[i][s] == 1 then
              e_y[i][s] = e
              y[i][s] = time_slot
            else
              break
            end
          end
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
  end
  return x, y, e_x, e_y
end

function ExternalMinibatchLoader_NextEvent:feval()
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, e_x, e_y = opt.loader:next_batch()
    
    local batch_size = x:size(2)
    local len_seq = x:size(1)
    
    local rnn_state = {[0] = init_state_global}
    local predictions = {}
    local loss = 0
    
    for t=1,len_seq do
      clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones.rnn[t]:forward{x[t], e_x[t], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      predictions[t] = {lst[#lst- 1], lst[#lst]} --
      
      local e_y_probs = lst[#lst]:clone():exp():double()
    
      local _, idx_max  = torch.max(e_y_probs, 2)
      print ("idx_max", idx_max)
      print ("e_x[t]", opt.event_inv_mapping[e_x[t][1]], opt.event_inv_mapping[idx_max[1][1]])
  
      
      loss = loss + clones.criterion[t]:forward(predictions[t], { y[t], e_y[t]})
    end
      
    loss = loss / len_seq
    
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[len_seq] = clone_list(init_state, true)} -- true also zeros the clones
    for t=len_seq,1,-1 do
      -- backprop through loss, and softmax/linear
      local doutput_t = clones.criterion[t]:backward(predictions[t], { y[t], e_y[t]})
      table.insert(drnn_state[t], doutput_t[1])
      table.insert(drnn_state[t], doutput_t[2])
      
      local dlst = clones.rnn[t]:backward({x[t], e_x[t], unpack(rnn_state[t-1])}, drnn_state[t])
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
    grad_params:div(len_seq) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    
    return loss, grad_params
end


return ExternalMinibatchLoader_NextEvent

