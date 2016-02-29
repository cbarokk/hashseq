
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local redis = require 'redis'
local redis_client = redis.connect()

local ExternalMinibatchLoader_past = {}
ExternalMinibatchLoader_past.__index = ExternalMinibatchLoader_past

theta_size = 8

function ExternalMinibatchLoader_past.create()
    local self = {}
    setmetatable(self, ExternalMinibatchLoader_past)
    
    print('reshaping tensor...')
    self.x = torch.DoubleTensor(opt.batch_size, opt.seq_length, theta_size) 
    self.e_x = torch.IntTensor(opt.batch_size, opt.seq_length, 1) 
    
    self.y = torch.DoubleTensor(opt.batch_size, opt.seq_length, opt.num_time_slots) 
    --self.y = torch.Doub leTensor(opt.batch_size, opt.seq_length, 1) 
    
    --self.e_y = torch.IntTensor(opt.batch_size, opt.seq_length, 1) 
    self.e_y = torch.DoubleTensor(opt.batch_size, opt.seq_length, opt.num_events) 
    
    assert(opt.redis_queue:len()>0, 'You forgot to specify the redis queue name.')
    print('Reading from redis queue: ' .. opt.redis_queue)
    collectgarbage()
    return self
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
      local time_slot = math.floor(min_of_the_week/10080*opt.num_time_slots) +1 -- +1 bcs index starts at 1 in lua 
      
      self.y[b]:sub(t, -1, time_slot, time_slot):fill(t*t)
      self.y[b][t]:div(self.y[b][t]:sum())
      self.e_y[b]:sub(t, -1, e, e):add(1)
      self.e_y[b][t]:div(self.e_y[b][t]:sum())
    end 
  end
  return self.x, self.y, self.e_x, self.e_y
  
end


return ExternalMinibatchLoader_past

