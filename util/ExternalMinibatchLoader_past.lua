
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)

local ExternalMinibatchLoader_past = {}
ExternalMinibatchLoader_past.__index = ExternalMinibatchLoader_past

theta_size = 8

function ExternalMinibatchLoader_past.create(batch_size, seq_length, num_events, queue_name)
    local self = {}
    setmetatable(self, ExternalMinibatchLoader_past)
    
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.x = torch.DoubleTensor(self.batch_size, self.seq_length, theta_size) 
    self.e_x = torch.IntTensor(self.batch_size, self.seq_length, 1) 
    
    self.y = torch.DoubleTensor(self.batch_size, self.seq_length, 60*24*7) 
    --self.y = torch.Doub leTensor(self.batch_size, self.seq_length, 1) 
    
    --self.e_y = torch.IntTensor(self.batch_size, self.seq_length, 1) 
    self.e_y = torch.DoubleTensor(self.batch_size, seq_length, num_events) 
    
    self.queue_name = queue_name
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
  
  for b=1, self.batch_size do
    seq = redis_client:blpop(self.queue_name, 0)
    table.insert(self.batch, seq[2])
    local events = seq[2]:split(",")
    
    for t=1, #events-1 do
      local words = events[t]:split("-")
      local e = tonumber(words[2])
      local timestamp = tonumber(words[1])
      local theta, date = ExternalMinibatchLoader_past.timestamp2theta(timestamp)
      
      self.x[b][t]:copy(theta)
      self.e_x[b][t] = e
      local week_mins = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1) + 1 -- +1 bcs index starts at one
      self.y[b]:sub(t, -1, week_mins, week_mins):add(t)
      self.y[b][t]:div(self.y[b][t]:sum())
      self.e_y[b]:sub(t, -1, e, e):add(1)
      self.e_y[b][t]:div(self.e_y[b][t]:sum())
    end 
  end
  return self.x, self.y, self.e_x, self.e_y
  
end


return ExternalMinibatchLoader_past

