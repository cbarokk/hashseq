
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits
local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)

local ExternalMinibatchLoader = {}
ExternalMinibatchLoader.__index = ExternalMinibatchLoader

theta_size = 8

function ExternalMinibatchLoader.create()
    local self = {}
    setmetatable(self, ExternalMinibatchLoader)

    print('reshaping tensor...')
    self.batch_size = opt.batch_size
    self.seq_length = opt.seq_length
    self.x = torch.DoubleTensor(self.batch_size, self.seq_length, theta_size) 
    self.e_x = torch.IntTensor(self.batch_size, self.seq_length, 1) 
    
    self.y = torch.DoubleTensor(self.batch_size, self.seq_length, 1) 
    self.e_y = torch.IntTensor(self.batch_size, self.seq_length, 1) 
    
    collectgarbage()
    return self
end

function ExternalMinibatchLoader.criterion()
  local crit1 = nn.ClassNLLCriterion()
  local crit2 = nn.ClassNLLCriterion()  
  return nn.ParallelCriterion():add(crit1, opt.theta_weight):add(crit2, opt.event_weight)
end

function ExternalMinibatchLoader.timestamp2theta(timestamp)
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

function ExternalMinibatchLoader:next_batch(queue)
  collectgarbage()
  self.x:zero()
  self.e_y:zero()
  self.e_x:zero()
  
  self.dates = {}
  self.batch = {}

  for b=1, self.batch_size do
    seq = redis_client:blpop(queue, 0)
    table.insert(self.batch, seq[2])
    
    local events = seq[2]:split(",")
      
    for t=1, #events do
      local words = events[t]:split("-")
      local e = tonumber(words[2])

      local timestamp = tonumber(words[1])
      table.insert(self.dates, timestamp)

      local theta, date = ExternalMinibatchLoader.timestamp2theta(timestamp)
      
      if t < #events then
        self.x[b][t]:sub(1,theta_size):copy(theta)
        self.e_x[b][t] = e
      end
      
      if t > 1 then
        local week_mins = date['min'] + 60*date['hour'] + 60*24*(date['wday']-1) + 1 -- +1 bcs index starts at one
        self.y[b][t-1] = week_mins 
        self.e_y[b][t-1] = e
      end
    end 
  end
  return self.x, self.y, self.e_x, self.e_y
end


return ExternalMinibatchLoader

