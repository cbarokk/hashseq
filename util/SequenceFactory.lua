local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)
require 'donkey'

function factory_initialization()
  sequence_providers = {}
  table.insert(sequence_providers, fetch_from_redis_q)
  table.insert(sequence_providers, synthetic_seqs)
  table.insert(sequence_providers, fetch_from_disk)
  
  
  init_functions = {}
  table.insert(init_functions, redis_init)
  table.insert(init_functions, synthetic_init)
  table.insert(init_functions, disk_init)
  
  init_functions[opt.sequence_provider]()
end


function redis_init()
  opt.train_queue = opt.redis_prefix .. '-train'  
end

function disk_init()
  loadEvents("2013.csv_events", redis_client)
  loadSourceNames("2013.csv_eoi/")  
end


function synthetic_init()
  redis_client:set('synthetic:period', 100)
  redis_client:set('synthetic:noise', 0.25)
  redis_client:set('synthetic:len_seq', 30)
  redis_client:set('synthetic:batch_size', opt.batch_size)
  
  redis_client:del(opt.redis_prefix .. '-events')
  redis_client:del(opt.redis_prefix .. '-events-of-interest')
  
  local eoi = 201
  for i=2, eoi do
    redis_client:sadd(opt.redis_prefix .. '-events', i)
  end
  redis_client:sadd(opt.redis_prefix .. '-events-of-interest', eoi)

end


function fetch_next_seqs(f, ...) return f(...) end

function fetch_from_redis_q()
  local batch = redis_client:blpop(opt.train_queue, 0)
  return batch[2]:split(";")
end
  
  
  
function fetch_from_disk()
  --[[
  local timer = torch.Timer()
  local batch = loadBatch(opt.batch_size, 20)
  local time = timer:time().real
  print ("fetch batch time: %.2fs", time)
  return batch
  ]]--
  return loadBatch(opt.batch_size, opt.seq_len)
end

  
  
  
  
function synthetic_seqs()
  local seqs = {};
  local period = math.random(redis_client:get('synthetic:period'))*60
  --local period = redis_client:get('synthetic:period')
  local noise = redis_client:get('synthetic:noise')
  local len_seq = tonumber(redis_client:get('synthetic:len_seq'))
  local batch_size = redis_client:get('synthetic:batch_size')

  
  for i=1, batch_size do
    local pos = opt.num_time_slots + math.random(1000)
    local seq = {}
    
    local rem_balance = math.random(200)
    for j=1, len_seq do
      local event = {}
        event[1] = pos
        
      if rem_balance < 10 then
        event[2] = 201
        --seq = seq .. pos .. "-" .. "201" -- top-up
        rem_balance = 200
      else
          event[2] = rem_balance
        --seq = seq .. pos .. "-" .. rem_balance
      end
      local decr = math.random(10)
      rem_balance = rem_balance - decr
      pos = pos + period + math.random(math.ceil(period*noise))
      --if j < len_seq then seq = seq .. "," end
      table.insert(seq, event)

    end
    --[[
    for j=1, len_seq do
      if j == math.ceil(len_seq*2/3) then 
        seq = seq .. pos .. "-2" 
      else
        seq = seq .. pos .. "-1" 
      end
      pos = pos + period + math.random(math.ceil(period*noise))
      
      if j < len_seq then seq = seq .. "," end
    end
    ]]--
    
    --print ("seq", seq)
    table.insert(seqs, seq)
  end
 return seqs
end



