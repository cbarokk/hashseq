local redis = require 'redis'
local redis_client = redis.connect('127.0.0.1', 6379)


function factory_initialization()
  sequence_providers = {}
  table.insert(sequence_providers, fetch_from_redis_q)
  table.insert(sequence_providers, synthetic_seqs)
  
  init_functions = {}
  table.insert(init_functions, redis_init)
  table.insert(init_functions, synthetic_init)
  
  init_functions[opt.sequence_provider]()
end


function redis_init()
  opt.train_queue = opt.redis_prefix .. '-train'  
end

function synthetic_init()
  redis_client:set('synthetic:period', 10)
  redis_client:set('synthetic:noise', 0)
  redis_client:set('synthetic:len_seq', 25)
  redis_client:set('synthetic:batch_size', 10)
  redis_client:hset(opt.redis_prefix .. '-events', 'ali', 2)
  redis_client:hset(opt.redis_prefix .. '-events', 'baba', 3)
  redis_client:sadd(opt.redis_prefix .. '-events-of-interest', 3)

end


function fetch_next_seqs(f, ...) return f(...) end

function fetch_from_redis_q()
  local batch = redis_client:blpop(opt.train_queue, 0)
  return batch[2]:split(";")
end
  
  
function synthetic_seqs()
  local seqs = {};
  local period = redis_client:get('synthetic:period')
  local noise = redis_client:get('synthetic:noise')
  local len_seq = tonumber(redis_client:get('synthetic:len_seq'))
  local batch_size = redis_client:get('synthetic:batch_size')

  

  for i=1, batch_size do
    local pos = opt.num_time_slots + math.random(1000)
    local seq = ""
    for j=1, len_seq do
      if j == math.ceil(len_seq*2/3) then 
        seq = seq .. pos .. "-3" 
      else
        seq = seq .. pos .. "-2" 
      end
      pos = pos + period + math.random(math.ceil(period*noise))
      
      if j < len_seq then seq = seq .. "," end
    end
  table.insert(seqs, seq)
  end
 return seqs
end



