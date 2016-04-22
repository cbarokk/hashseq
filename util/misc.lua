
-- misc utilities
require 'gnuplot'

local redis = require 'redis'
redis_client = redis.connect('127.0.0.1', 6379)


function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function get_size_softmax_layer(forwardnodes, annotation)
  for _,node in ipairs(forwardnodes) do
     if node.data.annotations.name == annotation then
      return node.data.module.weight:size(1)
    end
  end
end

function Welch(N)
  local w = torch.Tensor(N)
  local i = -1
  local half = (N-1)/2
  w:apply(function()
    i = i + 1   
    return 1-(math.pow((i-half)/half,2))
  end)
  w:div(w:sum())
  return w
end

function smooth_probs(probs, N)
  local w = Welch(N+1)
  local left_half_w = w:sub(1,N/2+1):div(w:sub(1,N/2+1):sum())
  
  local tmp = torch.cat({probs[1], probs[1]})
  local i=10080-1
  probs[1]:apply(function(x)
      i = i+1
      local window = tmp:sub(i-N/2, i):clone():cmul(left_half_w):sum() --smooth left only
      return window
  end)
end
  
function normal_equations(X, y)
  return torch.inverse(X:t()*X)*X:t()*y
end


theta_size = 8

function timestamp2theta(timestamp)
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
      
  local weekday = date['wday']-1
  theta[7] = math.cos((2*math.pi)/7*weekday) --cos_weekday
  theta[8] = math.sin((2*math.pi)/7*weekday) --sin_weekday
  --[[
  local monthday = date['day']
  theta[9] = math.cos((2*math.pi)/31*monthday) --cos_monthday
  theta[10] = math.sin((2*math.pi)/31*monthday) --sin_monthday

  local month = date['month']
  theta[11] = math.cos((2*math.pi)/12*month) --cos_month
  theta[12] = math.sin((2*math.pi)/12*month) --sin_month
  
  local yearday = date['yday']
  theta[13] = math.cos((2*math.pi)/365*yearday) --cos_yearday
  theta[14] = math.sin((2*math.pi)/365*yearday) --sin_yearday
]]--
  return theta, date
end

function PCA(X)
  local mean = torch.mean(X, 1) -- 1 x n
  local m = X:size(1)
  local Xm = X - torch.ones(m, 1) * mean
  
  Xm:div(math.sqrt(m - 1))
  local v,s,_ = torch.svd(Xm:t())
  
  s:cmul(s) -- n

  --[[
  -- v: eigenvectors, s: eigenvalues of covariance matrix
  local b = sys.COLORS.blue; n = sys.COLORS.none
  print(b .. 'eigenvectors (columns):' .. n); print(v)
  print(b .. 'eigenvalues (power/variance):' .. n); print(s)
  print(b .. 'sqrt of the above (energy/std):' .. n); print(torch.sqrt(s))
  ]]--
  
  local vv = v * torch.diag(torch.sqrt(s))
  vv = torch.cat(torch.ones(2,1) * mean, vv:t())
  return vv
  
end

function display_uncertainty(y_pred, y_fasit, e_y_pred, e_y_fasit)
      local y_probs = y_pred:clone():exp():double()[1]
      local y_truth = y_fasit:clone()[1]
      print ("y_probs", y_probs, "y_truth", y_truth)
      
      
      local s=0
      --local left_cumul = y_probs:clone():mean(1)
      local left_cumul = y_probs:clone()
      
      left_cumul:apply(function(x)
          s = s + x
          return s
        end)
      
      --local right_cumul = y_probs:clone():mean(1)
      local right_cumul = y_probs:clone()
      s=0
      for i=y_probs:size()[1], 1, -1 do
        s = s + right_cumul[i]
        right_cumul[i] = s
      end
      local y_truth_dist = left_cumul:clone():zero()
      
      y_truth_dist[y_truth] = 1 
        
      y_truth_dist:div(y_truth_dist:sum())
      
      
      gnuplot.figure('horizon')
      gnuplot.title('horizon ')
      gnuplot.axis{1, y_probs:size()[1],0,1}
      gnuplot.plot({"cpdf (left)", left_cumul,'-'}, {"y", y_truth_dist, '|'}, {"bets", y_probs, '-'}, {"cpdf (right)", right_cumul,'-'})
      --gnuplot.raw('set xtics ("Sun" 1, "06:00" 360, "12:00" 720, "18:00" 1080, "Mon" 1440, "06:00" 1800, "12:00" 2160, "18:00" 2520, "Tue" 2880, "06:00" 3240, "12:00" 3600, "18:00" 3960,"Wed" 4320, "06:00" 4680, "12:00" 5040, "18:00" 5400, "Thu" 5760, "06:00" 6120, "12:00" 6480, "18:00" 6840, "Fri" 7200, "06:00" 7560, "12:00" 7920, "18:00" 8280, "Sat" 8640, "06:00" 9000, "12:00" 9360, "18:00" 9720)')

      
      
      local e_y_probs = e_y_pred:clone():exp():double()
      local e_y_truth = e_y_probs:clone():zero()[1]
      
      e_y_truth[e_y_fasit[1] ] = e_y_probs:max()
      
      gnuplot.axis{1, e_y_probs:size()[1],0,e_y_probs:max()}
      
      gnuplot.figure('next event')
      gnuplot.title('next event ')
      gnuplot.plot({"e_y", e_y_truth, '|'}, {"bets", e_y_probs[1], '-'})
      
      
    end
    
    
  function fetch_events()
    if string.len(opt.init_from) > 0 then --
      opt.num_events = checkpoint.opt.num_events
      opt.event_mapping = checkpoint.opt.event_mapping
      opt.event_inv_mapping = checkpoint.opt.event_inv_mapping
      opt.time_slot_size = checkpoint.opt.time_slot_size
      opt.horizon = checkpoint.opt.horizon
      opt.num_time_slots = checkpoint.opt.num_time_slots
    else
      opt.num_events = redis_client:scard(opt.redis_prefix .. '-events') + 1
      local tmp = redis_client:smembers(opt.redis_prefix .. '-events')
      opt.num_events = # tmp + 1
      opt.event_mapping = {}
      opt.event_inv_mapping = {}

      for id, event_id in pairs(tmp) do
        opt.event_mapping[event_id] = id+1 -- 1 reserved to "not intersting"
        opt.event_inv_mapping[id+1] = event_id
      end
      opt.event_inv_mapping[1] = "PROBE"
      opt.time_slot_size = opt.horizon/opt.num_time_slots
    end

  end
    