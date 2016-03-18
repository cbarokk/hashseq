
-- misc utilities
require 'gnuplot'

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

function timestamp2theta(timestamp, theta_size)
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
