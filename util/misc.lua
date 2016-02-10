
-- misc utilities

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
  
  
  
  --  sub(week_mins-self.halfW, week_mins+self.halfW):add(self.w)
  --      self.y[b][t-1]:add(probs:sub(1, 60*24*7))
  --      self.y[b][t-1]:add(probs:sub(60*24*7+1, 60*24*7*2))
  --      self.y[b][t-1]:add(probs:sub(60*24*7*2+1,-1)) ]]--


