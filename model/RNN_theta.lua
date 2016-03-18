local RNN_theta = {}

function RNN_theta.rnn()
  local rnn_size = opt.rnn_size
  local num_time_slots = opt.num_time_slots
  local num_layers = opt.num_layers
  local num_eoi = opt.num_eoi
  local num_events = opt.num_events
  local dropout = opt.dropout or 0
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- e_x

  for L = 1,num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  
  local outputs = {}
  for L = 1,num_layers do
    
    local prev_h = inputs[L+2]
    if L == 1 then 
      local theta_x = inputs[1]
      local e_x = inputs[2]
      local embedings_size = 100
      local e_embedings = nn.LookupTable(num_events, embedings_size)(e_x):annotate{name='emb_e'}
      e_embedings = nn.Reshape(embedings_size)(e_embedings)
      
      x = nn.JoinTable(2)({theta_x, e_embedings}) 
      input_size_L = theta_size+embedings_size
      --x = OneHot(input_size)(inputs[1])
      --input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})
    table.insert(outputs, next_h)
  end
  local layer = outputs[#outputs]
  
  if dropout > 0 then layer = nn.Dropout(dropout)(layer) end
  
  local time_pred = nn.Linear(rnn_size, num_time_slots)(layer)
  time_pred = nn.LogSoftMax()(time_pred)
  table.insert(outputs, time_pred)
  
  local event_pred = nn.Linear(rnn_size, num_eoi)(layer)
  event_pred = nn.LogSoftMax()(event_pred)
  table.insert(outputs, event_pred)
  
  return nn.gModule(inputs, outputs)
end

return RNN_theta
