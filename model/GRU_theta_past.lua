
local GRU_theta = {}

--[[
Creates one timestep of one GRU
Paper reference: http://arxiv.org/pdf/1412.3555v1.pdf
]]--
function GRU_theta.gru()
  local rnn_size = opt.rnn_size
  local num_weekly_slots = opt.num_weekly_slots
  local num_layers = opt.num_layers
  local num_events = opt.num_events
  local dropout = opt.dropout or 0
  local lambda = opt.lambda
  local encoder = opt.encoder
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- e_x
  for L = 1, num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  function new_input_sum(insize, xv, hv)
    local i2h = nn.Linear(insize, rnn_size)(xv)
    local h2h = nn.Linear(rnn_size, rnn_size)(hv)
    return nn.CAddTable()({i2h, h2h})
  end

  local x
  local theta_x, e_x, input_size_L
  local outputs = {}
  local embedings_size = 100
  for L = 1, opt.num_layers do

    local prev_h = inputs[L+2]
    -- the input to this layer

    if L == 1 then 
      theta_x = inputs[1]
      --theta_embedings = nn.Linear(theta_size, embedings_size)(theta_x):annotate{name='emb_theta_lin'}
      --theta_embedings = nn.Sigmoid()(theta_embedings):annotate{name='emb_theta_sigm'}
      e_x = inputs[2]
      e_embedings = nn.LookupTable(num_events, embedings_size)(e_x):annotate{name='emb_e'}
      e_embedings = nn.Reshape(embedings_size)(e_embedings)
      
      --x = nn.JoinTable(2)({theta_embedings, e_embedings}) 
      x = nn.JoinTable(2)({theta_x, e_embedings}) 
      input_size_L = theta_size + embedings_size
      --x = OneHot(input_size)(inputs[1])
      --input_size_L = input_size
    else 
      x = outputs[(L-1)] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- GRU tick
    -- forward the update and reset gates
    local update_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    local reset_gate = nn.Sigmoid()(new_input_sum(input_size_L, x, prev_h))
    -- compute candidate hidden state
    local gated_hidden = nn.CMulTable()({reset_gate, prev_h})
    local p2 = nn.Linear(rnn_size, rnn_size)(gated_hidden)
    local p1 = nn.Linear(input_size_L, rnn_size)(x)
    local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
    -- compute new interpolated hidden state, based on the update gate
    local zh = nn.CMulTable()({update_gate, hidden_candidate})
    local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prev_h})
    
    local next_h = nn.CAddTable()({zh, zhm1})

    table.insert(outputs, next_h)
  end
  local prev_layer = outputs[#outputs]
  local layer
  for i=1, #encoder-1 do
    layer = nn.Linear(encoder[i], encoder[i+1])(prev_layer)
    layer = nn.Tanh()(layer)
    if dropout > 0 then layer = nn.Dropout(dropout)(layer) end
    prev_layer = layer
    print ("encoder", encoder[i], "-->", encoder[i+1])
  end
  layer = QuadraticPenalty(lambda)(layer):annotate{name='hashcode'}
  if dropout > 0 then layer = nn.Dropout(dropout)(layer) end
  table.insert(outputs, layer)

  prev_layer = layer
  for i=#encoder, 3, -1 do
    layer = nn.Linear(encoder[i], encoder[i-1])(prev_layer)
    layer = nn.Tanh()(layer)
    if dropout > 0 then layer = nn.Dropout(dropout)(layer) end
    prev_layer = layer
    print ("decoder", encoder[i], "-->", encoder[i-1])
  end
  
  local theta_pred = nn.Linear(encoder[2], num_weekly_slots)(layer):annotate{name='theta_pred'}
  local logsoft_theta = nn.LogSoftMax()(theta_pred)
  table.insert(outputs, logsoft_theta)
  
  local proj = nn.Linear(encoder[2], num_events)(layer):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  
  return nn.gModule(inputs, outputs)
end

return GRU_theta
