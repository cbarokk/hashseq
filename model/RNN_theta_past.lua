local RNN_theta = {}

function RNN_theta.rnn(input_size, rnn_size, n, num_events, dropout, lambda)
  
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- e_x

  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local embedings_size = 100

  for L = 1,n do
    
    local prev_h = inputs[L+2]
    if L == 1 then 
      theta_x = inputs[1]
      theta_embedings = nn.Linear(theta_size, embedings_size)(theta_x):annotate{name='emb_theta'}
      e_x = inputs[2]
      e_embedings = nn.LookupTable(num_events, embedings_size)(e_x):annotate{name='emb_e'}
      e_embedings = nn.Reshape(embedings_size)(e_embedings)
      
      x = nn.JoinTable(2)({theta_embedings, e_embedings}) 
      input_size_L = 2*embedings_size
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

    if L == n then
      next_h = QuadraticPenalty(lambda)(next_h):annotate{name='top_h_sparse'}
    end

    table.insert(outputs, next_h)
  end
-- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  
  local theta_pred = nn.Linear(rnn_size, 60*24*7)(top_h):annotate{name='theta_pred'}
  local logsoft_theta = nn.LogSoftMax()(theta_pred)
  table.insert(outputs, logsoft_theta)
  
  local proj = nn.Linear(rnn_size, num_events)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return RNN_theta
