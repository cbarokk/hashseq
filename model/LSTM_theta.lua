
local LSTM_theta = {}
function LSTM_theta.lstm(input_size, rnn_size, n, num_events, dropout, lambda)
  dropout = dropout or 0 
  -- there will be 2*n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- e_x

  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local theta_x, e_x, input_size_L
  local outputs = {}
  local embedings_size = 100
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+2]
    local prev_c = inputs[L*2+1]
    -- the input to this layer
    if L == 1 then 
      theta_x = inputs[1]
      theta_embedings = nn.Linear(input_size, embedings_size)(theta_x):annotate{name='emb_theta_lin'}
      theta_embedings = nn.Sigmoid()(theta_embedings):annotate{name='emb_theta_sigm'}
      e_x = inputs[2]
      e_embedings = nn.LookupTable(num_events, embedings_size)(e_x):annotate{name='emb_e'}
      e_embedings = nn.Reshape(embedings_size)(e_embedings)
      
      
      x = nn.JoinTable(2)({theta_embedings, e_embedings}) 
      input_size_L = 2*embedings_size
      
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size

    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1):annotate{name='in_gate_'..L}
    local forget_gate = nn.Sigmoid()(n2):annotate{name='forget_gate_'..L}
    local out_gate = nn.Sigmoid()(n3):annotate{name='out_gate_'..L}
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}):annotate{name='next_h_'..L}
    
    if L == n then
      next_h = QuadraticPenalty(lambda)(next_h):annotate{name='top_h_sparse'}
    end
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
    
  end

  -- set up the decoder
  
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  
  local proj = nn.Linear(rnn_size, num_events)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  
  return nn.gModule(inputs, outputs)
end

return LSTM_theta

