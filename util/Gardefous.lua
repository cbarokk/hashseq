local Gardefous, parent = torch.class('Gardefous', 'nn.Module')

function Gardefous:__init(lambda, sizeAverage)
  parent.__init(self)
  self.lambda = lambda or 0
  self.train = true
end

function Gardefous:updateOutput(input)
  self.sparsity = input:clone():abs():mean()
  local loss = input:clone():cmul(input):mul(self.lambda):mean()
  self.loss = loss
  self.output = input
  --gnuplot.figure("top_h sparse")
  --gnuplot.title("top_h sparse")
  --gnuplot.hist(input, 100)
  
  return self.output 
end

function Gardefous:updateGradInput(input, gradOutput)
  local m = self.lambda/input:nElement() 
  self.gradInput:resizeAs(input):copy(input):mul(2*m):add(gradOutput)
  return self.gradInput 
end

