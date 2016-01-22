local QuadraticPenalty, parent = torch.class('QuadraticPenalty', 'nn.Module')

function QuadraticPenalty:__init(lambda, sizeAverage)
  parent.__init(self)
  self.lambda = lambda or 0
  self.train = true
end

function QuadraticPenalty:updateOutput(input)
  self.sparsity = input:clone():abs():mean()
  local loss = input:clone():cmul(input):mul(-1):add(1):mul(self.lambda):mean()
  self.loss = loss
  self.output = input
  --gnuplot.figure("top_h sparse")
  --gnuplot.title("top_h sparse")
  --gnuplot.hist(input, 100)
  
  return self.output 
end

function QuadraticPenalty:updateGradInput(input, gradOutput)
  local m = self.lambda/input:nElement() 
  self.gradInput:resizeAs(input):copy(input):mul(-2*m):add(gradOutput)
  return self.gradInput 
end

