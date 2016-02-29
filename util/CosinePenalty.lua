local CosinePenalty, parent = torch.class('CosinePenalty', 'nn.Module')

function CosinePenalty:__init(lambda, sizeAverage)
  parent.__init(self)
  self.lambda = lambda or 0
  self.train = true
end

function CosinePenalty:updateOutput(input)
  self.sparsity = input:clone():abs():mean()
  self.loss = input:clone():cmul(input):mul(-1):add(1):mul(self.lambda):mean()
  --local cosine_loss = input:clone():mul(2*math.pi):cos():add(1):mean(1):sum()
  --local cosine_loss = 0
  local avg_loss = input:clone():mean(1):mul(4):sum()
 -- self.loss = self.lambda * (cosine_loss + avg_loss)
  self.output = input
  --print (input:clone():mul(2*math.pi):cos():add(1):mul(self.lambda):mean(1):sum())
  --gnuplot.figure("top_h sparse")
  --gnuplot.title("top_h sparse")
  --gnuplot.hist(input, 2)
  return self.output 
end

function CosinePenalty:updateGradInput(input, gradOutput)
  local m = self.lambda/input:nElement() 
  self.gradInput:resizeAs(input):copy(input):mul(2*math.pi):sin():mul(-2*math.pi*m):add(gradOutput)
  --self.gradInput:add(input:clone():sign():mul(m*4))
  --self.gradInput:resizeAs(input):copy(input):sign():mul(m):add(gradOutput)
  return self.gradInput 
end

