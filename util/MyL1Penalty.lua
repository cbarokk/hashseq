local MyL1Penalty, parent = torch.class('MyL1Penalty','nn.Module')

--This module acts as an L1 latent state regularizer, adding the 
--[gradOutput] to the gradient of the L1 loss. The [input] is copied to 
--the [output]. 

function MyL1Penalty:__init(lambda, sizeAverage, provideOutput)
    parent.__init(self)
    self.lambda = lambda or 0
    self.sizeAverage = sizeAverage or false  
    if provideOutput == nil then
       self.provideOutput = true
    else
       self.provideOutput = provideOutput
    end
end
    
function MyL1Penalty:updateOutput(input)
    self.sparsity = input:clone():abs():mean()
    local m = self.lambda 
    if self.sizeAverage == true then 
      m = m/input:nElement()
    end
    local loss = m*input:norm(1) 
    self.loss = loss  
    self.output = input 
    return self.output 
end

function MyL1Penalty:updateGradInput(input, gradOutput)
    local m = self.lambda 
    if self.sizeAverage == true then 
      m = m/input:nElement() 
    end
    
    self.gradInput:resizeAs(input):copy(input):sign():mul(m)
    
    if self.provideOutput == true then 
        self.gradInput:add(gradOutput)  
    end 

    return self.gradInput 
end

function MyL1Penalty:clearState()
   if self.loss then self.loss:set() end
   return parent.clearState(self)
end