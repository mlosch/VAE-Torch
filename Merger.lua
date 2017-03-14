-- Based on JoinTable module

require 'nn'

local Merger, parent = torch.class('nn.Merger', 'nn.Module')

function Merger:__init()
    parent.__init(self)
    self.gradInput = {}
    self.output = self.output:cuda()
end 

function Merger:updateOutput(input)
    self.ouput = nn.JoinTable(1):forward{input[1], input[2]}
    return self.output
end

function Merger:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:chunk(2,1)
    return self.gradInput
end
