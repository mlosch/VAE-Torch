-- Based on JoinTable module

require 'nn'

local Merger, parent = torch.class('nn.Merger', 'nn.Module')

function Merger:__init()
    parent.__init(self)
    self.gradInput = {}
    self.output = self.output:cuda()
end 

function Merger:updateOutput(input)
    self.output = self.output or self.output.new()
    local y = nn.JoinTable(1, 3):forward({input[1], input[2]}):cuda()
    self.output:resizeAs(y):copy(y)
    return self.output
end

function Merger:updateGradInput(input, gradOutput)
--    print(gradOutput:size())	
    self.gradInput[1] = self.gradInput[1] or input[1].new()
    self.gradInput[1]:resizeAs(gradOutput:chunk(2,2)[1]):copy(gradOutput:chunk(2,2)[1])

--    print(self.gradInput[1]:size())
    self.gradInput[2] = self.gradInput[2] or input[2].new()
    self.gradInput[2]:resizeAs(gradOutput:chunk(2,2)[2]):copy(gradOutput:chunk(2,2)[2])

--    print(self.gradInput[2]:size())
    return self.gradInput
end
