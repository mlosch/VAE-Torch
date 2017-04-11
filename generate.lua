require 'image'
require 'nn'
require 'Sampler'
require 'Merger'

local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of gen_noise distribution (uniform / normal).
    gennet = 'cp_test/cvae_5_gen.t7',              -- path to the generator network
    decnet = 'cp_test/cvae_5_decoder.t7',
    gendec = 'cp_test/cvae_5_gendec.t7',
    mean = 'cp_test/5_mean.t7',
    log_var = 'cp_test/5_log_var.t7',
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 1,           -- Display image: 0 = false, 1 = true
    nz = 100,              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(gennet ~= '', 'provide a generator model')

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

gen_noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
gennet = torch.load(opt.gennet)
decnet = torch.load(opt.decnet)
gendec = torch.load(opt.gendec)
mean   = torch.load(opt.mean)
log_var= torch.load(opt.log_var)

Sampler = nn.Sampler()
Merger  = nn.Merger()
-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(gennet:get(1)) == 'nn.View' then
    gennet:remove(1)
end

print(decnet)
print(gendec)

if opt.noisetype == 'uniform' then
    gen_noise:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    gen_noise:normal(0, 1)
end
--[[
noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
if opt.noisemode == 'line' then
   -- do a linear interpolation in Z space between point A and point B
   -- each sample in the mini-batch is a point on the line
    line  = torch.linspace(0, 1, opt.batchSize)
    for i = 1, opt.batchSize do
        gen_noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull1d' then
   -- do a linear interpolation in Z space between point A and point B
   -- however, generate the samples convolutionally, so a giant image is produced
    assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    gen_noise = gen_noise:narrow(3, 1, 1):clone()
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        gen_noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
elseif opt.noisemode == 'linefull' then
   -- just like linefull1d above, but try to do it in 2D
    assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    line  = torch.linspace(0, 1, opt.imsize)
    for i = 1, opt.imsize do
        gen_noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    end
end
--]]
local sample_input = torch.randn(2,100,1,1)
if opt.gpu > 0 then
    gennet:cuda()
    cudnn.convert(gennet, cudnn)
    cudnn.convert(gendec, cudnn)
    cudnn.convert(decnet, cudnn)
    cudnn.convert(Sampler, cudnn)
    cudnn.convert(Merger, cudnn)
    gen_noise = gen_noise:cuda()
    sample_input = sample_input:cuda()
    mean = mean:cuda()
    log_var = log_var:cuda()
else
   sample_input = sample_input:float()
   gennet:float()
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(gennet, sample_input)

local i2 = gennet:forward(gen_noise)
local sample = Sampler:forward({mean, log_var}):cuda()
print(sample:size())
local i1 = decnet:forward(sample)
local images = gendec:forward(Merger:updateOutput({i1, i2}))

print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor{input=images, nrow=8})
print('Saved image to: ', opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(images)
    print('Displayed image')
end
