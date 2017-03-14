--
-- Created by mlosch.
-- Date: 11-4-16
-- Time: 15:21
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local VAE = require 'CVAE'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'
require 'Merger'

util = paths.dofile('util.lua')

opt = {
   dataset = 'folder',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 10000,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_out = 'images',        -- display window id or output folder
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'cvae',
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local Disl = 10

local real_label = 1
local fake_label = 0

-- Added Discriminator here --

local netD = nn.Sequential()
local SpatialConvolution = nn.SpatialConvolution
local SpatialBatchNormalization = nn.SpatialBatchNormalization


-- input is (nc) x 64 x 64
local nc = 3

netD:add(SpatialConvolution(nc, opt.ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf) x 32 x 32
netD:add(SpatialConvolution(opt.ndf, opt.ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(opt.ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf*2) x 16 x 16
netD:add(SpatialConvolution(opt.ndf * 2, opt.ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(opt.ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf*4) x 8 x 8
netD:add(SpatialConvolution(opt.ndf * 4, opt.ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(opt.ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (opt.ndf*8) x 4 x 4
netD:add(SpatialConvolution(opt.ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

-- Addition ends here --

local nz = opt.nz

local encoder = VAE.get_encoder(3, opt.ndf, nz)
local decoder = VAE.get_decoder(3, opt.ngf, nz)
local gen     = VAE.get_generator(3, opt.ngf, nz)
local gendec  = VAE.get_gendec(3, opt.ngf)


local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})
local zp = nn.Identity()()


local decoder_output = decoder(z)
local gen_ouput = gen(zp)
local gen_dec_output = nn.Merger()({decoder_ouput, gen_output})
local reconstruction = gendec(gen_dec_output)

local model = nn.gModule({input, zp},{reconstruction, mean, log_var})
criterion = nn.MSECriterion():cuda()


encoder:apply(weights_init)
decoder:apply(weights_init)
gen:apply(weights_init)
gendec:apply(weights_init)

KLD = nn.KLDCriterion():cuda()

local parameters, gradients = model:getParameters()

---------------------------------------------------------------------------
optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
	learningRate = opt.lr,
	beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local gen_noise =  torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local lowerbound = 0

if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda()
   gen_noise = gen_noise:cuda()
   label = label:cuda()
   decoder = util.cudnn(decoder)
   encoder = util.cudnn(encoder)
   gen     = util.cudnn(gen)
   gendec  = util.cudnn(gendec)
   encoder:cuda()
   decoder:cuda()
   gen:cuda()
   gendec:cuda()
   criterion:cuda()
end

if opt.display then
    disp = require 'display'
    require 'image'
end

local parametersD, gradParametersD = netD:getParameters()
local gan_criterion = nn.BCECriterion()

local real_disl = torch.Tensor(opt.batchSize, 512, 4, 4)
local fake_disl = torch.Tensor(opt.batchSize, 512, 4, 4)

local fDx = function(x)
   gradParametersD:zero()
   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   real_disl:copy(netD:get(Disl).output)
   local errD_real = gan_criterion:forward(output, label)
   local df_do = gan_criterion:backward(output, label)
   netD:backward(input, df_do)

   gen_noise:normal(0, 1)
   label:fill(fake_label)
   model:forward({input, gen_noise})

   -- train with reconstructed image
   local output = netD:forward(model.output[1])
   fake_disl:copy(netD:get(Disl).output)
   local errD_fake = gan_criterion:forward(output, label)
   local df_do = gan_criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

local fx = function(x)
    if x ~= parameters then
        parameters:copy(x)
    end

    -- Taken from DCGAN:
    --encoder:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    model:zeroGradParameters()
    local err = criterion:forward(fake_disl, real_disl) -- replacing MSE criteria, autoencoding beyond pixel

    local KLDerr = KLD:forward(model.ouput[2], model.output[3])
    local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(model.output[2], model.output[3]))
    
    label:fill(real_label)

    local output = netD.output
    local errG = gan_criterion:forward(output, label)
    local df_do = gan_criterion:backward(output, label)
    local df_dg = netD:updateGradInput(model.output[1], df_do)

    error_grads = { df_dg, dKLD_dmu, dKLD_dlog_var}

    model:backward({input, gen_noise}, error_grads)

    local batchlowerbound = err + KLDerr + errG

    return batchlowerbound, gradients
end


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0

   lowerbound = 0

   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      -- Update model
      optim.adam(fDx, parametersD, optimStateD)
      x, batchlowerbound = optim.adam(fx, parameters, optimState)
      lowerbound = lowerbound + batchlowerbound[1]

      -- display
      counter = counter + 1
      if counter % 100 == 0 and opt.display then
          local reconstruction, mean, log_var = unpack(model:forward(input))
          if reconstruction then
            --disp.image(fake, {win=opt.display_id, title=opt.name})
            image.save(('%s/epoch_%d_iter_%d_real.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=input, nrow=8})
            image.save(('%s/epoch_%d_iter_%d_fake.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=reconstruction, nrow=8})
          else
            print('Fake image is Nil')
          end
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Lowerbound: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 lowerbound/((i-1)/opt.batchSize)))
      end
   end

   lowerboundlist = torch.Tensor(1,1):fill(lowerbound/(epoch * math.min(data:size(), opt.ntrain)))

   paths.mkdir('checkpoints')
   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_encoder.t7', encoder, opt.gpu)
   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_decoder.t7', decoder, opt.gpu)
--   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_state.t7', state, opt.gpu)
--   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_lowerbound.t7', torch.Tensor(lowerboundlist), opt.gpu)
--   parameters = nil
--   gradients = nil
--   parameters, gradients = model:getParameters() -- reflatten the params and get them
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end


