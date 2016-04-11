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
   niter = 50,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
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

local nz = opt.nz

local encoder = VAE.get_encoder(3, opt.ndf, nz)
local decoder = VAE.get_decoder(3, opt.ngf, nz)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model

reconstruction, reconstruction_var = decoder(z):split(2)
model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})
criterion = nn.GaussianCriterion():cuda()


encoder:apply(weights_init)
decoder:apply(weights_init)

KLD = nn.KLDCriterion():cuda()

local parameters, gradients = model:getParameters()

---------------------------------------------------------------------------
optimState = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local lowerbound = 0
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();
   decoder = util.cudnn(decoder);
   encoder = util.cudnn(encoder)
   encoder:cuda();
   decoder:cuda();
   criterion:cuda()
end

if opt.display then disp = require 'display' end
require 'image'


local fx = function(x)
    if x ~= parameters then
        parameters:copy(x)
    end

    -- Taken from DCGAN:
    --encoder:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    model:zeroGradParameters()
    local reconstruction, reconstruction_var, mean, log_var

    data_tm:reset(); data_tm:resume()
    local real = data:getBatch()
    while real == nil do
        print('Got nil batch for real')
        real = data:getBatch()
    end
    data_tm:stop()
    input:copy(real)

    reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(input))
    reconstruction = {reconstruction, reconstruction_var}

    local err = criterion:forward(reconstruction, input)
    local df_dw = criterion:backward(reconstruction, input)

    local KLDerr = KLD:forward(mean, log_var)
    local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

    error_grads = {df_dw[1], df_dw[2], dKLD_dmu, dKLD_dlog_var}

    model:backward(input, error_grads)

    local batchlowerbound = err + KLDerr

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
      x, batchlowerbound = optim.adam(fx, parameters, optimState)
      lowerbound = lowerbound + batchlowerbound[1]

      -- display
      counter = counter + 1
      if counter % 100 == 0 and opt.display then
          local reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(input))
          if reconstruction then
            --disp.image(fake, {win=opt.display_id, title=opt.name})
            image.save(('images/epoch_%d_iter_%d_real.jpg'):format(epoch, counter), image.toDisplayTensor{input=input, nrow=8})
            image.save(('images/epoch_%d_iter_%d_fake.jpg'):format(epoch, counter), image.toDisplayTensor{input=reconstruction, nrow=8})
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
   parameters = nil
   gradients = nil
   parameters, gradients = model:getParameters() -- reflatten the params and get them
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end


