--
-- Created by mlosch.
-- Date: 11-4-16
-- Time: 15:21
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

require 'KLDCriterion'
require 'GaussianCriterion'
require 'Sampler'

util = paths.dofile('util.lua')

opt = {
   dataset = 'folder',       -- imagenet / lsun / folder
   batchSize = 100,
   loadSize = 96,
   fineSize = 64,
   nz = 200,               -- #  of dim for Z
   ngf = 96,               -- #  of gen filters in first conv layer
   ndf = 96,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 10000,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   beta_n = 10,            -- warm up epochs
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_out = 'images',        -- display window id or output folder
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   vae = 'CVAE',
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

local VAE = require(opt.vae)

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

--local encoder = torch.load('checkpoints/imagenet_all_1_encoder.t7'):cuda()
--local decoder = torch.load('checkpoints/imagenet_all_1_decoder.t7'):cuda()

local criterion = nn.GaussianCriterion()
local input = nn.Identity()()


if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   decoder = util.cudnn(decoder);
   encoder = util.cudnn(encoder)
   encoder:cuda();
   decoder:cuda();
   criterion:cuda()
end




local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local reconstruction, reconstruction_var, model

reconstruction, reconstruction_var = decoder(z):split(2)
model = nn.gModule({input},{reconstruction, reconstruction_var, mean, log_var})

--encoder:apply(weights_init)
--decoder:apply(weights_init)

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
local sumKLDerr = 0
local sumCriterionErr = 0
local epoch = 0 -- public due to usage for warmup phase

local eval = require 'manvsmachine/evaluate.lua'
local matio = require 'matio'
----------------------------------------------------------------------------

if opt.gpu > 0 then input = input:cuda() end

if opt.display then
    disp = require 'display'
    require 'image'
end


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

--    print(input:min(), input:max())

    reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(input))
    reconstruction = {reconstruction, reconstruction_var }

    local err = criterion:forward(reconstruction, input)
    local df_dw = criterion:backward(reconstruction, input)

    local KLDerr = KLD:forward(mean, log_var)
    local dKLD_dmu, dKLD_dlog_var = unpack(KLD:backward(mean, log_var))

    --local wu_beta = 1.0 - math.max(0, epoch - opt.beta_n) / opt.beta_n
    local wu_beta = 1.0
    KLDerr = KLDerr*wu_beta

    error_grads = {df_dw[1], df_dw[2], dKLD_dmu*wu_beta, dKLD_dlog_var*wu_beta}

    model:backward(input, error_grads)

    local batchlowerbound = err + KLDerr

    return {batchlowerbound, KLDerr, err}, gradients
end

function get_node_params(model)
    local params = {}
    local names = {}
    for indexNode, node in ipairs(model.forwardnodes) do
      if node.data.module and node.data.module.modules then
          for indexModule, module in ipairs(node.data.module.modules) do
              if module.weight then
                  params[#params+1] = module.weight:float()
                  names[#names+1] = tostring(module)
              end
          end
      end
    end
    return params, names
end

local node_params1, node_names = get_node_params(model)


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0

   lowerbound = 0
   sumKLDerr = 0
   sumCriterionErr = 0

   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()

      -- Update model
      x, batchlowerbound = optim.adam(fx, parameters, optimState)
      lowerbound = lowerbound + batchlowerbound[1][1]
      sumKLDerr = sumKLDerr + batchlowerbound[1][2]
      sumCriterionErr = sumCriterionErr + batchlowerbound[1][3]

      node_params2, _ = get_node_params(model)
      for k=#node_params1,1,-1 do
          diff = node_params2[k] - node_params1[k]
          if torch.max(torch.abs(diff)) > 0 then
              --print('> 0 | '.. k .. ' | '.. node_names[k])
          else
              print('= 0 | '.. k .. ' | '.. node_names[k])
          end
      end
      node_params1 = node_params2

      -- display
      counter = counter + 1
      if counter % 1000 == 0 and opt.display then
          local reconstruction, reconstruction_var, mean, log_var = unpack(model:forward(input))
          if reconstruction then
            --disp.image(fake, {win=opt.display_id, title=opt.name})
            image.save(('%s/epoch_%d_iter_%d_real.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=input, nrow=8})
            image.save(('%s/epoch_%d_iter_%d_reconstr.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=reconstruction, nrow=8})
            image.save(('%s/epoch_%d_iter_%d_encconv1.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=model.forwardnodes[3].data.module.modules[1].weight, nrow=8, padding=1})

            local randz = torch.randn(opt.batchSize, opt.nz)
            if opt.gpu > 0 then randz = randz:cuda() end
            local fake, fake_var = unpack(decoder:forward(randz))
            image.save(('%s/epoch_%d_iter_%d_fake.jpg'):format(opt.display_out, epoch, counter), image.toDisplayTensor{input=fake, nrow=8})
          else
            print('Reconstruction is Nil')
          end
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Lowerbound: %.4f  KLDerr: %.4f  Criterion: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 lowerbound/((i-1)/opt.batchSize),
                 sumKLDerr/((i-1)/opt.batchSize),
                 sumCriterionErr/((i-1)/opt.batchSize))
         )
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


   encodings = {}
   ys = {}
   for i=1,10 do
       batch, labels = data:getBatch()

       for l=1,labels:size(1) do
           ys[#ys+1] = labels[l]
       end

       enc = eval.encode(model, batch:cuda(), nil, nil)

       for b=1,#enc do
           encodings[#encodings+1] = enc[b]
       end
   end
   out = {}
   out.zs = torch.cat(encodings,2):t()
   out.ys = torch.Tensor(ys)
   matio.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_encodings.mat', out)

end


