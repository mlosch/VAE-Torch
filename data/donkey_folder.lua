--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '/data/local/imagenet-fetch/256'
if not paths.dirp(opt.data) then
    print('Did not find directory: ', opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "cache"
local cache_prefix = opt.data:gsub('/', '_')
os.execute('mkdir -p cache')
local trainCache = trainCache or paths.concat(cache, cache_prefix .. '_trainCache.t7')
local meanStdCache = meaStdCache or paths.concat(cache, cache_prefix .. '_meanstdCache.t7')

--------------------------------------------------------------------------------------------
local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(path)

   local loadok, input = pcall(image.load, path, 3, 'float')
   if not loadok or not input then
    print('Error while loading image at path: '.. path)
    return nil
   end
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input
end

-- pixel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path)
   collectgarbage()
   local input = loadImage(path)

   if input == nil then
      return nil
   end

   local iW = input:size(3)
   local iH = input:size(2)

   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local out = input

   if iW ~= oW or iH ~= oH then
      -- do random crop
      local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   end

   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   --out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

   -- mean/std
   if mean then out:add(-mean) end
   if std then out:cdiv(std) end

--   if mean then
--      print(mean:min(), mean:max(), mean:mean())
--      print(std:min(), std:max(), std:mean())
--   end
   return out
end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {opt.data},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   print('saved metadata cache at', trainCache)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end


-- Estimate the per-pixel mean/std (so that the loaders can normalize appropriately)
if paths.filep(meanStdCache) then
   local meanstd = torch.load(meanStdCache)
   mean = meanstd.mean
   std = meanstd.std
   print(meanStdCache)
   print('Loaded mean and std from cache.')
else
   local tm = torch.Timer()
   local nSamples = math.min(trainLoader:size(), 10000)
   print('Estimating the mean (per-pixel) over ' .. nSamples .. ' randomly sampled training images')
   sample = trainLoader:sample(1)[1]
   local meanEstimate = torch.zeros(sample:size())
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      meanEstimate:add(img)
   end

   meanEstimate:div(nSamples)
   mean = meanEstimate

   print('Estimating the std (per-pixel) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = torch.zeros(sample:size())
   for i=1,nSamples do
      local img = trainLoader:sample(1)[1]
      t = img:add(-mean)
      stdEstimate:add(t:cmul(t))
   end

   stdEstimate:div(nSamples):sqrt()

--   potentially necessary for some datasets:
--   stdEstimate[stdEstimate:eq(0)] = 1e-6

   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std
   torch.save(meanStdCache, cache)
   print('Time to estimate:', tm:time().real)
end