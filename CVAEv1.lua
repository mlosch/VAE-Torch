require 'torch'
require 'cunn'

local CVAEv1 = {}

function CVAEv1.get_encoder(nc, ndf, latent_variable_size)
    -- The Encoder

    local encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(nc, ndf, 12, 12, 2, 2, 5, 5 ))
    encoder:add(nn.SpatialBatchNormalization(ndf))
    encoder:add(nn.LeakyReLU(0.2, true))

    encoder:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf * 2))
    encoder:add(nn.LeakyReLU(0.2, true))

    encoder:add(nn.SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
    encoder:add(nn.SpatialBatchNormalization(ndf * 4))
    encoder:add(nn.LeakyReLU(0.2, true))

    local w = 4

    encoder:add(nn.View(ndf * 4 * w*w))

    encoder:add(nn.Linear(ndf * 4 * w*w, ndf * 4 * w*w))
    encoder:add(nn.BatchNormalization(ndf * 4 * w*w))
    encoder:add(nn.LeakyReLU(0.2, true))

    local mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(ndf * 4 * w*w, latent_variable_size))
    mean_logvar:add(nn.Linear(ndf * 4 * w*w, latent_variable_size))

    encoder:add(mean_logvar)

--    o = torch.rand(64, 3, 32, 32)
--    for i=1,#encoder.modules do
--        o = encoder.modules[i]:forward(o)
--        print(encoder.modules[i], #o)
--    end

    return encoder:cuda()
end

function CVAEv1.get_decoder(nc, ngf, latent_variable_size)
    -- The Decoder

    local w = 4

    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, ngf * 4 * w*w))
    decoder:add(nn.BatchNormalization(ngf * 4 * w*w))
    decoder:add(nn.ReLU(true))

    decoder:add(nn.Linear(ngf * 4 * w*w, ngf * 4 * w*w))
    decoder:add(nn.BatchNormalization(ngf * 4 * w*w))
    decoder:add(nn.ReLU(true))

    decoder:add(nn.View(ngf * 4, w, w))

    decoder:add(nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.SpatialBatchNormalization(ngf * 2))
    decoder:add(nn.ReLU(true))

    decoder:add(nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
    decoder:add(nn.SpatialBatchNormalization(ngf))
    decoder:add(nn.ReLU(true))

    local mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.SpatialFullConvolution(ngf, nc, 12, 12, 2, 2, 5, 5))
    mean_logvar:add(nn.SpatialFullConvolution(ngf, nc, 12, 12, 2, 2, 5, 5))
    decoder:add(mean_logvar)

--    decoder:add(nn.Tanh())

    o = torch.rand(64, latent_variable_size)
    for i=1,#decoder.modules do
        o = decoder.modules[i]:forward(o)
        print(decoder.modules[i], #o)
    end
    print(#o[1])
    exit()

    return decoder:cuda()
end

return CVAEv1