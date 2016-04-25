require 'torch'
require 'cunn'

local CVAE_unrolled = {}

function CVAE_unrolled.get_encoder(nc, ndf, latent_variable_size)
    -- The Encoder

    local encoder = nn.Sequential()
    encoder:add(nn.View(3 * 32 * 32))

    encoder:add(nn.Linear(3 * 32 * 32, 4096))
    encoder:add(nn.ReLU(true))

    mean_logvar = nn.ConcatTable()
    mean_logvar:add(nn.Linear(4096, latent_variable_size))
    mean_logvar:add(nn.Linear(4096, latent_variable_size))

    encoder:add(mean_logvar)

--    o = torch.rand(64, 3, 32, 32)
--    for i=1,#encoder.modules do
--        o = encoder.modules[i]:forward(o)
--        print(encoder.modules[i], #o)
--    end

    return encoder:cuda()
end

function CVAE_unrolled.get_decoder(nc, ngf, latent_variable_size)
    -- The Decoder

    local decoder = nn.Sequential()
    decoder:add(nn.Linear(latent_variable_size, 4096))
    decoder:add(nn.ReLU(true))

    mean_logvar = nn.ConcatTable()
    s1 = nn.Sequential()
    s1:add(nn.Linear(4096, 3*32*32))
    s1:add(nn.View(3,32,32))
    mean_logvar:add(s1)

    s2 = nn.Sequential()
    s2:add(nn.Linear(4096, 3*32*32))
    s2:add(nn.View(3,32,32))
    mean_logvar:add(s2)
    decoder:add(mean_logvar)


--    o = torch.rand(64, 200)
--    for i=1,#decoder.modules do
--        o = decoder.modules[i]:forward(o)
--        print(decoder.modules[i], #o)
--    end
--    print(#o[1])

    return decoder:cuda()
end

return CVAE_unrolled