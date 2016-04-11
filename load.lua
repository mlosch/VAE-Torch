require 'hdf5'

function load(continuous)
    if continuous then
        return loadfreyfaces()
    else
        return loadmnist()
    end
end

function loadmnist()
    -- This loads an hdf5 version of the MNIST dataset used here: http://deeplearning.net/tutorial/gettingstarted.html
    -- Direct link: http://deeplearning.net/data/mnist/mnist.pkl.gz

    local f = hdf5.open('datasets/mnist.hdf5', 'r')

    data = {}
    data.train = f:read('x_train'):all():cuda()
    data.test = f:read('x_test'):all():cuda()

    f:close()

    return data
end

function loadfreyfaces()
    require 'hdf5'
    local f = hdf5.open('datasets/freyfaces.hdf5', 'r')
    local data = {}
    data.train = f:read('train'):all():cuda()
    data.test = f:read('test'):all():cuda()
    f:close()

    return data
end
