import vgg
import wgg

net_dict = {
    'vgg': vgg,
    'wgg': wgg
}

def get_basenet(name, inputs):
    net = net_dict[name];
    return net.basenet(inputs);
