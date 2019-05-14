from operator import methodcaller

import torch
import torch.nn as nn
from torch.autograd import Variable
from settings import *


def get_cuda_state(obj):
    """
    Get cuda state of any object.

    :param obj: an object (a tensor or an `torch.nn.Module`)
    :raise TypeError:
    :return: True if the object or the parameter set of the object
             is on GPU
    """
    if isinstance(obj, nn.Module):
        try:
            return next(obj.parameters()).is_cuda
        except StopIteration:
            return None
    elif hasattr(obj, 'is_cuda'):
        return obj.is_cuda
    else:
        raise TypeError('unrecognized type ({}) in args'.format(type(obj)))


def is_cuda_consistent(*args):
    """
    See if the cuda states are consistent among variables (of type either
    tensors or torch.autograd.Variable). For example,

        import torch
        from torch.autograd import Variable
        import torch.nn as nn

        net = nn.Linear(512, 10)
        tensor = torch.rand(10, 10).cuda()
        assert not is_cuda_consistent(net=net, tensor=tensor)

    :param args: the variables to test
    :return: True if len(args) == 0 or the cuda states of all elements in args
             are consistent; False otherwise
    """
    result = dict()
    for v in args:
        cur_cuda_state = get_cuda_state(v)
        cuda_state = result.get('cuda', cur_cuda_state)
        if cur_cuda_state is not cuda_state:
            return False
        result['cuda'] = cur_cuda_state
    return True

def make_cuda_consistent(refobj, *args):
    """
    Attempt to make the cuda states of args consistent with that of ``refobj``.
    If any element of args is a Variable and the cuda state of the element is
    inconsistent with ``refobj``, raise ValueError, since changing the cuda state
    of a Variable involves rewrapping it in a new Variable, which changes the
    semantics of the code.

    :param refobj: either the referential object or the cuda state of the
           referential object
    :param args: the variables to test
    :return: tuple of the same data as ``args`` but on the same device as
             ``refobj``
    """
    ref_cuda_state = refobj if type(refobj) is bool else get_cuda_state(refobj)
    if ref_cuda_state is None:
        raise ValueError('cannot determine the cuda state of `refobj` ({})'
                .format(refobj))

    result_args = list()
    for v in args:
        cuda_state = get_cuda_state(v)
        if cuda_state != ref_cuda_state:
            v = v.to(device)
        result_args.append(v)
    return tuple(result_args)

def predict(net, inputs):
    """
    Predict labels. The cuda state of `net` decides that of the returned
    prediction tensor.

    :param net: the network
    :param inputs: the input tensor (non Variable), of dimension [B x C x W x H]
    :return: prediction tensor (LongTensor), of dimension [B]
    """
    inputs = make_cuda_consistent(net, inputs)[0]
    inputs_var = Variable(inputs)
    outputs_var = net(inputs_var)
    predictions = torch.max(outputs_var.data, dim=1)[1]
    return predictions
