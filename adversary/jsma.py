"""
Referential implementation: cleverhans tensorflow
"""
import torch
import numpy as np
from settings import *
from torch.autograd import Variable

class SaliencyMapMethod(object):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    :param model: pytorch model
    :param kwargs: passed through to super constructor
    """

    def __init__(self, model, **kwargs):
        super(SaliencyMapMethod, self).__init__()
        self.model = model

        self.theta = kwargs['theta']
        self.gamma = kwargs['gamma']
        self.clip_min = kwargs['clip_min']
        self.clip_max = kwargs['clip_max']
        self.nb_classes = kwargs['nb_classes']

    def generate(self, x, y=None, y_target=None):
        """
        :param x: The model's inputs.
        :return:
        """
        self.y = y
        self.y_target = y_target
        # Create random targets if y_target not provided
        if self.y_target is None:
            from random import randint

            def random_targets(gt):
                result = gt.copy()
                for i in range(len(gt)):
                    rand_num = randint(0, self.nb_classes-1)
                    while rand_num == result[i]:
                        rand_num = randint(0, self.nb_classes - 1)
                    result[i] = rand_num
                return result

            labels = self.get_or_guess_labels(x)
            self.y_target = torch.from_numpy(random_targets(labels.cpu().numpy())).to(device)

        x_adv = jsma_symbolic(
            x,
            model=self.model,
            y_target=self.y_target,
            theta=self.theta,
            gamma=self.gamma,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            nb_classes=self.nb_classes)
        return x_adv

    def get_or_guess_labels(self, x):
        if self.y is not None:
            labels = self.y
        else:
            outputs = self.model(x)
            _, labels = outputs.max(1)
        return labels

def jsma_symbolic(x, y_target, model, theta, gamma, clip_min, clip_max, nb_classes):
    """
    :param x: the input tensor
    :param y_target: the target tensor
    :param model: a pytorch model object.
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: a tensor for the adversarial example
    """
    nb_features = int(np.prod(x.size()[1:]))

    max_iters = np.floor(nb_features * gamma / 2)
    increase = bool(theta > 0)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).float().to(device)

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).x
    if increase:
        search_domain = (x < clip_max).float().reshape(-1, nb_features)
    else:
        search_domain = (x > clip_min).float().reshape(-1, nb_features)

    # Loop variables
    # x_in: the tensor that holds the latest adversarial outputs that are in
    #       progress.
    # y_in: the tensor for target labels
    # domain_in: the tensor that holds the latest search domain
    # cond_in: the boolean tensor to show if more iteration is needed for
    #          generating adversarial samples

    def condition(x_in, y_in, domain_in, i_in, cond_in):
        # Repeat the loop until we have achieved misclassification or
        # reaches the maximum iterations
        return (i_in < max_iters) and cond_in

    def body(x_in, y_in, domain_in, i_in, cond_in):
        x_in = Variable(x_in.data, requires_grad=True)
        y_in_one_hot = torch.zeros(y_in.shape[0], nb_classes).scatter_(1, y_in.cpu().reshape(-1, 1).long(), 1).to(device)
        logits = model(x_in)
        _, preds = logits.max(1)

        # create the Jacobian
        grads = None
        for class_ind in range(nb_classes):
            model.zero_grad()
            logits[:, class_ind].sum().backward(retain_graph=True)
            derivatives = x_in.grad
            if class_ind == 0:
                grads = derivatives
            else:
                grads = torch.cat((grads, derivatives))
        grads = grads.reshape(nb_classes, -1, nb_features)

        # Compute the Jacobian components
        # To help with the computation later, reshape the target_class
        # and other_class to [nb_classes, -1, 1].
        # The last dimention is added to allow broadcasting later.
        target_class = y_in_one_hot.permute(1, 0).reshape(nb_classes, -1, 1)
        other_class = (target_class != 1).float()

        grads_target = torch.sum(grads * target_class, dim=0)
        grads_other = torch.sum(grads * other_class, dim=0)

        # Remove the already-used input features from the search space
        # Subtract 2 times the maximum value from those value so that
        # they won't be picked later
        increase_coef = (4 * int(increase) - 2) * (domain_in == 0).float()

        target_tmp = grads_target
        target_tmp -= increase_coef * torch.max(torch.abs(grads_target), dim=1, keepdim=True)[0]
        target_sum = target_tmp.reshape(-1, nb_features, 1) + target_tmp.reshape(-1, 1, nb_features)

        other_tmp = grads_other
        other_tmp -= increase_coef * torch.max(torch.abs(grads_other), dim=1, keepdim=True)[0]
        other_sum = other_tmp.reshape(-1, nb_features, 1) + other_tmp.reshape(-1, 1, nb_features)

        # Create a mask to only keep features that match conditions
        if increase:
            scores_mask = ((target_sum > 0) & (other_sum < 0))
        else:
            scores_mask = ((target_sum < 0) & (other_sum > 0))

        # Create a 2D numpy array of scores for each pair of candidate features
        scores = scores_mask.float() * (-target_sum * other_sum) * zero_diagonal

        # Extract the best two pixels
        best = torch.argmax(scores.reshape(-1, nb_features * nb_features), dim=1)

        p1 = np.mod(best, nb_features)
        p2 = np.floor_divide(best, nb_features)
        p1_one_hot = torch.zeros(y_in.shape[0], nb_features).scatter_(1, p1.reshape(-1,1).long(), 1).to(device)
        p2_one_hot = torch.zeros(y_in.shape[0], nb_features).scatter_(1, p2.reshape(-1,1).long(), 1).to(device)

        # Check if more modification is needed for each sample
        mod_not_done = y_in != preds
        cond = mod_not_done & (torch.sum(domain_in, dim=1) >= 2)

        #update the search domain
        cond_float = cond.reshape(-1, 1).float().to(device)
        to_mod = (p1_one_hot + p2_one_hot) * cond_float

        domain_out = domain_in - to_mod

        # Apply the modification to the images
        to_mod_reshape = to_mod.reshape([-1] + list(x_in.shape[1:]))
        if increase:
            x_out = torch.clamp(x_in + to_mod_reshape * theta, max=clip_max)
        else:
            x_out = torch.clamp(x_in - to_mod_reshape * theta, min=clip_min)

        # Increase the iterator, and check if all misclassifications are done
        i_out = i_in + 1
        cond_out = torch.sum(cond) != 0

        return x_out, y_in, domain_out, i_out, cond_out

    # Run loop to do JSMA
    x_adv, y_in, domain_out, i_out, cond_out = x, y_target, search_domain, 0, True
    conditions = condition(x_adv, y_in, domain_out, i_out, cond_out)
    while (conditions):
        x_adv, y_in, domain_out, i_out, cond_out = body(x_adv, y_in, domain_out, i_out, cond_out)
        conditions = condition(x_adv, y_in, domain_out, i_out, cond_out)

    return x_adv