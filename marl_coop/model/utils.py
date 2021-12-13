
import torch.optim as optim
from torch import nn


class OptimizerCreator():
    '''
    Factory that build the optimizer that will update the actor and critic network.
    '''
    def __init__(self):
        self.builders = {
            'Adam': lambda parameters, kwargs : optim.Adam(parameters, **kwargs),
            'SGD' : lambda parameters, kwargs : optim.SGD(parameters, **kwargs),
            'RMSprop' : lambda parameters, kwargs : optim.RMSprop(parameters, **kwargs),
            'Adadelta' : lambda parameters, kwargs : optim.Adadelta(parameters, **kwargs),
            'NAdam' : lambda parameters, kwargs : optim.NAdam(parameters, **kwargs)
        }

    def create(self, optimizer, parameters, kwargs):
        return self.builders[optimizer](parameters, kwargs)

class CriterionCreator():
    '''
    Factory that build a loss function adaptated to regression.
    '''
    def __init__(self):
        self.builders = {
            'MSE': nn.MSELoss,
            'L1' : nn.L1Loss,
            'SmoothL1': nn.SmoothL1Loss
        }

    def create(self, criterion):
        return self.builders[criterion]()