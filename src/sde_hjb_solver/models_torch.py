
from torch import nn

def mlp(sizes, activation, output_activation=nn.Identity()):
    ''' Multilayer perceptron (MLP)
    '''

    # preallocate layers list
    layers = []

    for j in range(len(sizes)-1):

        # actiavtion function
        act = activation if j < len(sizes)-2 else output_activation

        # linear layer
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]

    # Sequential model with given layers
    return nn.Sequential(*layers)


class FeedForwardNN(nn.Module):
    def __init__(self, d_in, d_out, hidden_sizes,
                 activation, output_activation=nn.Identity()):
        super().__init__()
        self.sizes = [d_in] + list(hidden_sizes) + [d_out]
        self.model = mlp(self.sizes, activation, output_activation)
        self.apply(self.init_last_layer_weights)

    def init_last_layer_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.sizes[-1]:
                #nn.init.uniform_(module.weight, -5e-3, 5e-3)
                nn.init.uniform_(module.weight, -5e-4, 5e-4)
                if module.bias is not None:
                    #nn.init.uniform_(module.bias, -5e-3, 5e-3)
                    nn.init.uniform_(module.bias, -5e-4, 5e-4)

    def forward(self, x):
        return self.model.forward(x)
