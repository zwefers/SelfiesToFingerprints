from torch import nn

class Net(nn.Module):

    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.neuralnet = nn.Sequential(nn.Linear(input_dim, layer1_dim),
                                    nn.ReLU(),
                                    nn.Linear(layer1_dim, layer2_dim),
                                    nn.ReLU(),
                                    nn.Linear(layer2_dim, output_dim),
                                    nn.Sigmoid())

    def forward(self, x):
        return self.neuralnet(x)