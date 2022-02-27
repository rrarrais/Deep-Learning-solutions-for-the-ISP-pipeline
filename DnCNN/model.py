from torch.nn import Module
from torch import nn


class DnCnn(Module):
    def __init__(self, in_channels=4, out_channels=3, depth=17):
        super(DnCnn, self).__init__()
        self._build(in_channels, out_channels, depth)

    def _build(self, in_channels, out_channels, depth):
        # Network Head
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.relu1 = nn.ReLU()

        # Define transpose convolution entry point
        num_blocks = depth - 2
        entry_point = int(num_blocks/2)
        num_output_blocks = num_blocks - entry_point

        # Initiate network body
        body = []

        # Add convolutions with input dimensions
        for i in range(entry_point):
            body.append(nn.Conv2d(64, 64, 3, 1, 1))
            body.append(nn.BatchNorm2d(64, momentum=0.9, eps=1e-04, affine=True))
            body.append(nn.ReLU(inplace=True))

        # Add transpose convolution
        body.append(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1))
        body.append(nn.BatchNorm2d(64, momentum=0.9, eps=1e-04, affine=True))
        body.append(nn.ReLU(inplace=True))

        # Add remaining body blocks with output dimensions
        for i in range(num_output_blocks+1):
            body.append(nn.Conv2d(64, 64, 3, 1, 1))
            body.append(nn.BatchNorm2d(64, momentum=0.9, eps=1e-04, affine=True))
            body.append(nn.ReLU(inplace=True))

        # Create body and append it to the network
        self.body = nn.Sequential(*body)

        # Network tail
        self.conv2 = nn.Conv2d(64, 3, 1, 1)

    def print_info(self):
        print("Model CNDNN")
        print("==================")
        print("conv1 weights", self.conv1.weight.shape)
        print("================")
        for idx, block in enumerate(self.body):
            print(f"Block {idx}")
            print("================")
            print("Conv weights", block[0].weight.shape)
            print("BatchNorm weight", block[1].weight.shape)
            print("================")
        print("Conv2 weight", self.conv2.weight.shape)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)

        y = self.body(y)

        y = self.conv2(y)
        return y