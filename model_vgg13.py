import torch
import torch.nn as nn

VGG_types = {
"VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],

"VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],

"VGG13_custom": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 256, 256, 256, "M"],

"VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"],

"VGG19": [64,64,"M",128,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512,"M",],}

VGGType = "VGG13"
# VGGType = "VGG13_custom"

class VGGnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(VGGnet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[VGGType])
        self.fcs = nn.Sequential(
        nn.Linear(512*2*2, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        x = nn.functional.softmax(x, dim=1)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                ),
                nn.BatchNorm2d(x),
                nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

class VGGnet_custom(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(VGGnet_custom, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[VGGType])
        self.fcs = nn.Sequential(
        nn.Linear(256*4*4, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        x = nn.functional.softmax(x, dim=1)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                ),
                nn.ReLU(),
                # nn.Dropout2d(p=0.2),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGnet(in_channels=3, num_classes=2).to(device)
#     # print(model)
#     # x = torch.randn(1, 3, 64, 64).to(device)
#     # print(model(x).shape)


      