import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

    def forward(self, x):

        return x + self.body(x)


class ResidualGroup(nn.Module):

    def __init__(
        self,
        channels,
        blocks=4
    ):
        super().__init__()

        layers = []

        for _ in range(blocks):
            layers.append(
                ResidualBlock(channels)
            )

        layers.append(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

        self.body = nn.Sequential(*layers)

    def forward(self, x):

        return x + self.body(x)


class SemiconRestorationNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=64,
        groups=6,
        blocks_per_group=4
    ):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            features,
            3,
            padding=1
        )

        self.groups = nn.Sequential(
            *[
                ResidualGroup(
                    features,
                    blocks=blocks_per_group
                )
                for _ in range(groups)
            ]
        )

        self.body = nn.Conv2d(
            features,
            features,
            3,
            padding=1
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                out_channels,
                3,
                padding=1
            )
        )

    def forward(self, x):

        shallow = self.head(x)

        deep = self.groups(shallow)

        deep = self.body(deep)

        deep = deep + shallow

        output = self.upsample(deep)

        return output


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    model = SemiconRestorationNet()

    x = torch.randn(
        2,
        1,
        128,
        128
    )

    with torch.no_grad():

        y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    print(
        "Parameters:",
        count_parameters(model)
    )