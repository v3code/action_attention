from torch import nn


class Decoder(nn.Module):
    def __init__(self,
                 *,
                 in_channels: int = 64,
                 hidden_channels: int = 64,
                 out_channels: int = 4,
                 ):
        super(Decoder, self).__init__()
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels,
                               kernel_size=3, stride=(1, 1), padding=1), nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=3, stride=(1, 1), padding=1), nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=3, stride=(1, 1), padding=1), nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=3, stride=(1, 1), padding=1), nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, out_channels,
                               kernel_size=3, stride=(1, 1), padding=1), nn.GELU(),
        )
    def forward(self, x):
        return self.decoder_cnn(x)


if __name__ == '__main__':
    import torch

    decoder = Decoder(in_channels=32, out_channels=4, hidden_channels=32)
    x = torch.randn((10, 32, 32, 32))

    out = decoder(x)

    print("Done")