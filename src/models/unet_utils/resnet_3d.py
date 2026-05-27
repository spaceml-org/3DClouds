# Define a residual block module to use in place of the convolution block:

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning import LightningModule


class ResBlock(LightningModule):
    """ 2D residual block with batch normalisation, LeakyReLU, and dropout. """

    def __init__(self, in_ch, out_ch, p=0.05, use_norm1=True, use_relu1=True):
        """ Initialize ResBlock.

            Parameters
            ----------
            in_ch : int. Number of input channels.
            out_ch : int. Number of output channels.
            p : float. Dropout probability (optional).
            use_norm1 : bool. Whether to apply batch normalisation before the first conv (optional).
            use_relu1 : bool. Whether to apply LeakyReLU before the first conv (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch) if use_norm1 else nn.Identity()
        self.relu1 = nn.LeakyReLU() if use_relu1 else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.dropout = nn.Dropout2d(p=p) # keep 95% at each block

        # Note that the order of the layers is changed
        self.block = nn.Sequential(
            self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2,
        )

        self.identity = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        """ Apply dropout then residual block (identity + main branch).

            Parameters
            ----------
            x : torch.Tensor. Input feature map.

            Returns
            -------
            torch.Tensor. Output of identity branch plus main conv branch.
        """
        x = self.dropout(x)
        return self.identity(x) + self.block(x)

# We also need to define a new input block:
class InputBlock(LightningModule):
    """ Input block applying two convolutions without pre-activation. """

    def __init__(self, in_ch, out_ch):
        """ Initialize InputBlock.

            Parameters
            ----------
            in_ch : int. Number of input channels.
            out_ch : int. Number of output channels.

            Returns
            -------
            None.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.dropout = nn.Dropout2d(p=0.05) # keep 95% at each block

        # Note that the order of the layers is changed
        self.block = nn.Sequential(
            self.conv1, self.norm1, self.relu1, self.conv2,
        )

        self.identity = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        """ Apply dropout then residual block for the input stage.

            Parameters
            ----------
            x : torch.Tensor. Input feature map.

            Returns
            -------
            torch.Tensor. Output of identity branch plus main conv branch.
        """
        x = self.dropout(x)
        return self.identity(x) + self.block(x)

# And now a new encoder and decoder which can use these:
class ResUNetEncoder(LightningModule):
    """ 2D Residual U-Net encoder that returns hierarchical skip-connection features. """

    def __init__(self, input_channels: int = 3, initial_layer_channels: int = 32, depth: int = 4, dropout=None):
        """ Initialize ResUNetEncoder.

            Parameters
            ----------
            input_channels : int. Number of input image channels (optional).
            initial_layer_channels : int. Number of channels in the first encoder block (optional).
            depth : int. Number of encoder stages (optional).
            dropout : float or None. Dropout probability for each ResBlock (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.channels = [initial_layer_channels*2**i for i in range(depth+1)]

        self.input_block = InputBlock(input_channels, initial_layer_channels)

        self.encode_blocks = nn.ModuleList(
            [ResBlock(self.channels[i], self.channels[i+1], p=dropout) for i in range(depth)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """ Encode input and collect skip-connection features at each stage.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, input_channels, H, W).

            Returns
            -------
            list of torch.Tensor. Feature maps from deepest to shallowest (reversed order).
        """
        features = []

        x = self.input_block(x)
        features.append(x)

        for block in self.encode_blocks:
            x = self.pool(x)
            x = block(x)
            features.append(x)
        return features[::-1]

class ResUNetDecoder(LightningModule):
    """ 2D Residual U-Net decoder that upsamples using skip-connection features. """

    def __init__(self, output_channels=3, initial_layer_channels: int = 32, depth: int = 4, dropout=None):
        """ Initialize ResUNetDecoder.

            Parameters
            ----------
            output_channels : int. Number of output image channels (optional).
            initial_layer_channels : int. Number of channels in the shallowest decoder block (optional).
            depth : int. Number of decoder stages (optional).
            dropout : float or None. Dropout probability for each ResBlock (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.channels = [initial_layer_channels*2**i for i in range(depth+1)][::-1]
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(self.channels[i], self.channels[i+1], 2, 2) for i in range(len(self.channels)-1)
        ])
        self.decode_blocks = nn.ModuleList(
            [ResBlock(self.channels[i], self.channels[i+1], p=dropout) for i in range(len(self.channels)-1)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(initial_layer_channels, output_channels, 1),
            nn.Tanh()
        )

    def forward(self, x, features):
        """ Decode bottleneck features using skip connections from the encoder.

            Parameters
            ----------
            x : torch.Tensor. Bottleneck feature map.
            features : list of torch.Tensor. Encoder skip-connection features (shallow-first order).

            Returns
            -------
            torch.Tensor. Reconstructed output of shape (B, output_channels, H, W).
        """
        for upconv, block, feature in zip(self.upconvs, self.decode_blocks, features):
            x = upconv(x)
            x = torch.cat([x, feature], dim=-3)
            x = block(x)
        return self.head(x)

class ResBlock3D(LightningModule):
    """ 3D residual block with batch normalisation, LeakyReLU, and dropout. """

    def __init__(self, in_ch, out_ch, p=0.05, use_norm1=True, use_relu1=True):
        """ Initialize ResBlock3D.

            Parameters
            ----------
            in_ch : int. Number of input channels.
            out_ch : int. Number of output channels.
            p : float. Dropout probability (optional).
            use_norm1 : bool. Whether to apply 3D batch normalisation before the first conv (optional).
            use_relu1 : bool. Whether to apply LeakyReLU before the first conv (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.norm1 = nn.BatchNorm3d(in_ch) if use_norm1 else nn.Identity()
        self.relu1 = nn.LeakyReLU(inplace=True) if use_relu1 else nn.Identity()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_ch)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        self.dropout = nn.Dropout3d(p=p, inplace=True) # keep 95% at each block

        # Note that the order of the layers is changed
        self.block = nn.Sequential(
            self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2,
        )

        self.identity = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """ Apply dropout then 3D residual block.

            Parameters
            ----------
            x : torch.Tensor. Input 3D feature map.

            Returns
            -------
            torch.Tensor. Output of identity branch plus main conv branch.
        """
        x = self.dropout(x)
        return self.identity(x) + self.block(x)

class ResUNet2DTo3DDecoder(LightningModule):
    """ Decoder that lifts 2D encoder features into a 3D volumetric output. """

    def __init__(
        self,
        output_channels: int = 1,
        output_height: int = 80,
        initial_layer_channels: int = 32,
        final_layer_channels: int = 8,
        depth: int = 4,
        dropout=None,
    ):
        """ Initialize ResUNet2DTo3DDecoder.

            Parameters
            ----------
            output_channels : int. Number of output volume channels (optional).
            output_height : int. Vertical (height) dimension of the output volume (optional).
            initial_layer_channels : int. Channels in the deepest 2D encoder feature map (optional).
            final_layer_channels : int. Channels in the shallowest 3D decoder block (optional).
            depth : int. Number of decoder stages (optional).
            dropout : float or None. Dropout probability for each 3D ResBlock (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.output_channels = output_channels
        self.output_height = output_height
        self.initial_layer_channels = initial_layer_channels
        self.final_layer_channels = final_layer_channels
        self.channel_reduction_factor = initial_layer_channels // final_layer_channels
        self.depth = depth
        self.reshape_bridge_height = output_height // (2**depth)

        self.channels = [final_layer_channels*2**i for i in range(depth+1)][::-1]

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose3d(
                    self.channels[0]*self.channel_reduction_factor,
                    self.channels[1],
                    (self.reshape_bridge_height*2, 2, 2),
                    (self.reshape_bridge_height*2, 2, 2)
                )
            ] + [
                nn.ConvTranspose3d(
                    self.channels[i], self.channels[i+1], 2, 2
                ) for i in range(1, len(self.channels)-1)
            ]
        )
        self.decode_blocks = nn.ModuleList(
            [ResBlock3D(self.channels[i], self.channels[i+1], p=dropout) for i in range(len(self.channels)-1)]
        )
        self.head = nn.Sequential(
            nn.Conv3d(final_layer_channels, output_channels, 1),
            nn.Tanh()
        )
        self.height_upconvs = nn.ModuleList(
            [nn.ConvTranspose3d(
                self.channels[i]*self.channel_reduction_factor,
                self.channels[i],
                (self.reshape_bridge_height*2**i, 1, 1),
            ) for i in range(1, len(self.channels))]
        )

    def forward(self, x, features):
        """ Decode 2D encoder features into a 3D volumetric output.

            Parameters
            ----------
            x : torch.Tensor. Bottleneck 2D feature map from the encoder.
            features : list of torch.Tensor. Encoder skip-connection features (shallow-first).

            Returns
            -------
            torch.Tensor. 3D output volume of shape (B, output_channels, output_height, H, W).
        """
        x = x.unsqueeze(2)
        for i, (height_upconv, upconv, block, feature) in enumerate(zip(self.height_upconvs, self.upconvs, self.decode_blocks, features)):
            x = torch.cat([upconv(x), height_upconv(feature.unsqueeze(2))], dim=1)
            x = block(x)
        return self.head(x)


class ResUNet2D(LightningModule):
    """ Full 2D Residual U-Net combining encoder and decoder. """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        initial_layer_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.05,
    ):
        """ Initialize ResUNet2D.

            Parameters
            ----------
            input_channels : int. Number of input image channels (optional).
            output_channels : int. Number of output image channels (optional).
            initial_layer_channels : int. Number of channels in the first encoder block (optional).
            depth : int. Number of encoder/decoder stages (optional).
            dropout : float. Dropout probability for each ResBlock (optional).

            Returns
            -------
            None.
        """
        super().__init__()

        # Define encoder and decoder
        self.encoder = ResUNetEncoder(
            input_channels=input_channels, initial_layer_channels=initial_layer_channels, depth=depth, dropout=dropout
        )
        self.decoder = ResUNetDecoder(
            output_channels=output_channels, initial_layer_channels=initial_layer_channels, depth=depth, dropout=dropout
        )

    def forward(self, x):
        """ Run encoder then decoder for 2D image-to-image prediction.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, input_channels, H, W).

            Returns
            -------
            torch.Tensor. Predicted output of shape (B, output_channels, H, W).
        """
        features = self.encoder(x)
        x = self.decoder(features[0], features[1:])
        return x


class ResUNet2DTo3D(LightningModule):
    """ 2D-to-3D Residual U-Net: encodes 2D inputs and decodes into a 3D volume. """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 1,
        output_height: int = 80,
        initial_layer_channels: int = 32,
        final_layer_channels: int = 8,
        depth: int = 4,
        dropout: float = 0.05,
    ):
        """ Initialize ResUNet2DTo3D.

            Parameters
            ----------
            input_channels : int. Number of input image channels (optional).
            output_channels : int. Number of output volume channels (optional).
            output_height : int. Vertical dimension of the output 3D volume (optional).
            initial_layer_channels : int. Channels in the first encoder block (optional).
            final_layer_channels : int. Channels in the shallowest 3D decoder block (optional).
            depth : int. Number of encoder/decoder stages (optional).
            dropout : float. Dropout probability for each ResBlock (optional).

            Returns
            -------
            None.
        """
        super().__init__()

        # Define encoder and decoder
        self.encoder = ResUNetEncoder(
            input_channels=input_channels, initial_layer_channels=initial_layer_channels, depth=depth, dropout=dropout
        )
        self.decoder = ResUNet2DTo3DDecoder(
            output_channels=output_channels,
            output_height=output_height,
            initial_layer_channels=initial_layer_channels,
            final_layer_channels=final_layer_channels,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, x):
        """ Encode 2D input and decode into a 3D volumetric prediction.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, input_channels, H, W).

            Returns
            -------
            torch.Tensor. 3D output of shape (B, output_height, H, W) (channel dim squeezed).
        """
        features = self.encoder(x)
        x = self.decoder(features[0], features[1:])
        return x.squeeze(1)

