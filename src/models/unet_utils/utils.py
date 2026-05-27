# This script contains the classes for the elements
# of the neural network. This includes a definition of
# the individual encoder and decoder blocks of the UNet
# added by a residual convolution.

from __future__ import annotations

# Import modules
import torch
import torch.nn as nn
from torchvision import transforms


# Classes for the model blocks
class ResidualConv(nn.Module):
    """ Residual convolutional block with batch normalisation and LeakyReLU activations. """

    def __init__(self, input_dim, output_dim, stride, padding):
        """ Initialize ResidualConv.

            Parameters
            ----------
            input_dim : int. Number of input channels.
            output_dim : int. Number of output channels.
            stride : int. Stride for the first convolution in the main branch.
            padding : int or str. Padding for the first convolution in the main branch.

            Returns
            -------
            None.
        """
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.LeakyReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        """ Apply the residual convolution block.

            Parameters
            ----------
            x : torch.Tensor. Input feature map.

            Returns
            -------
            torch.Tensor. Output of the residual block (main branch + skip connection).
        """
        x1 = self.conv_block(x)
        x2 = self.conv_skip(x)
        return x1 + x2


class Upsample(nn.Module):
    """ Transposed-convolution upsample block. """

    def __init__(self, input_dim, output_dim, kernel, stride):
        """ Initialize Upsample.

            Parameters
            ----------
            input_dim : int. Number of input channels.
            output_dim : int. Number of output channels.
            kernel : int. Kernel size for the transposed convolution.
            stride : int. Stride for the transposed convolution.

            Returns
            -------
            None.
        """
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        """ Apply transposed convolution to upsample the input.

            Parameters
            ----------
            x : torch.Tensor. Input feature map.

            Returns
            -------
            torch.Tensor. Upsampled feature map.
        """
        return self.upsample(x)


class UpBlock(nn.Module):
    """ Upsampling block that concatenates skip connection and applies a residual convolution. """

    def __init__(self, input_dim, output_dim):
        """ Initialize UpBlock.

            Parameters
            ----------
            input_dim : int. Number of input channels (from lower decoder level).
            output_dim : int. Number of output channels.

            Returns
            -------
            None.
        """
        super().__init__()

        self.up = Upsample(input_dim, output_dim, kernel=2, stride=2)
        self.resconv = ResidualConv(output_dim + output_dim, output_dim, 1, 1)

    def forward(self, x, res):
        """ Upsample, concatenate skip connection, and apply residual convolution.

            Parameters
            ----------
            x : torch.Tensor. Feature map from the previous decoder step.
            res : torch.Tensor. Skip-connection feature map from the encoder.

            Returns
            -------
            torch.Tensor. Output feature map after upsampling and residual convolution.
        """
        x = self.up(x)

        return self.resconv(torch.cat([x, res], dim=1))


class Bridge(nn.Module):
    """ Bottleneck bridge module connecting encoder and decoder. """

    def __init__(self, infilts, outfilts, dropout):
        """ Initialize Bridge.

            Parameters
            ----------
            infilts : int. Number of input channels.
            outfilts : int. Number of output channels.
            dropout : float. Dropout probability.

            Returns
            -------
            None.
        """
        super().__init__()

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(infilts, outfilts, kernel_size=3, padding="same"),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(outfilts),
            torch.nn.Conv2d(outfilts, outfilts, kernel_size=(1, 1), padding="valid"),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(outfilts),
            torch.nn.Dropout2d(dropout),
        )

    def forward(self, x):
        """ Pass input through the bottleneck sequential block.

            Parameters
            ----------
            x : torch.Tensor. Input feature map.

            Returns
            -------
            torch.Tensor. Output feature map from the bottleneck.
        """
        return self.bottleneck(x)


# Class for the 2D Res-Unet
class ResUnet(nn.Module):
    """ 2D Residual U-Net for image-to-image prediction. """

    def __init__(
        self,
        depth=4,
        dropout=0.5,
        start_filters=32,
        end_filters=16,
        input_dims=[22, 128, 128],
    ):
        """ Initialize ResUnet.

            Parameters
            ----------
            depth : int. Number of encoder/decoder levels (optional).
            dropout : float. Dropout probability applied at each down-block (optional).
            start_filters : int. Number of filters in the first encoder block (optional).
            end_filters : int. Number of filters in the final output layer (optional).
            input_dims : list of int. Input tensor dimensions [channels, H, W] (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.depth = depth

        in_filters = input_dims[0]

        for idx in range(depth - 1):
            down_block = torch.nn.Sequential(
                torch.nn.Dropout2d(dropout),
                ResidualConv(in_filters, start_filters, stride=1, padding="same"),
            )
            in_filters = start_filters
            start_filters = start_filters * 2
            setattr(self, "down_block_%d" % idx, down_block)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.bridge = Bridge(in_filters, start_filters, dropout)

        for idx in range(depth - 1):
            up_block = UpBlock(start_filters, in_filters)
            setattr(self, "up_block_%d" % idx, up_block)
            start_filters = in_filters
            in_filters = in_filters // 2

        self.output_layer = nn.Sequential(
            nn.Conv2d(start_filters, end_filters, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """ Run the full encoder-bridge-decoder forward pass.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, C, H, W).

            Returns
            -------
            torch.Tensor. Predicted output of shape (B, end_filters, H, W).
        """
        skip_connections = []

        for idx in range(self.depth - 1):
            intermed = getattr(self, "down_block_%d" % idx)(x)
            skip_connections.append(intermed)
            x = getattr(self, "max_pool")(intermed)

        x = self.bridge(x)

        for idx in range(self.depth - 1):
            res = skip_connections.pop(-1)
            x = getattr(self, "up_block_%d" % idx)(x, res)

        # return getattr(self, "final_crop")(self.output_layer(x))
        return self.output_layer(x)


class Permute(nn.Module):
    """ Module that permutes tensor dimensions. """

    def __init__(self, *dims):
        """ Initialize Permute.

            Parameters
            ----------
            *dims : int. Desired ordering of dimensions.

            Returns
            -------
            None.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        """ Permute the input tensor dimensions.

            Parameters
            ----------
            x : torch.Tensor. Input tensor.

            Returns
            -------
            torch.Tensor. Tensor with permuted dimensions.
        """
        return x.permute(*self.dims)


class Classify(nn.Module):
    """ Reshape module that converts flattened class logits to a per-class volume. """

    def __init__(self, num_classes):
        """ Initialize Classify.

            Parameters
            ----------
            num_classes : int. Number of target classes.

            Returns
            -------
            None.
        """
        super().__init__()

        self.num_classes = num_classes
        self.perm1 = Permute(0, 2, 3, 1)
        self.perm2 = Permute(0, 4, 3, 1, 2)

    def forward(self, x):
        """ Reshape logits into a per-class volume.

            Parameters
            ----------
            x : torch.Tensor. Input of shape (B, C*num_classes, H, W).

            Returns
            -------
            torch.Tensor. Reshaped output of shape (B, num_classes, C, H, W).
        """
        x = self.perm1(x)
        bs, ps, ps, nf = x.shape
        nf = nf // (self.num_classes)
        x = x.reshape(bs, ps, ps, nf, self.num_classes)
        x = self.perm2(x)

        return x

    # Class for the 2D Res-Unet


class ResUnet_seg(nn.Module):
    """ 2D Residual U-Net for semantic segmentation. """

    def __init__(
        self,
        depth=4,
        dropout=0.5,
        start_filters=32,
        end_filters=16,
        input_dims=[22, 128, 128],
        num_classes=9,
    ):
        """ Initialize ResUnet_seg.

            Parameters
            ----------
            depth : int. Number of encoder/decoder levels (optional).
            dropout : float. Dropout probability applied at each down-block (optional).
            start_filters : int. Number of filters in the first encoder block (optional).
            end_filters : int. Number of filters per class in the output layer (optional).
            input_dims : list of int. Input tensor dimensions [channels, H, W] (optional).
            num_classes : int. Number of segmentation classes (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes

        in_filters = input_dims[0]

        for idx in range(depth - 1):
            down_block = torch.nn.Sequential(
                torch.nn.Dropout2d(dropout),
                ResidualConv(in_filters, start_filters, stride=1, padding="same"),
            )
            in_filters = start_filters
            start_filters = start_filters * 2
            setattr(self, "down_block_%d" % idx, down_block)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.bridge = Bridge(in_filters, start_filters, dropout)

        for idx in range(depth - 1):
            up_block = UpBlock(start_filters, in_filters)
            setattr(self, "up_block_%d" % idx, up_block)
            start_filters = in_filters
            in_filters = in_filters // 2

        self.output_layer = nn.Sequential(
            nn.Conv2d(start_filters, end_filters * num_classes, 1, 1),
        )

        self.classify = Classify(num_classes)

    def forward(self, x):
        """ Run the encoder-bridge-decoder forward pass and return per-class logits.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, C, H, W).

            Returns
            -------
            torch.Tensor. Per-class logit volume of shape (B, num_classes, D, H, W).
        """
        skip_connections = []

        for idx in range(self.depth - 1):
            intermed = getattr(self, "down_block_%d" % idx)(x)
            skip_connections.append(intermed)
            x = getattr(self, "max_pool")(intermed)

        x = self.bridge(x)

        for idx in range(self.depth - 1):
            res = skip_connections.pop(-1)
            x = getattr(self, "up_block_%d" % idx)(x, res)

        # return getattr(self, "final_crop")(self.output_layer(x))
        x = self.output_layer(x)
        x = self.classify(x)
        return x


# Classes that divide Unet into an enconder an encoder in order to tokenize within mae
class UnetEncoder(nn.Module):
    """ Encoder half of a U-Net for feature extraction and MAE tokenisation. """

    def __init__(
        self,
        num_channels=11,
        hidden_dim=768,
        filters_seq=[32, 64, 128, 256],
        dropout=0.5,
    ):
        """ Initialize UnetEncoder.

            Parameters
            ----------
            num_channels : int. Number of input image channels (optional).
            hidden_dim : int. Number of channels in the final (bridge) feature map (optional).
            filters_seq : list of int. Channel sizes for each encoder stage (optional).
            dropout : float. Dropout probability applied at each down-block (optional).

            Returns
            -------
            None.
        """
        super().__init__()

        self.max_pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

        self.input_layer = nn.Sequential(
            nn.Conv2d(num_channels, filters_seq[0], 1, 1),
        )

        filters_seq2 = filters_seq + [hidden_dim]
        self.depth = len(filters_seq2)
        for idx in range(self.depth - 1):
            down_block = torch.nn.Sequential(
                torch.nn.Dropout2d(dropout),
                ResidualConv(
                    filters_seq2[idx], filters_seq2[idx + 1], stride=1, padding="same"
                ),
            )
            setattr(self, "down_block_%d" % idx, down_block)

    def forward(self, x):
        """ Encode input through successive down-blocks with max pooling.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, num_channels, H, W).

            Returns
            -------
            torch.Tensor. Encoded feature map at the bottleneck level.
        """
        # skip_connections = []

        x = self.input_layer(x)
        for idx in range(self.depth - 1):
            intermed = getattr(self, "down_block_%d" % idx)(x)
            # skip_connections.append(intermed)
            x = getattr(self, "max_pool")(intermed)

        # return getattr(self, "final_crop")(self.output_layer(x))
        return x


class UpBlock_noskip(nn.Module):
    """ Upsampling block without a skip connection. """

    def __init__(self, input_dim, output_dim):
        """ Initialize UpBlock_noskip.

            Parameters
            ----------
            input_dim : int. Number of input channels.
            output_dim : int. Number of output channels.

            Returns
            -------
            None.
        """
        super().__init__()

        self.up = Upsample(input_dim, output_dim, kernel=2, stride=2)
        self.resconv = ResidualConv(output_dim, output_dim, 1, 1)

    def forward(self, x):
        """ Upsample and apply residual convolution (no skip connection).

            Parameters
            ----------
            x : torch.Tensor. Input feature map.

            Returns
            -------
            torch.Tensor. Output feature map after upsampling and residual convolution.
        """
        x = self.up(x)
        return self.resconv(x)


class UnetDecoder(nn.Module):
    """ Decoder half of a U-Net (no skip connections) for feature reconstruction. """

    def __init__(
        self,
        num_channels=11,
        hidden_dim=768,
        filters_seq=[128, 64, 32],
    ):
        """ Initialize UnetDecoder.

            Parameters
            ----------
            num_channels : int. Number of output image channels (optional).
            hidden_dim : int. Number of channels in the input bottleneck feature map (optional).
            filters_seq : list of int. Channel sizes for each decoder stage (optional).

            Returns
            -------
            None.
        """
        super().__init__()

        filters_seq2 = [hidden_dim] + filters_seq  # +[num_channels]
        self.depth = len(filters_seq2)
        for idx in range(self.depth - 1):
            up_block = UpBlock_noskip(filters_seq2[idx], filters_seq2[idx + 1])
            setattr(self, "up_block_%d" % idx, up_block)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters_seq2[-1], num_channels, 1, 1),
        )

    def forward(self, x):
        """ Decode bottleneck features back to the output resolution.

            Parameters
            ----------
            x : torch.Tensor. Bottleneck feature map of shape (B, hidden_dim, H, W).

            Returns
            -------
            torch.Tensor. Reconstructed output of shape (B, num_channels, H', W').
        """
        for idx in range(self.depth - 1):
            # res = skip_connections.pop(-1)
            x = getattr(self, "up_block_%d" % idx)(x)

        # return getattr(self, "final_crop")(self.output_layer(x))
        return self.output_layer(x)

        # return getattr(self, "final_crop")(self.output_layer(x))
        return x

