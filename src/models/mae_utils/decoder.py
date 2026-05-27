from __future__ import annotations

import math

import autoroot  # required to load from src
import torch
from torch import nn

from src.models.unet_utils.utils import ResidualConv
from src.models.mae_utils import masked_autoencoder_satmae, utils

class Permute(nn.Module):
    """ Simple module that permutes tensor dimensions. """

    def __init__(self, *dims):
        """ Initialize Permute.

            Parameters
            ----------
            *dims : int. Dimension indices specifying the desired ordering.

            Returns
            -------
            None.
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        """ Permute the dimensions of the input tensor.

            Parameters
            ----------
            x : torch.Tensor. Input tensor to permute.

            Returns
            -------
            torch.Tensor. Tensor with dimensions reordered according to self.dims.
        """
        return x.permute(*self.dims)


class RemoveClassToken(nn.Module):
    """ Module that removes the class token from the sequence. """

    def __init__(self):
        """ Initialize RemoveClassToken.

            Returns
            -------
            None.
        """
        super().__init__()

    def forward(self, x):
        """ Remove the class token from the token sequence.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, N+1, D) including class token.

            Returns
            -------
            torch.Tensor. Tensor of shape (B, N, D) with class token removed.
        """
        return x[:, 1:, :]


class ReshapeToken(nn.Module):
    """ Module that reshapes a token sequence into a spatial feature map. """

    def __init__(self, num_timesteps):
        """ Initialize ReshapeToken.

            Parameters
            ----------
            num_timesteps : int. Number of input timesteps used to compute spatial dimensions.

            Returns
            -------
            None.
        """
        self.num_timesteps = num_timesteps
        super().__init__()

    def forward(self, x):
        """ Reshape token sequence into a spatial feature map.

            Parameters
            ----------
            x : torch.Tensor. Input tensor of shape (B, hd, ps2*nt).

            Returns
            -------
            torch.Tensor. Reshaped tensor of shape (B, hd*nt, ps, ps).
        """
        bs, hd, ps2_nt = x.shape
        ps2 = ps2_nt // self.num_timesteps
        ps = int(math.sqrt(ps2))
        nt = self.num_timesteps
        nt // 2
        x = x.reshape(bs, hd * nt, ps, ps)
        # NOTE: this relies on the fact that we are always predicting the middle timestep
        # out of an odd number of timesteps.
        # x = x[:,:,midpoint,:,:]
        # x = x.squeeze()

        return x


class VanillaDecoder_Conv2DSingleHead(nn.Module):
    """ Decoder with transposed convolution upsampling and a single output head. """

    def __init__(self, end_filters, hidden_dim, filters_seq, num_timesteps):
        """ Initialize VanillaDecoder_Conv2DSingleHead.

            Parameters
            ----------
            end_filters : int. Number of output channels.
            hidden_dim : int. Token hidden dimension.
            filters_seq : list[int]. Channel sizes for the upsampling blocks.
            num_timesteps : int. Number of input timesteps.

            Returns
            -------
            None.
        """
        super().__init__()
        self.end_filters = end_filters
        self.hidden_dim = hidden_dim
        self.filters_seq = filters_seq
        self.num_timesteps = num_timesteps
        self.decode = nn.Sequential(
            RemoveClassToken(),  # [B, 256, 768]
            Permute(0, 2, 1),  # [B, 768, 256]
            ReshapeToken(self.num_timesteps),
        )  # [B , 768, 16, 16]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * self.num_timesteps,
                out_channels=self.filters_seq[0],
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
        )
        for idx in range(len(filters_seq) - 1):
            up_block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.filters_seq[idx],
                    out_channels=self.filters_seq[idx + 1],
                    kernel_size=2,
                    stride=2,
                ),
                nn.ReLU(),
            )
            setattr(self, "up%d" % int(idx + 2), up_block)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=filters_seq[len(filters_seq) - 1],
                out_channels=self.end_filters,
                kernel_size=2,
                stride=2,
            ),
            nn.Tanh(),
        )

    def forward(self, x, timestamps=None, coords=None, sat_angle=None, solar_angle=None):
        """ Decode and upsample encoder tokens to the output image.

            Parameters
            ----------
            x : torch.Tensor. Encoded token sequence from the backbone.
            timestamps : torch.Tensor. Optional temporal encodings.
            coords : torch.Tensor. Optional coordinate encodings.
            sat_angle : torch.Tensor. Optional satellite angle encodings.
            solar_angle : torch.Tensor. Optional solar angle encodings.

            Returns
            -------
            torch.Tensor. Decoded output tensor of shape (B, end_filters, H, W).
        """
        x = self.decode(x)
        x = self.up1(x)
        # NOTE: This does nothing if filter_seq only contains one element (i.e. for vit4)
        for idx in range(len(self.filters_seq) - 1):
            x = getattr(self, "up%d" % int(idx + 2))(x)

        x = self.out(x)
        return x


class VanillaDecoder_Conv2DMultiHead(nn.Module):
    """ Decoder with transposed convolution upsampling and multiple output heads. """

    def __init__(
        self, end_filters, hidden_dim, filters_seq, num_timesteps, num_properties=1
    ):
        """ Initialize VanillaDecoder_Conv2DMultiHead.

            Parameters
            ----------
            end_filters : int. Number of output channels per head.
            hidden_dim : int. Token hidden dimension.
            filters_seq : list[int]. Channel sizes for the upsampling blocks.
            num_timesteps : int. Number of input timesteps.
            num_properties : int. Number of output property heads (optional, default 1).

            Returns
            -------
            None.
        """
        super().__init__()
        self.end_filters = end_filters
        self.hidden_dim = hidden_dim
        self.filters_seq = filters_seq
        self.num_timesteps = num_timesteps
        self.num_properties = num_properties
        self.decode = nn.Sequential(
            RemoveClassToken(),  # [B, 256, 768]
            Permute(0, 2, 1),  # [B, 768, 256]
            ReshapeToken(self.num_timesteps),
        )  # [B , 768, 16, 16]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim * self.num_timesteps,
                out_channels=self.filters_seq[0],
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
        )
        for idx in range(len(filters_seq) - 1):
            up_block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.filters_seq[idx],
                    out_channels=self.filters_seq[idx + 1],
                    kernel_size=2,
                    stride=2,
                ),
                nn.ReLU(),
            )
            setattr(self, "up%d" % int(idx + 2), up_block)

        # NOTE: compared to the single head, the final size needs to be reached before self.out
        self.out = self._output_layers()

    def _output_layers(self):
        """ Build the output ModuleList with one head per property.

            Returns
            -------
            nn.ModuleList. List of sequential output heads.
        """
        out = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualConv(
                        input_dim=self.filters_seq[len(self.filters_seq) - 1],
                        output_dim=self.filters_seq[len(self.filters_seq) - 1],
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.filters_seq[len(self.filters_seq) - 1],
                        out_channels=self.end_filters,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.Tanh(),  # To normalize to [-1, 1]
                )
                for _ in range(self.num_properties)
            ]
        )

        return out

    def forward(self, x, timestamps=None, coords=None, sat_angle=None, solar_angle=None):
        """ Decode, upsample, and apply each output head.

            Parameters
            ----------
            x : torch.Tensor. Encoded token sequence from the backbone.
            timestamps : torch.Tensor. Optional temporal encodings.
            coords : torch.Tensor. Optional coordinate encodings.
            sat_angle : torch.Tensor. Optional satellite angle encodings.
            solar_angle : torch.Tensor. Optional solar angle encodings.

            Returns
            -------
            torch.Tensor. Output tensor of shape (B, num_properties, end_filters, H, W).
        """
        x = self.decode(x)
        x = self.up1(x)
        # NOTE: This does nothing if filter_seq only contains one element (i.e. for vit4)
        for idx in range(len(self.filters_seq) - 1):
            x = getattr(self, "up%d" % int(idx + 2))(x)

        # Apply each property head to the output
        props = [head(x).unsqueeze(1) for head in self.out]
        x = torch.cat(props, dim=1)  # Stack along the property dimension
        return x


class ConvDecoder_Conv2DMultiHead(nn.Module):
    """ Decoder with adaptive conv upsampling (residual blocks) and multiple output heads. """

    def __init__(
        self,
        end_filters,
        hidden_dim,
        num_timesteps,
        num_properties=1,
        image_size=256,
        target_size=256,
        patch_size=4,
    ):
        """ Initialize ConvDecoder_Conv2DMultiHead.

            Parameters
            ----------
            end_filters : int. Number of output channels per head.
            hidden_dim : int. Token hidden dimension.
            num_timesteps : int. Number of input timesteps.
            num_properties : int. Number of output property heads (optional, default 1).
            image_size : int. Input image spatial size (optional, default 256).
            target_size : int. Target output spatial size (optional, default 256).
            patch_size : int. Patch size used for tokenization (optional, default 4).

            Returns
            -------
            None.
        """
        super().__init__()
        self.end_filters = end_filters
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.num_properties = num_properties
        self.image_size = image_size
        self.target_size = target_size
        self.patch_size = patch_size
        self.decode = nn.Sequential(
            RemoveClassToken(),  # [B, 256, 768]
            Permute(0, 2, 1),  # [B, 768, 256]
            ReshapeToken(self.num_timesteps),
        )  # [B , 768, 16, 16]
        # Conv_up layers
        self.conv_up = self._conv_up_layers()
        # Output layers
        self.out = self._output_layers()

    def _conv_up_layers(self):
        """ Build conv transpose and residual upsampling layers.

            Returns
            -------
            nn.Sequential. Sequential upsampling block.
        """
        size = self.image_size // self.patch_size  # Initial size after reshaping
        in_dim = self.hidden_dim * self.num_timesteps
        self.out_dim = (self.hidden_dim * self.num_timesteps) // 2

        up_blocks = []

        while size < self.target_size:
            up_blocks.append(
                nn.ConvTranspose2d(
                    in_channels=in_dim,
                    out_channels=self.out_dim,
                    kernel_size=2,
                    stride=2,
                )
            )
            up_blocks.append(
                ResidualConv(
                    input_dim=self.out_dim,
                    output_dim=self.out_dim,
                    stride=1,
                    padding=1,
                ),
            )
            up_blocks.append(nn.ReLU())
            size *= 2  # The size is doubled after the transpose convolution
            in_dim = self.out_dim
            # Add ReLU only if it's NOT the last block

        conv_up = nn.Sequential(
            *up_blocks,
        )

        return conv_up

    def _output_layers(self):
        """ Build the output ModuleList with one head per property.

            Returns
            -------
            nn.ModuleList. List of sequential output heads.
        """
        out = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualConv(
                        input_dim=self.out_dim,
                        output_dim=self.out_dim,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.out_dim,
                        out_channels=self.end_filters,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.Tanh(),  # To normalize to [-1, 1]
                )
                for _ in range(self.num_properties)
            ]
        )

        return out

    def forward(self, x, timestamps=None, coords=None, sat_angle=None, solar_angle=None):
        """ Decode, upsample, and apply each output head.

            Parameters
            ----------
            x : torch.Tensor. Encoded token sequence from the backbone.
            timestamps : torch.Tensor. Optional temporal encodings.
            coords : torch.Tensor. Optional coordinate encodings.
            sat_angle : torch.Tensor. Optional satellite angle encodings.
            solar_angle : torch.Tensor. Optional solar angle encodings.

            Returns
            -------
            torch.Tensor. Output tensor of shape (B, num_properties, end_filters, H, W).
        """
        x = self.decode(x)
        x = self.conv_up(x)  # Shape [B, 384, 128, 128] --> [B, 384, 256,256]
        # Apply each property head to the output
        props = [head(x).unsqueeze(1) for head in self.out]
        x = torch.cat(props, dim=1)  # Stack along the property dimension
        return x

class ViTDecoder_Conv2DMultiHead(nn.Module):
    """ Decoder using a ViT transformer decoder followed by un-patchification and conv output heads. """

    # Potential TODOs:
    # Test parameters for decoder_dim?
    # Test transition to end_filters?
    # Add linear projection head? --> ViTDecoder_LinearMultiHead
    def __init__(
        self,
        end_filters,
        num_timesteps,
        num_properties=1,
        backbone=None,
        decoder_dim=576,
    ):
        """ Initialize ViTDecoder_Conv2DMultiHead.

            Parameters
            ----------
            end_filters : int. Number of output channels per head.
            num_timesteps : int. Number of input timesteps.
            num_properties : int. Number of output property heads (optional, default 1).
            backbone : MaskedAutoEncoderBackbone. Backbone model (optional, default None).
            decoder_dim : int. Hidden dimension of the ViT decoder (optional, default 576).

            Returns
            -------
            None.
        """
        super().__init__()
        self.end_filters = end_filters
        self.num_timesteps = num_timesteps
        self.num_properties = num_properties
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.hidden_dim = backbone.hidden_dim
        self.decoder_dim = decoder_dim

        self.decoder = masked_autoencoder_satmae.MaskedAutoEncoderDecoder(
            seq_length=backbone.num_tokens+1,
            num_layers=1,
            num_heads=16,
            embed_input_dim=backbone.hidden_dim,
            hidden_dim=self.decoder_dim,
            mlp_dim=self.decoder_dim * 4,
            out_dim=self.end_filters,  # Added from the paper implementation
            dropout=0,
            attention_dropout=0,
            encode_time=False, # NOTE: geospatial encoding turned off for now
            encode_coords=False,
            encode_sat_angle=False,
            encode_solar_angle=False,
        )

        self.decode = nn.Sequential(
            RemoveClassToken(),  # [B, 256, 768]
        )  # [B , 768, 16, 16]
        # Output layers
        self.out_dim = self.decoder_dim // (self.patch_size**2)
        self.out = self._output_layers()


    def _output_layers(self):
        """ Build the output ModuleList with one head per property.

            Returns
            -------
            nn.ModuleList. List of sequential output heads.
        """
        out = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualConv(
                        input_dim=self.out_dim,
                        output_dim=self.end_filters, # NOTE: We're getting to end_filters sooner in this implementation
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.end_filters,
                        out_channels=self.end_filters,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.Tanh(),  # To normalize to [-1, 1]
                )
                for _ in range(self.num_properties)
            ]
        )

        return out

    def forward(self, x, timestamps=None, coords=None, sat_angle=None, solar_angle=None):
        """Run a full decoder forward pass with patch-based reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Encoded token sequence of shape
            ``(B, sequence_length, hidden_dim)``.
        timestamps : torch.Tensor or None, optional
            Temporal metadata forwarded to the decoder (default None).
        coords : torch.Tensor or None, optional
            Coordinate metadata forwarded to the decoder (default None).
        sat_angle : torch.Tensor or None, optional
            Satellite-angle metadata forwarded to the decoder (default None).
        solar_angle : torch.Tensor or None, optional
            Solar-angle metadata forwarded to the decoder (default None).

        Returns
        -------
        torch.Tensor
            Reconstructed property maps of shape
            ``(B, num_properties, end_filters, target_size, target_size)``.
        """
        # build decoder input
        x = self.decoder.embed(x) # Shape [B, squence_length, hidden_dim]
        # decoder forward pass
        x = self.decoder.decode(
            input = x,
            num_imgs=self.num_timesteps,
            timestamps=timestamps,
            coords=coords,
            sat_angle=sat_angle,
            solar_angle=solar_angle
        ) # Shape [B, sequence_length, decoder_dim]
        x = self.decode(x) # To remove the class token
        x = utils.unpatchify(
            x, self.patch_size
        ) # Shape [B, decoder_dim/(patch_size**2), target_size, target_size]

        # Apply each property head to the output
        props = [head(x).unsqueeze(1) for head in self.out]
        x = torch.cat(props, dim=1)  # Stack along the property dimension
        return x

class ViTConvDecoder_Conv2DMultiHead(nn.Module):
    """A decoder combining ViT-based decoding with convolutional upsampling.

    Processes multi-temporal satellite image sequences using a Transformer
    decoder followed by transposed-convolution upsampling, producing
    per-property 2-D output maps.

    Parameters
    ----------
    end_filters : int
        Number of output channels (one per output map).
    num_timesteps : int
        Number of temporal frames in each input sequence.
    num_properties : int, optional
        Number of geophysical properties to predict (default 1).
    backbone : nn.Module
        Pre-trained ViT backbone that exposes ``patch_size``, ``hidden_dim``,
        and ``num_tokens`` attributes.
    image_size : int, optional
        Spatial size of the input imagery in pixels (default 256).
    target_size : int, optional
        Desired spatial size of the reconstructed output (default 256).
    encode_time : bool, optional
        Append sinusoidal time encoding to decoder tokens (default False).
    encode_coords : bool, optional
        Append coordinate encoding to decoder tokens (default False).
    encode_sat_angle : bool, optional
        Append satellite-angle encoding to decoder tokens (default False).
    encode_solar_angle : bool, optional
        Append solar-angle encoding to decoder tokens (default False).
    """

    def __init__(
        self,
        end_filters,
        num_timesteps,
        num_properties=1,
        backbone=None,
        image_size=256,
        target_size=256,
        encode_time=False,
        encode_coords=False,
        encode_sat_angle=False,
        encode_solar_angle=False
    ):
        """Initialise all sub-modules and hyper-parameter attributes.

        Parameters
        ----------
        end_filters : int
            Number of output channels per property head.
        num_timesteps : int
            Number of temporal frames in each input sequence.
        num_properties : int, optional
            Number of geophysical properties to predict.
        backbone : nn.Module
            Pre-trained ViT backbone providing ``patch_size``, ``hidden_dim``,
            and ``num_tokens``.
        image_size : int, optional
            Spatial size of the input imagery in pixels.
        target_size : int, optional
            Desired spatial size of the reconstructed output.
        encode_time : bool, optional
            Forward sinusoidal time encoding into the decoder.
        encode_coords : bool, optional
            Forward coordinate encoding into the decoder.
        encode_sat_angle : bool, optional
            Forward satellite-angle encoding into the decoder.
        encode_solar_angle : bool, optional
            Forward solar-angle encoding into the decoder.
        """
        super().__init__()
        self.end_filters = end_filters
        self.num_timesteps = num_timesteps
        self.num_properties = num_properties
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.hidden_dim = backbone.hidden_dim
        self.image_size = image_size
        self.target_size = target_size

        self.decoder = masked_autoencoder_satmae.MaskedAutoEncoderDecoder(
            seq_length=backbone.num_tokens+1,
            num_layers=1,
            num_heads=16,
            embed_input_dim=backbone.hidden_dim,
            hidden_dim=backbone.hidden_dim,
            mlp_dim=backbone.hidden_dim * 4,
            out_dim=self.end_filters,  # Added from the paper implementation
            dropout=0,
            attention_dropout=0,
            encode_time=encode_time,
            encode_coords=encode_coords,
            encode_sat_angle=encode_sat_angle,
            encode_solar_angle=encode_solar_angle,
        )

        self.decode = nn.Sequential(
            RemoveClassToken(),  # [B, 256, 768]
            Permute(0, 2, 1),  # [B, 768, 256]
            ReshapeToken(self.num_timesteps),
        )  # [B , 768, 16, 16]

        # Conv_up layers
        self.conv_up = self._conv_up_layers()
        # Output layers
        self.out = self._output_layers()

    def _conv_up_layers(self):
        """Build the transposed-convolution upsampling stack.

        Doubles the spatial resolution at each stage with a
        ``ConvTranspose2d`` followed by a ``ResidualConv`` and ``ReLU``
        until ``target_size`` is reached.

        Returns
        -------
        nn.Sequential
            Sequential module that maps
            ``(B, hidden_dim * num_timesteps, H, H)`` to
            ``(B, out_dim, target_size, target_size)``.
        """
        size = self.image_size // self.patch_size  # Initial size after reshaping
        in_dim = self.hidden_dim * self.num_timesteps
        self.out_dim = (self.hidden_dim * self.num_timesteps) // 2

        up_blocks = []

        while size < self.target_size:
            up_blocks.append(
                nn.ConvTranspose2d(
                    in_channels=in_dim,
                    out_channels=self.out_dim,
                    kernel_size=2,
                    stride=2,
                )
            )
            up_blocks.append(
                ResidualConv(
                    input_dim=self.out_dim,
                    output_dim=self.out_dim,
                    stride=1,
                    padding=1,
                ),
            )
            up_blocks.append(nn.ReLU())
            size *= 2  # The size is doubled after the transpose convolution
            in_dim = self.out_dim
            # Add ReLU only if it's NOT the last block

        conv_up = nn.Sequential(
            *up_blocks,
        )

        return conv_up

    def _output_layers(self):
        """Build the per-property output heads.

        Returns
        -------
        nn.ModuleList
            One ``nn.Sequential`` per property, each mapping
            ``(B, out_dim, H, W)`` to ``(B, end_filters, H, W)`` via
            a residual block, ReLU, 1×1 Conv, and Tanh activation.
        """
        # NOTE: We could also implement a linear project head here
        out = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualConv(
                        input_dim=self.out_dim,
                        output_dim=self.out_dim,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.out_dim,
                        out_channels=self.end_filters,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.Tanh(),  # To normalize to [-1, 1]
                )
                for _ in range(self.num_properties)
            ]
        )

        return out

    def forward(self, x, timestamps=None, coords=None, sat_angle=None, solar_angle=None):
        """Run a full decoder forward pass with convolutional upsampling.

        Parameters
        ----------
        x : torch.Tensor
            Encoded token sequence of shape
            ``(B, sequence_length, hidden_dim)``.
        timestamps : torch.Tensor or None, optional
            Temporal metadata forwarded to the decoder (default None).
        coords : torch.Tensor or None, optional
            Coordinate metadata forwarded to the decoder (default None).
        sat_angle : torch.Tensor or None, optional
            Satellite-angle metadata forwarded to the decoder (default None).
        solar_angle : torch.Tensor or None, optional
            Solar-angle metadata forwarded to the decoder (default None).

        Returns
        -------
        torch.Tensor
            Reconstructed property maps of shape
            ``(B, num_properties, end_filters, target_size, target_size)``.
        """
        # build decoder input
        x = self.decoder.embed(x) # Shape [B, squence_length, hidden_dim]
        # decoder forward pass
        x = self.decoder.decode(
            input = x,
            num_imgs=self.num_timesteps,
            timestamps=timestamps,
            coords=coords,
            sat_angle=sat_angle,
            solar_angle=solar_angle,
        ) # Shape [B, sequence_length, decoder_dim]
        x = self.decode(x) # from Shape [B, sequence_length, hidden_dim] to [B, hidden_dim, 128, 128]
        x = self.conv_up(x)  # Shape [B, 384, 64, 64] --> [B, 162, 256, 256]
        # Apply each property head to the output
        props = [head(x).unsqueeze(1) for head in self.out] # Shape [B, num_properties, 80, 256, 256]
        x = torch.cat(props, dim=1)  # Stack along the property dimension
        return x

