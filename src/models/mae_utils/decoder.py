from __future__ import annotations

import math

import autoroot  # required to load from src
import torch
from torch import nn

from src.models.unet_utils.utils import ResidualConv
from src.models.mae_utils import masked_autoencoder_satmae, utils

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class RemoveClassToken(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, 1:, :]


class ReshapeToken(nn.Module):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        super().__init__()

    def forward(self, x):
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
    def __init__(self, end_filters, hidden_dim, filters_seq, num_timesteps):
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
        x = self.decode(x)
        x = self.up1(x)
        # NOTE: This does nothing if filter_seq only contains one element (i.e. for vit4)
        for idx in range(len(self.filters_seq) - 1):
            x = getattr(self, "up%d" % int(idx + 2))(x)

        x = self.out(x)
        return x


class VanillaDecoder_Conv2DMultiHead(nn.Module):
    def __init__(
        self, end_filters, hidden_dim, filters_seq, num_timesteps, num_properties=1
    ):
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
        x = self.decode(x)
        x = self.conv_up(x)  # Shape [B, 384, 128, 128] --> [B, 384, 256,256]
        # Apply each property head to the output
        props = [head(x).unsqueeze(1) for head in self.out]
        x = torch.cat(props, dim=1)  # Stack along the property dimension
        return x

class ViTDecoder_Conv2DMultiHead(nn.Module):
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

