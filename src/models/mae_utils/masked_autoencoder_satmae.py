from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn as nn

# vision_transformer requires torchvision >= 0.12
from . import utils, vision_transformer
from .pos_embedding import (
    get_2d_sincos_pos_embed,
    get_coords_embedding,
    get_angle_embedding,
    get_fractional_time_embedding,
)
from .vision_transformer import ConvStemConfig


class MaskedAutoEncoderEncoder(vision_transformer.Encoder):
    """Encoder for the Masked Autoencoder model [0].

    Encodes patch embeddings. Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes
    ----------
    seq_length : int. Token sequence length, including the class token.
    num_layers : int. Number of transformer blocks.
    num_heads : int. Number of attention heads.
    hidden_dim : int. Dimension of the input and output tokens.
    mlp_dim : int. Dimension of the MLP in the transformer block.
    dropout : float. Percentage of elements set to zero after the MLP in the transformer.
    attention_dropout : float. Percentage of elements set to zero after the attention head.

    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        """ Initialize MaskedAutoEncoderEncoder.

            Parameters
            ----------
            seq_length : int. Token sequence length, including the class token.
            num_layers : int. Number of transformer blocks.
            num_heads : int. Number of attention heads.
            hidden_dim : int. Dimension of the input and output tokens.
            mlp_dim : int. Dimension of the MLP in the transformer block.
            dropout : float. Percentage of elements set to zero after the MLP in the transformer.
            attention_dropout : float. Percentage of elements set to zero after the attention head.
            norm_layer : Callable. Callable that creates a normalization layer.

            Returns
            -------
            None.
        """
        super().__init__(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )

        # self.num_channels = 1  # Matt

    @classmethod
    def from_vit_encoder(
        cls, vit_encoder: vision_transformer.Encoder
    ) -> MaskedAutoEncoderEncoder:
        """Creates a MaskedAutoEncoderEncoder from a torchvision ViT encoder."""
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        encoder = cls(
            seq_length=1,
            num_layers=1,
            num_heads=1,
            hidden_dim=1,
            mlp_dim=1,
            dropout=0,
            attention_dropout=0,
        )
        encoder.dropout = vit_encoder.dropout
        encoder.layers = vit_encoder.layers
        encoder.ln = vit_encoder.ln
        return encoder

    def forward_mask_layers(
        self, input: torch.Tensor, idx_keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode input tokens.

        Parameters
        ----------
        input : torch.Tensor. Batch of token sequences after addition of positional encoding.
        idx_keep : torch.Tensor or None. Tensor with shape (batch_size, num_tokens_to_keep) where each
            entry is an index of the token to keep in the respective batch.
            If specified, only the indexed tokens will be encoded.

        Returns
        -------
        torch.Tensor. Batch of encoded output tokens.
        """
        if (
            idx_keep is not None
        ):  # Original implementation does not interpolate positional encoding
            input = utils.get_at_index(input, idx_keep)
        # pass through encoder blocks, including layer normalization
        return self.ln(
            self.layers(self.dropout(input))
        )  # shape = [batch_size, (img_size / patch_size)**2 +1, hidden_dim]


class MaskedAutoEncoderBackbone(vision_transformer.VisionTransformer):
    """Backbone for the Masked Autoencoder model [0].

    Converts images into patches and encodes them. Code inspired by [1].
    Note that this implementation uses a learned positional embedding while [0]
    uses a fixed positional embedding.

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae
    - [2]: Early Convolutions Help Transformers See Better, 2021, https://arxiv.org/abs/2106.14881.

    Attributes
    ----------
    image_size : int. Input image size.
    patch_size : int. Width and height of the image patches. image_size must be a multiple of patch_size.
    num_layers : int. Number of transformer blocks.
    num_heads : int. Number of attention heads.
    hidden_dim : int. Dimension of the input and output tokens.
    mlp_dim : int. Dimension of the MLP in the transformer block.
    dropout : float. Percentage of elements set to zero after the MLP in the transformer.
    attention_dropout : float. Percentage of elements set to zero after the attention head.
    num_classes : int. Number of classes for the classification head. Currently not used.
    representation_size : int or None. If specified, an additional linear layer is added before the
        classification head to change the token dimension from hidden_dim to representation_size. Currently not used.
    norm_layer : Callable. Callable that creates a normalization layer.
    conv_stem_configs : list[ConvStemConfig] or None. If specified, a convolutional stem is added at the beginning
        of the network following [2]. Not used in the original Masked Autoencoder paper [0].

    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0,
        attention_dropout: float = 0,
        num_classes: int = 1000,
        representation_size: int | None = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: list[ConvStemConfig] | None = None,
        encode_time: bool = False,
        encode_coords: bool = False,
        encode_sat_angle: bool = False,
        encode_solar_angle: bool = False,
    ):
        """ Initialize MaskedAutoEncoderBackbone.

            Parameters
            ----------
            image_size : int. Input image size.
            patch_size : int. Width and height of the image patches.
            num_layers : int. Number of transformer blocks.
            num_heads : int. Number of attention heads.
            hidden_dim : int. Dimension of the input and output tokens.
            mlp_dim : int. Dimension of the MLP in the transformer block.
            dropout : float. Percentage of elements set to zero after the MLP in the transformer.
            attention_dropout : float. Percentage of elements set to zero after the attention head.
            num_classes : int. Number of classes for the classification head.
            representation_size : int or None. If specified, an additional linear layer is added before the classification head.
            norm_layer : Callable. Callable that creates a normalization layer.
            conv_stem_configs : list[ConvStemConfig] or None. If specified, a convolutional stem is added.
            encode_time : bool. If True, encode time information in the positional embedding.
            encode_coords : bool. If True, encode spatial coordinates in the positional embedding.
            encode_sat_angle : bool. If True, encode satellite angle in the positional embedding.
            encode_solar_angle : bool. If True, encode solar angle in the positional embedding.

            Returns
            -------
            None.
        """
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            representation_size=representation_size,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )
        self.encoder = MaskedAutoEncoderEncoder(
            seq_length=self.seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.num_tokens = self.seq_length - 1
        pos_hidden_dim = self.hidden_dim
        # Split the hidden dimension across the positional, time and spatial embeddings
        self.split_dim = pos_hidden_dim // 8

        if encode_time:
            pos_hidden_dim -= self.split_dim
        if encode_coords:
            pos_hidden_dim -= self.split_dim
        if encode_sat_angle:
            pos_hidden_dim -= self.split_dim
        if encode_solar_angle:
            pos_hidden_dim -= self.split_dim
        self.get_pos_embed(
            pos_hidden_dim=pos_hidden_dim
        )  # Initializes static position embedding

    @classmethod
    def from_vit(
        cls, vit: vision_transformer.VisionTransformer, **kwargs
    ) -> MaskedAutoEncoderBackbone:
        """Creates a MaskedAutoEncoderBackbone from a torchvision ViT model."""
        # Create a new instance with dummy values as they will be overwritten
        # by the copied vit_encoder attributes
        backbone = cls(
            image_size=vit.image_size,
            patch_size=vit.patch_size,
            num_layers=1,
            num_heads=1,
            hidden_dim=vit.hidden_dim,
            mlp_dim=vit.mlp_dim,
            dropout=vit.dropout,
            attention_dropout=vit.attention_dropout,
            num_classes=vit.num_classes,
            representation_size=vit.representation_size,
            norm_layer=vit.norm_layer,
            **kwargs,
        )
        backbone.conv_proj = vit.conv_proj
        backbone.class_token = vit.class_token
        backbone.seq_length = vit.seq_length
        backbone.num_tokens = vit.seq_length - 1
        backbone.heads = vit.heads
        backbone.encoder = MaskedAutoEncoderEncoder.from_vit_encoder(vit.encoder)
        return backbone

    def get_pos_embed(self, pos_hidden_dim: int):
        """ Initialize the static sinusoidal positional embedding.

            Parameters
            ----------
            pos_hidden_dim : int. Dimension of the positional embedding.

            Returns
            -------
            None.
        """
        # Generate the positional embedding values
        pos_embed_values = get_2d_sincos_pos_embed(
            pos_hidden_dim, int(self.num_tokens**0.5), cls_token=True
        )
        pos_embed_values = torch.from_numpy(pos_embed_values).float().unsqueeze(0)

        # Directly create the nn.Parameter with the generated values and set requires_grad=False
        self.pos_embed = nn.Parameter(pos_embed_values, requires_grad=False)

    def images_to_tokens(
        self, images: torch.Tensor, prepend_class_token: bool
    ) -> torch.Tensor:
        """Converts images into patch tokens.

        Parameters
        ----------
        images : torch.Tensor. Tensor with shape (batch_size, channels, image_size, image_size).
        prepend_class_token : bool. If True, prepend the class token to the sequence.

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, sequence_length - 1, hidden_dim)
            containing the patch tokens.
        """
        # image.shape = [batch_size, n_channels, img_size, img_size]
        x = self.conv_proj(images)
        # x.shape = [batch_size, hidden_dim, img_size/patch_size, img_size/patch_size]
        tokens = x.flatten(2).transpose(1, 2)
        # tokens.shape = [batch_size, (img_size/patch_size)**2, hidden_dim]
        if prepend_class_token:
            tokens = utils.prepend_class_token(tokens, self.class_token)
        return tokens

    def forward(
        self, images: torch.Tensor, idx_keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Returns encoded class tokens from a batch of images.

        Parameters
        ----------
        images : torch.Tensor. Tensor with shape (batch_size, channels, image_size, image_size).
        idx_keep : torch.Tensor or None. Tensor with shape (batch_size, num_tokens_to_keep) where each
            entry is an index of the token to keep in the respective batch.
            If specified, only the indexed tokens will be passed to the encoder.

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, hidden_dim) containing the
            encoded class token for every image.

        """
        out = self.encode(images, idx_keep)
        class_token = out[:, 0]
        return class_token

    def encode(
        self,
        images: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        sat_angle: torch.Tensor | None = None,
        solar_angle: torch.Tensor | None = None,
        idx_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns encoded class and patch tokens from images.

        Parameters
        ----------
        images : torch.Tensor. Tensor with shape (batch_size, channels, image_size, image_size).
        timestamps : torch.Tensor or None. Tensor with temporal information for time embedding.
        coords : torch.Tensor or None. Tensor with spatial coordinates for coordinate embedding.
        sat_angle : torch.Tensor or None. Tensor with satellite angle for angle embedding.
        solar_angle : torch.Tensor or None. Tensor with solar angle for angle embedding.
        idx_keep : torch.Tensor or None. Tensor with shape (batch_size, num_tokens_to_keep) where each
            entry is an index of the token to keep in the respective batch.
            If specified, only the indexed tokens will be passed to the encoder.

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, sequence_length, hidden_dim)
            containing the encoded class and patch tokens for every image.

        """
        batch_size, num_imgs, channels, height, width = images.shape
        xs_encoded = []
        for num in range(num_imgs):
            # Only call images_to_tokens function
            x_encoded = self.images_to_tokens(
                images[:, num, :, :, :], prepend_class_token=False
            )  # Don't return class token yet
            xs_encoded.append(x_encoded)
        # Concatenate the encoded representations of all images
        x = torch.cat(
            xs_encoded, dim=1
        )  # Shape = (B, num_imgs*sequence_length, hidden_dim) [4, 3*256, 384]

        # Split the hidden dimension across the positional, time and spatial embeddings
        spacetime_embed = []
        if timestamps is not None:
            # Add time get_fractional_time_embeddings_embedding
            ts_embed = get_fractional_time_embedding(
                timestamps, self.split_dim, self.num_tokens
            ).to(x.device)
            # ts_embed = get_time_embedding(timestamps, self.split_dim, self.num_tokens)
            spacetime_embed.append(ts_embed)

        if coords is not None:
            # Add spatial encoding to pos_embedding
            coords_embed = get_coords_embedding(coords, self.split_dim, self.num_tokens).to(x.device)
            spacetime_embed.append(coords_embed)

        if sat_angle is not None:
            sat_angle_embed = get_angle_embedding(
                sat_angle, self.split_dim, self.num_tokens
            ).to(x.device)
            spacetime_embed.append(sat_angle_embed)

        if solar_angle is not None:
            solar_angle_embed = get_angle_embedding(
                solar_angle, self.split_dim, self.num_tokens
            ).to(x.device)
            spacetime_embed.append(solar_angle_embed)

        # Reshape positional embedding
        pos_embed = (
            self.pos_embed[:, 1:, :].repeat(batch_size, num_imgs, 1).to(x.device)
        )
        # Concatingate embeddings and add to input
        x = x + torch.cat([pos_embed] + spacetime_embed, dim=-1)

        # Append class token
        x = utils.prepend_class_token(x, self.class_token)

        # Call forward_mask_layers function, which only applies mask and passes through the transformer layers
        x_encoded = self.encoder.forward_mask_layers(x, idx_keep)
        return x_encoded


class MaskedAutoEncoderDecoder(vision_transformer.Encoder):
    """Decoder for the Masked Autoencoder model [0].

    Decodes encoded patches and predicts pixel values for every patch.
    Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes
    ----------
    seq_length : int. Token sequence length, including the class token.
    num_layers : int. Number of transformer blocks.
    num_heads : int. Number of attention heads.
    embed_input_dim : int. Dimension of the input tokens. Usually equal to the hidden
        dimension of the MaskedAutoEncoderEncoder or MaskedAutoEncoderBackbone.
    hidden_dim : int. Dimension of the decoder tokens.
    mlp_dim : int. Dimension of the MLP in the transformer block.
    out_dim : int. Output dimension of the prediction for a single patch. Usually equal to (3 * patch_size ** 2).
    dropout : float. Percentage of elements set to zero after the MLP in the transformer.
    attention_dropout : float. Percentage of elements set to zero after the attention head.

    """

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embed_input_dim: int,
        hidden_dim: int,
        mlp_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        encode_time: bool = False,
        encode_coords: bool = False,
        encode_sat_angle: bool = False,
        encode_solar_angle: bool = False,
    ):
        """ Initialize MaskedAutoEncoderDecoder.

            Parameters
            ----------
            seq_length : int. Token sequence length, including the class token.
            num_layers : int. Number of transformer blocks.
            num_heads : int. Number of attention heads.
            embed_input_dim : int. Dimension of the input tokens.
            hidden_dim : int. Dimension of the decoder tokens.
            mlp_dim : int. Dimension of the MLP in the transformer block.
            out_dim : int. Output dimension of the prediction for a single patch.
            dropout : float. Percentage of elements set to zero after the MLP in the transformer.
            attention_dropout : float. Percentage of elements set to zero after the attention head.
            norm_layer : Callable. Callable that creates a normalization layer.
            encode_time : bool. If True, encode time information in the positional embedding.
            encode_coords : bool. If True, encode spatial coordinates in the positional embedding.
            encode_sat_angle : bool. If True, encode satellite angle in the positional embedding.
            encode_solar_angle : bool. If True, encode solar angle in the positional embedding.

            Returns
            -------
            None.
        """
        super().__init__(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.hidden_dim = hidden_dim
        self.decoder_embed = nn.Linear(embed_input_dim, hidden_dim, bias=True)
        self.prediction_head = nn.Linear(hidden_dim, out_dim)
        self.num_tokens = seq_length - 1

        # Split the hidden dimension across the positional, time and spatial embeddings
        pos_hidden_dim = self.hidden_dim
        self.split_dim = pos_hidden_dim // 8
        if encode_time:
            pos_hidden_dim -= self.split_dim
        if encode_coords:
            pos_hidden_dim -= self.split_dim
        if encode_sat_angle:
            pos_hidden_dim -= self.split_dim
        if encode_solar_angle:
            pos_hidden_dim -= self.split_dim
        # Initializes static position embedding
        self.get_pos_embed(pos_hidden_dim)

    def get_pos_embed(self, pos_hidden_dim: int):
        """ Initialize the static sinusoidal positional embedding for the decoder.

            Parameters
            ----------
            pos_hidden_dim : int. Dimension of the positional embedding.

            Returns
            -------
            None.
        """
        # Generate the positional embedding values
        pos_embed_values = get_2d_sincos_pos_embed(
            pos_hidden_dim, int(self.num_tokens**0.5), cls_token=True
        )
        pos_embed_values = torch.from_numpy(pos_embed_values).float().unsqueeze(0)

        # Directly create the nn.Parameter with the generated values and set requires_grad=False
        self.decoder_pos_embed = nn.Parameter(pos_embed_values, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns predicted pixel values from encoded tokens.

        Parameters
        ----------
        input : torch.Tensor. Tensor with shape (batch_size, seq_length, embed_input_dim).

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, seq_length, out_dim).

        """
        out = self.embed(input)
        out = self.decode(out)
        return self.predict(out)

    def embed(self, input: torch.Tensor) -> torch.Tensor:
        """Embeds encoded input tokens into decoder token dimension.

        This is a single linear layer that changes the token dimension from
        embed_input_dim to hidden_dim.

        Parameters
        ----------
        input : torch.Tensor. Tensor with shape (batch_size, seq_length, embed_input_dim)
            containing the encoded tokens.

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the embedded tokens.

        """
        return self.decoder_embed(input)

    def decode(
        self,
        input: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        sat_angle: torch.Tensor | None = None,
        solar_angle: torch.Tensor | None = None,
        num_imgs: int = 1,
    ) -> torch.Tensor:
        """Forward pass through the decoder transformer.

        Parameters
        ----------
        input : torch.Tensor. Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the encoded tokens.
        timestamps : torch.Tensor or None. Tensor with temporal information for time embedding.
        coords : torch.Tensor or None. Tensor with spatial coordinates for coordinate embedding.
        sat_angle : torch.Tensor or None. Tensor with satellite angle for angle embedding.
        solar_angle : torch.Tensor or None. Tensor with solar angle for angle embedding.
        num_imgs : int. Number of images in the sequence.

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        batch_size = input.shape[0]

        spacetime_embed = []

        if timestamps is not None:
            # # Add time encoding to pos_embedding

            ts_embed = get_fractional_time_embedding(
                timestamps, self.split_dim, self.num_tokens
            ).to(input.device)
            # ts_embed = get_time_embedding(timestamps, self.split_dim, self.num_tokens)
            # Add zero vector in the spot of the class token
            ts_embed = torch.cat(
                [
                    torch.zeros(
                        (ts_embed.shape[0], 1, ts_embed.shape[2]),
                        device=ts_embed.device,
                    ),
                    ts_embed,
                ],
                dim=1,
            )
            spacetime_embed.append(ts_embed)

        if coords is not None:
            # Add spatial encoding to pos_embedding
            coords_embed = get_coords_embedding(coords, self.split_dim, self.num_tokens).to(input.device)
            # Add zero vector in the spot of the class token
            coords_embed = torch.cat(
                [
                    torch.zeros(
                        (coords_embed.shape[0], 1, coords_embed.shape[2]),
                        device=coords_embed.device,
                    ),
                    coords_embed,
                ],
                dim=1,
            )
            spacetime_embed.append(coords_embed)

        if sat_angle is not None:
            # Add spatial encoding to pos_embedding
            sat_angle_embed = get_angle_embedding(
                sat_angle, self.split_dim, self.num_tokens
            ).to(input.device)
            # Add zero vector in the spot of the class token
            sat_angle_embed = torch.cat(
                [
                    torch.zeros(
                        (sat_angle_embed.shape[0], 1, sat_angle_embed.shape[2]),
                        device=sat_angle_embed.device,
                    ),
                    sat_angle_embed,
                ],
                dim=1,
            )
            spacetime_embed.append(sat_angle_embed)

        if solar_angle is not None:
            # Add spatial encoding to pos_embedding
            solar_angle_embed = get_angle_embedding(
                solar_angle, self.split_dim, self.num_tokens
            ).to(input.device)
            # Add zero vector in the spot of the class token
            solar_angle_embed = torch.cat(
                [
                    torch.zeros(
                        (solar_angle_embed.shape[0], 1, solar_angle_embed.shape[2]),
                        device=solar_angle_embed.device,
                    ),
                    solar_angle_embed,
                ],
                dim=1,
            )
            spacetime_embed.append(solar_angle_embed)

        # Expand positional embedding (without class token) to match the number of images
        # Re-attach class token to the start of the positional embedding
        pos_embed = torch.cat(
            [
                self.decoder_pos_embed[:, :1, :],
                self.decoder_pos_embed[:, 1:, :].repeat(1, num_imgs, 1),
            ],
            dim=1,
        )
        # Expand positional embedding to match the batch size
        pos_embed = pos_embed.expand(batch_size, -1, -1).to(input.device)
        # Concatingate embeddings and add to input
        input = input + torch.cat([pos_embed] + spacetime_embed, dim=-1)

        return self.ln(self.layers(self.dropout(input)))

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Predicts pixel values from decoded tokens.

        Parameters
        ----------
        input : torch.Tensor. Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        Returns
        -------
        torch.Tensor. Tensor with shape (batch_size, seq_length, out_dim) containing
            predictions for each token.

        """
        return self.prediction_head(input)

