from typing import List, Optional, Tuple, Union
from loguru import logger
import einops
import torch


class GlobalStats(torch.nn.Module):
    """
    Calculates various percentiles of the input bands and
    concatenates them to the descriptors. This allows SEnSeI
    to learn something about the reflectance values in the
    bands, without having to consider the entire image space
    (would be too memory intensive).
    """

    def __init__(
        self,
        in_features: int,
        percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)  # Convert device to torch.device
        self.percentiles = torch.tensor(percentiles, device=self.device)
        self.in_features = in_features

    def forward(
        self, bands: torch.Tensor, descriptor_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GlobalStats module.

        Args:
            bands: Input bands tensor
            descriptor_enc: Input descriptors tensor

        Returns:
            Updated bands tensor and updated descriptors tensor
        """

        # Calculate percentiles for each band
        batch_size, num_bands, _, _ = bands.shape
        num_percentiles = len(self.percentiles)
        stats = torch.zeros(
            (batch_size, num_bands, num_percentiles), device=bands.device
        )

        # Reshape bands for percentile calculation
        #reshaped_bands = bands.view(batch_size, num_bands, -1)
        reshaped_bands = bands.reshape(batch_size, num_bands, -1)

        # Calculate percentiles for each band
        stats[...] = torch.quantile(reshaped_bands, self.percentiles, dim=2).permute(1,2,0)

        # Concatenate descriptors with calculated percentiles
        new_descriptor_enc = torch.cat((descriptor_enc, stats), dim=2)

        return new_descriptor_enc

    def get_output_size(self) -> int:
        """
        Returns the output size of the GlobalStats module.
        """
        return self.in_features + len(self.percentiles)


class FCLBlock(torch.nn.Module):
    """
    Fully Connected Layer Block (FCLBlock)

    This module defines a sequence of fully connected
    layers with LayerNorm and ReLU activations.

    Args:
        in_features (int): Number of input features. Default is 79.
        blocks (List[int]): List of integers defining the number of
            neurons in each layer. Default is [128, 128, 128].
    """

    def __init__(
        self,
        in_features: int = 79,
        blocks: List[int] = [128, 128, 128],
        skip_connections: bool = True,
    ):
        super().__init__()

        # Create a list to hold the layers
        self.layers = torch.nn.ModuleList()

        # Append input normalization and activation layers
        self.layers.append(torch.nn.LayerNorm(in_features))
        self.layers.append(torch.nn.ReLU())

        # Append the first linear layer
        self.layers.append(torch.nn.Linear(in_features, blocks[0]))
        if skip_connections:
            self.layers.append(torch.nn.Identity())
        # Iterate over the blocks to add subsequent layers
        for i in range(len(blocks) - 1):
            self.layers.append(torch.nn.LayerNorm(blocks[i]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Linear(blocks[i], blocks[i + 1]))
            if skip_connections:
                self.layers.append(torch.nn.Identity())

        # self.forward_layers = torch.nn.Sequential(self.layers)

    def forward(
        self, descriptor_enc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the FCLBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the layers.
        """
        batch_size, num_bands, descriptors_size = descriptor_enc.shape

        # Merge the batch and band dimensions
        descriptor_enc_reshaped = einops.rearrange(
            tensor=descriptor_enc, pattern="b n d -> (b n) d"
        )

        # Iterate over the layers
        for index, layer in enumerate(self.layers):
            # Apply the layer to the reshaped descriptors
            new_descriptor_enc_reshaped = layer(descriptor_enc_reshaped)

            # This is important because in the first iteration, the tensor are
            # not the same shape, so we need to skip the "skip connection"
            condition = (
                new_descriptor_enc_reshaped.shape == descriptor_enc_reshaped.shape
            )

            # Skip connection if the layer is an identity
            if isinstance(layer, torch.nn.Identity) and condition:
                new_descriptor_enc_reshaped = (
                    descriptor_enc_reshaped + new_descriptor_enc_reshaped
                )

            # Update the reshaped descriptors
            descriptor_enc_reshaped = new_descriptor_enc_reshaped
        # descriptor_enc_reshaped = self.forward_layers(descriptor_enc_reshaped)

        # Reshape the descriptors back to the original shape
        new_descriptor_enc = einops.rearrange(
            tensor=descriptor_enc_reshaped,
            pattern="(b n) d -> b n d",
            b=batch_size,
            n=num_bands,
        )

        return new_descriptor_enc

    def get_output_size(self):
        """
        Returns the output size of the FCLBlock.
        """
        return self.layers[-2].out_features


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int = 79,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()

        # Define the list of Transformer Encoder layers
        self.d_model = d_model
        self.layers = torch.nn.Sequential(
            *[
                torch.nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Check if the input descriptor dims are correct
        if in_features != self.get_output_size():
            raise ValueError(
                "Input descriptor dims must be equal to num_heads*dims_per_head"
            )

        if in_features % nhead != 0:
            raise ValueError("Input descriptor dims must be divisible by nhead")

    def forward(self, descriptor_enc):
        # Run through layers
        return self.layers(descriptor_enc)

    def get_output_size(self):
        """
        Returns the output size of the AttentionBlock.
        """
        return self.d_model
    
class AttentionBlockDecoder(torch.nn.Module): # NOTE: Needed for the reverse pass of sensei
    def __init__(
        self,
        in_features: int = 79,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()

        # Define the list of Transformer Encoder layers
        self.d_model = d_model
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Check if the input descriptor dims are correct
        if in_features != self.get_output_size():
            raise ValueError(
                "Input descriptor dims must be equal to num_heads*dims_per_head"
            )

        if in_features % nhead != 0:
            raise ValueError("Input descriptor dims must be divisible by nhead")

    def forward(self, descriptor_enc, memory):
        # Run through layers
        for index, layer in enumerate(self.layers):
            descriptor_enc = layer(descriptor_enc, memory)

        return descriptor_enc

    def get_output_size(self):
        """
        Returns the output size of the AttentionBlock.
        """
        return self.d_model


class BandEmbedding(torch.nn.Module):
    """
    A way of intelligently embedding the band values into the
    output latent space of SEnSeI v2

    This is done by learning a frequency, phase offset, and gain
    as functions of the descriptor vectors, and then using these
    to embed the band values with scaled sinusoidal functions. This
    allows the model to (hypothetically) select where and how to
    encode each band in the latent space.
    """

    def __init__(
        self,
        in_features: int,
        embedding_dims: int,
        head_layer_sizes: List[int],
        skips_heads: bool,
        normalize: bool,
    ):
        super().__init__()

        self.embedding_dims = embedding_dims
        self.head_layer_sizes = head_layer_sizes
        self.skips_heads = skips_heads
        self.in_features = in_features
        self.normalize = normalize

        # Simple FCL feedforward networks to learn the embedding parameters
        self.frequency_head = FCLBlock(
            in_features=in_features, blocks=head_layer_sizes, skip_connections=False
        )

        self.phase_offset_head = FCLBlock(
            in_features=in_features, blocks=head_layer_sizes, skip_connections=False
        )

        self.gain_head = FCLBlock(
            in_features=in_features, blocks=head_layer_sizes, skip_connections=False
        )

        # Optional batch normalization
        if self.normalize:
            self.norm = torch.nn.BatchNorm2d(self.embedding_dims, momentum=0.05)

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the network. Calculates embedding parameters and applies scaled sinusoidal functions.

        Args:
            bands (torch.Tensor): Input tensor representing bands.
            descriptors (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        # Get the device of the bands tensor
        device = bands.device

        # Calculate embedding parameters for all descriptors
        # Add extra dimensions for the width and height of the bands tensor
        # B x C x E x H x W
        freq = einops.rearrange(
            self.frequency_head(descriptor_enc), "b c e -> b c e 1 1"
        )
        phase = einops.rearrange(
            self.phase_offset_head(descriptor_enc), "b c e -> b c e 1 1"
        )
        gains = einops.rearrange(self.gain_head(descriptor_enc), "b c e -> b c e 1 1")

        # Initialize embeddings tensor B x E x H x W
        embeddings = torch.zeros(
            bands.shape[0], self.embedding_dims, bands.shape[-2], bands.shape[-1]
        ).to(device)
        # Add extra dimensions for the embedding dimensions
        bands = einops.rearrange(bands, "b c h w -> b c 1 h w")

        # Calculate embeddings in a loop (slower but more memory efficient)
        # for i in range(bands.shape[1]):
        #     # freq -> B x C x E x 1 x 1 | bands -> B x C x 1 x H x W | phase -> B x C x E x 1 x 1
        #     modulated_signal = freq[:, i, ...] * (bands[:, i, ...] + phase[:, i, ...])
        #     embeddings = embeddings + gains[:, i, ...] * torch.sin(modulated_signal)
        embeddings += (
            gains * torch.sin(freq * bands + phase)
        ).sum(1)

        # Apply normalization if enabled
        if self.normalize:
            embeddings = self.norm(embeddings)

        return embeddings

    def get_output_size(self):
        """
        Get the size of the output embedding.
        Returns:
            int: Size of the output embedding.
        """
        return self.embedding_dims


class SpectralEmbedding(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """
    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32,
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        # Initialize the FCLBlock module
        self.fcl_block = FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks
        ).to(device)

        

        # Initialize the AttentionBlock module
        self.attention_block = AttentionBlock(
            in_features=self.fcl_block.get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device)

        self.embedding_dims = m4_embedding_dims

        # Initialize the BandEmbedding module
        self.band_embedding = BandEmbedding(
            in_features=self.attention_block.get_output_size(),
            embedding_dims=m4_embedding_dims,
            head_layer_sizes=m4_head_layer_sizes,
            skips_heads=m4_skips_heads,
            normalize=m4_normalize,
        ).to(device)

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            bands (torch.Tensor): Input tensor representing bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        # Calculate global statistics
        descriptor_enc = self.global_stats(bands, descriptor_enc)

        # Pass through the FCLBlock
        descriptor_enc = self.fcl_block(descriptor_enc)

        # Pass through the AttentionBlock
        descriptor_enc = self.attention_block(descriptor_enc)

        # Pass through the BandEmbedding
        embeddings = self.band_embedding(bands, descriptor_enc)

        return embeddings, descriptor_enc
    

class SpectralEmbeddingSimple(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        # Initialize the FCLBlock module
        self.fcl_block = FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks
        ).to(device)

        # Initialize the AttentionBlockDecoder module
        self.attention_block = AttentionBlock(
            in_features=self.fcl_block.get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device)

        self.linear = torch.nn.Linear(
            in_features=m3_d_model, # should be self.attention_block.get_output_size(),
            out_features=m4_embedding_dims
        ).to(device)

        self.embedding_dims = m4_embedding_dims

        self.mlp = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
        ).to(device)

        
        self.device = device

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """


        # Calculate global statistics
        descriptor_encoded = self.global_stats(bands, descriptor_enc)

        # Pass through the FCLBlock
        descriptor_encoded = self.fcl_block(descriptor_encoded)

        # Pass through the AttentionBlock
        descriptor_encoded = self.attention_block(descriptor_encoded)

        # Pass through the linear layer
        weight_matrix = self.linear(descriptor_encoded).transpose(1,2)
        bands_transposed = bands.transpose(1, 3)
        b, h, w, k = bands_transposed.shape
        bands_reshaped = bands_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]
        weight_matrix = weight_matrix.transpose(1,2)
        # Calculate bands in final shape
        bands = (bands_reshaped @ weight_matrix).reshape(b, h, w, weight_matrix.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]
        bands = self.mlp(bands.transpose(1,3)).transpose(1,3)

        return bands, descriptor_encoded

class SpectralEmbeddingSimpleA(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32,
        num_head: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        # Initialize the FCLBlock module
        self.fcl_block = FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks
        ).to(device)

        # Initialize the AttentionBlockDecoder module
        self.attention_block = AttentionBlock(
            in_features=self.fcl_block.get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device)
        
        
        self.device = device

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """


        # Calculate global statistics
        descriptor_encoded = self.global_stats(bands, descriptor_enc)

        # Pass through the FCLBlock
        descriptor_encoded = self.fcl_block(descriptor_encoded)

        # Pass through the AttentionBlock
        descriptor_encoded = self.attention_block(descriptor_encoded)

        
        return  descriptor_encoded

class SpectralEmbeddingSimpleB(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        m3_d_model: int = 128,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        
        self.linear = torch.nn.Linear(
            in_features=m3_d_model, # should be self.attention_block.get_output_size(),
            out_features=m4_embedding_dims
        ).to(device)

        self.embedding_dims = m4_embedding_dims

        self.mlp = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
        ).to(device)

        # TODO: include layer normalization after FDL
        # self.layer_norm = torch.nn.LayerNorm(m4_embedding_dims).to(device)

        self.device = device

    def forward(self, bands, descriptor_encoded):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """


        # Pass through the linear layer
        weight_matrix = self.linear(descriptor_encoded).transpose(1,2)
        bands_transposed = bands.transpose(1, 3)
        b, h, w, k = bands_transposed.shape
        bands_reshaped = bands_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]
        weight_matrix = weight_matrix.transpose(1,2)
        # Calculate bands in final shape
        bands = (bands_reshaped @ weight_matrix).reshape(b, h, w, weight_matrix.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]
        bands = self.mlp(bands.transpose(1,3)).transpose(1,3)

        # TODO: include layer normalization after FDL
        # bands = self.layer_norm(bands)

        return bands


class SpectralEmbeddingMulti(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 4,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.num_head = num_head
        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        # Initialize the FCLBlock module
        self.fcl_block = torch.nn.ModuleList([FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks
        ).to(device) for _ in range(num_head)])

        # Initialize the AttentionBlockDecoder module
        self.attention_block = torch.nn.ModuleList([AttentionBlock(
            in_features=self.fcl_block[i].get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device) for i in range(num_head)])

        

        self.linear = torch.nn.ModuleList([torch.nn.Linear(
            in_features=m3_d_model, # should be self.attention_block.get_output_size(),
            out_features=m4_embedding_dims
        ).to(device) for _ in range(num_head)])

        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
        ).to(device) 

        self.relu = torch.nn.ReLU()

        self.embedding_dims = m4_embedding_dims

        
        self.device = device

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        descriptor_encoded_l = []
        weight_matrices = []
        # Calculate global statistics
        descriptor_encoded0 = self.global_stats(bands, descriptor_enc)
        for i in range(self.num_head):
            #logger.info(f"i: {i} out of {self.num_head}")
            

            # Pass through the FCLBlock
            descriptor_encoded = self.fcl_block[i](descriptor_encoded0)

            # Pass through the AttentionBlock
            descriptor_encoded = self.attention_block[i](descriptor_encoded)
            descriptor_encoded_l.append(descriptor_encoded)
            # Pass through the linear layer
            #logger.info(f"descriptor encoded shape: {descriptor_encoded.shape}")
            weight_matrix = self.linear[i](descriptor_encoded).transpose(1,2)
            bands_transposed = bands.transpose(1, 3)
            b, h, w, k = bands_transposed.shape
            bands_reshaped = bands_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]
            weight_matrix = weight_matrix.transpose(1,2)
            weight_matrices.append(weight_matrix)
            if i < self.num_head - 1:
                #weight_matrix = weight_matrix@weight_matrix.transpose(1,2)
                weight_matrix = weight_matrices[i-1]@weight_matrix.transpose(1,2)
                

            # Calculate bands in final shape
            #logger.info(f"bands reshaped shape: {bands_reshaped.shape}")
            #logger.info(f"weight matrix shape: {weight_matrix.shape}")
            bands = (bands_reshaped @ weight_matrix).reshape(b, h, w, weight_matrix.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]
            bands = self.relu(bands)
            
        bands = self.mlp(bands.transpose(1,3)).transpose(1,3)
        return bands, descriptor_encoded_l


class SpectralEmbeddingMultiA(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32, 
        num_head: int = 4,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.num_head = num_head
        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        m2_blocks_bis = m2_blocks.copy()
        m2_blocks_bis[-1] = m4_embedding_dims
        
        
        # Initialize the FCLBlock module
        self.fcl_block1 = torch.nn.ModuleList([FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks
        ).to(device) for _ in range(num_head)])

        self.fcl_block2 = torch.nn.ModuleList([FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks_bis
        ).to(device) for _ in range(num_head-1)])

        
        # Initialize the AttentionBlockDecoder module
        self.attention_block1 = torch.nn.ModuleList([AttentionBlock(
            in_features=self.fcl_block1[i].get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device) for i in range(num_head)])

        
        self.attention_block2 = torch.nn.ModuleList([AttentionBlock(
            in_features=self.fcl_block2[i].get_output_size(),
            d_model=m4_embedding_dims,
            nhead=m3_nhead,
            dim_feedforward=m4_embedding_dims,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device) for i in range(num_head-1)])


        
        self.device = device

    def forward(self, bands, descriptor_enc):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        descriptor_encoded_l1 = []
        descriptor_encoded_l2 = []
        
        # Calculate global statistics
        descriptor_encoded0 = self.global_stats(bands, descriptor_enc)
        for i in range(self.num_head):
            

            # Pass through the FCLBlock
            descriptor_encoded = self.fcl_block1[i](descriptor_encoded0)

            # Pass through the AttentionBlock
            descriptor_encoded = self.attention_block1[i](descriptor_encoded)
            descriptor_encoded_l1.append(descriptor_encoded)
            
            if i < (self.num_head-1):
                descriptor_encoded = self.fcl_block2[i](descriptor_encoded0)
                # Pass through the AttentionBlock
                descriptor_encoded = self.attention_block2[i](descriptor_encoded)
                descriptor_encoded_l2.append(descriptor_encoded)


        return  [descriptor_encoded_l1, descriptor_encoded_l2]

class SpectralEmbeddingMultiB(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        m3_d_model: int = 128,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 4,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.num_head = num_head
        # Initialize the GlobalStats module


        self.linear = torch.nn.ModuleList([torch.nn.Linear(
            in_features=m3_d_model, # should be self.attention_block.get_output_size(),
            out_features=m4_embedding_dims
        ).to(device) for _ in range(num_head)])

        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
        ).to(device) 

        self.relu = torch.nn.ReLU()

        self.embedding_dims = m4_embedding_dims

        
        self.device = device

    def forward(self, bands, descriptor_encoded_l):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        
        weight_matrices = []
        # Calculate global statistics
        
        for i in range(self.num_head):
            
            
            descriptor_encoded =descriptor_encoded_l[0][i]
            # Pass through the linear layer
            #logger.info(f"descriptor encoded shape: {descriptor_encoded.shape}")
            weight_matrix = self.linear[i](descriptor_encoded).transpose(1,2)
            bands_transposed = bands.transpose(1, 3)
            b, h, w, k = bands_transposed.shape
            bands_reshaped = bands_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]
            weight_matrix = weight_matrix.transpose(1,2)
            weight_matrices.append(weight_matrix)
            if i < self.num_head - 1:
                #weight_matrix = weight_matrix@weight_matrix.transpose(1,2)
                weight_matrix = descriptor_encoded_l[1][i]@weight_matrix.transpose(1,2)
                

            # Calculate bands in final shape
            #logger.info(f"bands reshaped shape: {bands_reshaped.shape}")
            #logger.info(f"weight matrix shape: {weight_matrix.shape}")
            bands = (bands_reshaped @ weight_matrix).reshape(b, h, w, weight_matrix.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]
            bands = self.relu(bands)
            
        bands = self.mlp(bands.transpose(1,3)).transpose(1,3)
        return bands

class SpectralEmbeddingReverse(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        # Initialize the FCLBlock module
        self.fcl_block = FCLBlock(
            in_features=self.global_stats.get_output_size(), blocks=m2_blocks
        ).to(device)

        # Initialize the AttentionBlockDecoder module
        self.attention_block = AttentionBlockDecoder(
            in_features=self.fcl_block.get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device)

        # Initialize the MHA layer
        self.mha = torch.nn.MultiheadAttention(embed_dim=m3_d_model, num_heads=8, batch_first=True).to(device)

        # Initialize the BandEmbedding module
        self.band_embedding = BandEmbedding(
            in_features=self.attention_block.get_output_size(),
            embedding_dims=m4_embedding_dims, # self.attention_block.get_output_size(),
            head_layer_sizes=m4_head_layer_sizes,
            skips_heads=m4_skips_heads,
            normalize=m4_normalize,
        ).to(device)

        self.embedding_dims = m4_embedding_dims
        self.device = device

    def forward(self, embedded_bands, descriptor_enc_transformed):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        aux_descriptor_enc = torch.zeros(
            embedded_bands.shape[0],
            embedded_bands.shape[1],
            0,
            device=self.device,
        )

        # Calculate global statistics
        descriptor_decoded = self.global_stats(embedded_bands, aux_descriptor_enc)

        # Pass through the FCLBlock
        descriptor_decoded = self.fcl_block(descriptor_decoded)

        # Pass through the AttentionBlock
        descriptor_decoded = self.attention_block(descriptor_decoded, memory=descriptor_enc_transformed)

        # Pass through the BandEmbedding
        embeddings = self.band_embedding(embedded_bands, descriptor_decoded)
        embeddings_transposed = embeddings.transpose(1, 3)

        _, attn_weights = self.mha(query=descriptor_enc_transformed, key=descriptor_decoded, value=descriptor_decoded)  # out: (Nt_x, B, D)
        attn_weights = attn_weights.transpose(1, 2)

        b, h, w, k = embeddings_transposed.shape
        embeddings_reshaped = embeddings_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]
        
        # Calculate bands in final shape
        embeddings = (embeddings_reshaped @ attn_weights)
        embeddings = embeddings.reshape(b, h, w, attn_weights.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]

        return embeddings
    
class SpectralEmbeddingReverseSimple(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        # Initialize the FCLBlock module
        self.fcl_block = FCLBlock(
            #in_features=self.global_stats.get_output_size(), blocks=m2_blocks
            in_features=m3_d_model, blocks=m2_blocks
        ).to(device)

        # Initialize the AttentionBlockDecoder module
        self.attention_block = AttentionBlockDecoder(
            in_features=self.fcl_block.get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
        ).to(device)

        self.linear = torch.nn.Linear(
            in_features=m3_d_model, # should be self.attention_block.get_output_size(),
            out_features=m4_embedding_dims
        ).to(device)

        self.embedding_dims = m4_embedding_dims


        self.device = device

    def forward(self, embedded_bands, descriptor_enc_transformed):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        aux_descriptor_enc = torch.zeros(
            embedded_bands.shape[0],
            embedded_bands.shape[1],
            0,
            device=self.device,
        )


        # Calculate global statistics
        descriptor_decoded = self.global_stats(embedded_bands, aux_descriptor_enc)
        
        embedded_bands = self.mlp(embedded_bands.transpose(1,3)).transpose(1,3)
        # Pass through the FCLBlock
        #descriptor_decoded = self.fcl_block(descriptor_decoded)
        descriptor_decoded = self.fcl_block(descriptor_enc_transformed)

        # Pass through the AttentionBlock
        descriptor_decoded = self.attention_block(descriptor_decoded, memory=descriptor_enc_transformed)

        

        # Pass through the linear layer
        #weight_matrix = self.linear(descriptor_enc_transformed).transpose(1,2)
        weight_matrix = self.linear(descriptor_decoded).transpose(1,2)

        embedded_bands_transposed = embedded_bands.transpose(1, 3)
        b, h, w, k = embedded_bands_transposed.shape
        embedded_bands_reshaped = embedded_bands_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]

        # Calculate bands in final shape
        bands = (embedded_bands_reshaped @ weight_matrix).reshape(b, h, w, weight_matrix.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]

        return bands

class SpectralEmbeddingReverseMulti(torch.nn.Module):
    """
    The SpectralEmbedding module is a combination of the GlobalStats, FCLBlock, AttentionBlock, and BandEmbedding
    modules. It is used to embed the spectral bands into the latent space of the SEnSeI model.
    """

    def __init__(
        self,
        in_features: int,
        m1_percentiles: Optional[List[float]] = [0.01, 0.1, 0.5, 0.9, 0.99],
        m2_blocks: List[int] = [128, 128, 128],
        m3_d_model: int = 128,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32, 
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 4,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()

        self.num_head = num_head
        # Initialize the GlobalStats module
        self.global_stats = GlobalStats(
            in_features=in_features, percentiles=m1_percentiles, device=device
        )

        m2_blocks_bis = [m4_embedding_dims for _ in range(len(m2_blocks))]
        

        # Initialize the FCLBlock module
        self.fcl_block1 = torch.nn.ModuleList([FCLBlock(
            in_features=m3_d_model, blocks=m2_blocks
        ).to(device) for _ in range(num_head)])

        self.fcl_block2 = torch.nn.ModuleList([FCLBlock(
            in_features=m4_embedding_dims, blocks=m2_blocks_bis
        ).to(device) for _ in range(num_head-1)])

        # Initialize the AttentionBlockDecoder module
        self.attention_block1 = torch.nn.ModuleList([AttentionBlockDecoder(
            in_features=self.fcl_block1[i].get_output_size(),
            d_model=m3_d_model,
            nhead=m3_nhead,
            dim_feedforward=m3_dim_feedforward,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device) for i in range(num_head)])

        self.attention_block2 = torch.nn.ModuleList([AttentionBlockDecoder(
            in_features=self.fcl_block2[i].get_output_size(),
            d_model=m4_embedding_dims,
            nhead=m3_nhead,
            dim_feedforward=m4_embedding_dims,
            dropout=m3_dropout,
            num_layers=m3_num_layers,
        ).to(device) for i in range(num_head-1)])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
            torch.nn.Linear(m4_embedding_dims, m4_embedding_dims),
            torch.nn.ReLU(),
        ).to(device) 

        self.linear = torch.nn.ModuleList([torch.nn.Linear(
            in_features=m3_d_model, # should be self.attention_block.get_output_size(),
            out_features=m4_embedding_dims
        ).to(device) for _ in range(num_head)])

        self.embedding_dims = m4_embedding_dims

        self.relu = torch.nn.ReLU()
        

        self.device = device

    def forward(self, embedded_bands, descriptor_enc_transformed):
        """
        Forward pass through the SpectralEmbedding module.

        Args:
            emedded_bands (torch.Tensor): Input tensor representing embedded bands.
            descriptor_enc (torch.Tensor): Input tensor representing descriptors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output embeddings and descriptors.
        """

        aux_descriptor_enc = torch.zeros(
            embedded_bands.shape[0],
            embedded_bands.shape[1],
            0,
            device=self.device,
        )

        # Calculate global statistics
        descriptor_decoded = self.global_stats(embedded_bands, aux_descriptor_enc)
        embedded_bands = self.mlp(embedded_bands.transpose(1,3)).transpose(1,3)
        weight_matrices = []
        for i in range(self.num_head):
            # Pass through the FCLBlock
            descriptor_decoded = self.fcl_block1[i](descriptor_enc_transformed[0][i])
    
            # Pass through the AttentionBlock
            descriptor_decoded = self.attention_block1[i](descriptor_decoded, memory=descriptor_enc_transformed[0][i])
    
            
    
            # Pass through the linear layer
            weight_matrix = self.linear[i](descriptor_decoded).transpose(1,2)
            weight_matrices.append(weight_matrix)
            embedded_bands_transposed = embedded_bands.transpose(1, 3)
            b, h, w, k = embedded_bands_transposed.shape
            embedded_bands_reshaped = embedded_bands_transposed.reshape(b, h*w, k)  # Reshape to [B, H*W, K]

            if i < self.num_head - 1:
                descriptor_decoded = descriptor_enc_transformed[1][i]
                descriptor_decoded = self.fcl_block2[i](descriptor_decoded)
                descriptor_decoded = self.attention_block2[i](descriptor_decoded, memory=descriptor_enc_transformed[1][i])
                #weight_matrix = weight_matrix@weight_matrix.transpose(1,2)
                weight_matrix = weight_matrix@descriptor_decoded
                
            # Calculate bands in final shape
            embedded_bands = (embedded_bands_reshaped @ weight_matrix).reshape(b, h, w, weight_matrix.shape[-1]).transpose(3,1)  # Reshape back to [B, H, W, D]
            #bands = self.relu(bands)
        return embedded_bands
