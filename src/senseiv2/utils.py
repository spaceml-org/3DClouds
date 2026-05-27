import torch

def encode_position_linear(val, encode_dim, n=1000):
    """ Encode a scalar position using sinusoidal linear positional encoding.

        Parameters
        ----------
        val : torch.Tensor. Scalar value(s) to encode.
        encode_dim : int. Dimensionality of the output encoding (must be even).
        n : int. Scaling factor for the denominator (optional).

        Returns
        -------
        torch.Tensor. Encoding of shape (val.shape[0], encode_dim).
    """
    val = torch.atleast_1d(val)
    d = encode_dim // 2
    assert (encode_dim % 2) == 0, f'encode_dim length must be even, recieved {encode_dim=}'
    denominator = n**(2*torch.arange(0, d)/d)
    out = torch.outer(val, 1/denominator)
    encoding = torch.zeros((val.shape[0], encode_dim))
    encoding[..., ::2] = torch.sin(out)
    encoding[..., 1::2] = torch.cos(out)
    return encoding


def encode_position_angle(val, encode_dim, min, max):
    """ Encode a circular angular position using sinusoidal encoding scaled to [min, max].

        Parameters
        ----------
        val : torch.Tensor. Angular value(s) to encode.
        encode_dim : int. Dimensionality of the output encoding (must be even).
        min : float. Minimum of the angular range.
        max : float. Maximum of the angular range.

        Returns
        -------
        torch.Tensor. Encoding of shape (val.shape[0], encode_dim).
    """
    val = torch.atleast_1d(val)
    d = encode_dim // 2
    assert (encode_dim % 2) == 0, f'encode_dim length must be even, recieved {encode_dim=}'
    val_radians = 2*torch.pi*(val - min)/(max-min)
    encoding = torch.zeros((val_radians.shape[0], encode_dim))
    encoding[..., ::2] = torch.sin(val_radians).unsqueeze(-1)
    encoding[..., 1::2] = torch.cos(val_radians).unsqueeze(-1)
    return encoding


def encode_position_half_angle(val, encode_dim, min, max):
    """ Encode a half-angle position using cosine encoding scaled to [min, max].

        Parameters
        ----------
        val : torch.Tensor. Half-angle value(s) to encode.
        encode_dim : int. Dimensionality of the output encoding.
        min : float. Minimum of the angular range.
        max : float. Maximum of the angular range.

        Returns
        -------
        torch.Tensor. Encoding of shape (val.shape[0], encode_dim).
    """
    val = torch.atleast_1d(val)
    val_radians = torch.pi*(val - min)/(max-min)
    encoding = torch.zeros((val_radians.shape[0], encode_dim))
    encoding[:] = torch.cos(val_radians).unsqueeze(-1)
    return encoding


def encode_position_spherical_angles(zenith, azimuth, encode_dim, zenith_min, zenith_max, azimuth_min, azimuth_max):
    """ Encode spherical coordinates (zenith + azimuth) into a sinusoidal positional encoding.

        Parameters
        ----------
        zenith : torch.Tensor. Zenith angle values to encode.
        azimuth : torch.Tensor. Azimuth angle values to encode.
        encode_dim : int. Dimensionality of the output encoding (must be a multiple of 3).
        zenith_min : float. Minimum of the zenith range.
        zenith_max : float. Maximum of the zenith range.
        azimuth_min : float. Minimum of the azimuth range.
        azimuth_max : float. Maximum of the azimuth range.

        Returns
        -------
        torch.Tensor. Encoding of shape (val.shape[0], encode_dim).
    """
    assert (encode_dim % 3) == 0, f'encode_dim length must be a multiple of 3, recieved {encode_dim=}'
    assert zenith.shape == azimuth.shape, "zenith and azimuth inputs must have the same shape"
    d = encode_dim // 3
    encoding = torch.cat(
        (
            encode_position_half_angle(zenith, d, zenith_min, zenith_max),
            encode_position_angle(azimuth, d*2, azimuth_min, azimuth_max),
        ),
        -1
    )
    return encoding


def torch_circmean(values, high=2*torch.pi, low=0, **kwargs):
    """ Compute the circular mean of a tensor of angular values.

        Parameters
        ----------
        values : torch.Tensor. Tensor of angular values.
        high : float. Upper bound of the angular range (optional).
        low : float. Lower bound of the angular range (optional).
        **kwargs : passed to torch.nanmean.

        Returns
        -------
        torch.Tensor. The circular mean in [low, high].
    """
    scaling_factor = (high - low) / (2 * torch.pi)
    vals_radians = (values - low) / scaling_factor
    mean_radians = torch.atan2(
        torch.nanmean(torch.sin(vals_radians), **kwargs),
        torch.nanmean(torch.cos(vals_radians), **kwargs),
    )
    mean_vals = ((mean_radians % (2*torch.pi)) * scaling_factor) + low
    return mean_vals

