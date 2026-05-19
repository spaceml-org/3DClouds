import torch

def encode_position_linear(val, encode_dim, n=1000):
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
    val = torch.atleast_1d(val)
    d = encode_dim // 2
    assert (encode_dim % 2) == 0, f'encode_dim length must be even, recieved {encode_dim=}'
    val_radians = 2*torch.pi*(val - min)/(max-min)
    encoding = torch.zeros((val_radians.shape[0], encode_dim))
    encoding[..., ::2] = torch.sin(val_radians).unsqueeze(-1)
    encoding[..., 1::2] = torch.cos(val_radians).unsqueeze(-1)
    return encoding


def encode_position_half_angle(val, encode_dim, min, max):
    val = torch.atleast_1d(val)
    val_radians = torch.pi*(val - min)/(max-min)
    encoding = torch.zeros((val_radians.shape[0], encode_dim))
    encoding[:] = torch.cos(val_radians).unsqueeze(-1)
    return encoding


def encode_position_spherical_angles(zenith, azimuth, encode_dim, zenith_min, zenith_max, azimuth_min, azimuth_max):
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
    scaling_factor = (high - low) / (2 * torch.pi)
    vals_radians = (values - low) / scaling_factor
    mean_radians = torch.atan2(
        torch.nanmean(torch.sin(vals_radians), **kwargs),
        torch.nanmean(torch.cos(vals_radians), **kwargs),
    )
    mean_vals = ((mean_radians % (2*torch.pi)) * scaling_factor) + low
    return mean_vals

