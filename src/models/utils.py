import torch


def get_profiles(cs, cs_p, overpass_mask):
    """
    Extracts profiles from the Clouds and Clouds Prediction tensors based on the overpass mask.

    Args:
        cs (torch.Tensor): Clouds tensor of shape (batch_size, height, padded_length).
        cs_p (torch.Tensor): Clouds Prediction tensor of shape (batch_size, height, width, length).
        overpass_mask (torch.Tensor): Overpass mask tensor of shape (batch_size, width, length).
    """
    batch_size, height, length = cs.shape
    cs_profiles = []
    cs_p_profiles = []
    for item in range(batch_size):
        cs_i = cs[item, :, :]
        cs_p_i = cs_p[item, :, :, :]
        overpass_mask_i = overpass_mask[item, :, :]

        binary_overpass_mask_i = overpass_mask_i > 0
        binary_overpass_mask_i = binary_overpass_mask_i.expand(
            height, -1, -1
        )  # (90, 256, 256)

        cs_profile_i = cs_i[~torch.isnan(cs_i)].reshape([height, -1])
        cs_p_profile_i = cs_p_i[binary_overpass_mask_i].reshape([height, -1])

        cs_profiles.append(cs_profile_i)
        cs_p_profiles.append(cs_p_profile_i)
    return cs_profiles, cs_p_profiles

