import torch


def get_profiles(cs, cs_p, overpass_mask):
    """ Extract per-sample profiles from cloud and prediction tensors using the overpass mask.

        Parameters
        ----------
        cs : torch.Tensor. Ground truth clouds of shape (B, H, padded_length).
        cs_p : torch.Tensor. Predicted clouds of shape (B, H, W, L).
        overpass_mask : torch.Tensor. Overpass mask of shape (B, W, L).

        Returns
        -------
        tuple of (list of torch.Tensor, list of torch.Tensor). Per-sample ground truth and
        predicted profiles, each of shape (H, N) where N is the number of valid overpass pixels.
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

