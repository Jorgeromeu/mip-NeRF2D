import torch
from torch import Tensor

def image_points(res) -> Tensor:
    """
    Given a resolution, return the pixel centers of the corresponding image
    :param res: resolution
    :return: a res,res,2 array of center-point coords
    """

    xs = (torch.arange(0, res) + 0.5) / res
    ys = (torch.arange(0, res) + 0.5) / res

    x, y = torch.meshgrid(xs, ys, indexing='ij')
    points = torch.dstack((y, x))
    return points

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def sample_bilinear(img: Tensor, p: Tensor):
    n_channels, res, res = img.shape

    p_pixel = p * res

    # get closest 4-pixel intersection point
    center = torch.round(p_pixel).to(int)

    # get indices of coordinates of adjacent x and y pixels
    lo_x_i = center[0] - 1
    hi_x_i = center[0]
    lo_y_i = center[1] - 1
    hi_y_i = center[1]

    # for pixels outside the image, clamp to border pixels
    lo_x_i = clamp(lo_x_i, 0, res - 1)
    hi_x_i = clamp(hi_x_i, 0, res - 1)
    lo_y_i = clamp(lo_y_i, 0, res - 1)
    hi_y_i = clamp(hi_y_i, 0, res - 1)

    # get color of neighboring four pixels
    color_top_left = img[:, hi_y_i, lo_x_i]
    color_top_right = img[:, hi_y_i, hi_x_i]
    color_bot_left = img[:, lo_y_i, lo_x_i]
    color_bot_right = img[:, lo_y_i, hi_x_i]

    # get coords of adjacent pixel centers to compute the interpolation t-values
    lo_x = center[0] - 0.5
    lo_y = center[1] - 0.5

    t_x = p_pixel[0] - lo_x
    t_y = p_pixel[1] - lo_y

    # interpolate bottom pixels along x-axis
    x_color_top = color_top_left * (1 - t_x) + color_top_right * t_x
    x_color_bot = color_bot_left * (1 - t_x) + color_bot_right * t_x

    # interpolate x-interpolations along the y-axis
    color = x_color_top * t_y + x_color_bot * (1 - t_y)

    return color
