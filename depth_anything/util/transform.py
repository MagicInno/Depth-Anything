import random
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
import cv2
import math

"""
调整给定样本（sample）中的图像、视差图（disparity）和掩码（mask）的大小，以确保它们至少达到指定的尺寸（size），同时保持图像的长宽比不变。
"""
def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    """Rezise the sample to ensure the given size. Keeps aspect ratio.

    Args:
        sample (dict): sample 输入的样本，一个字典，包含至少image（图像）、disparity（视差图）和mask（掩码）三个键。
        size (tuple): image size 一个元组，指定了图像的最小尺寸，格式为(高度, 宽度)。

    Returns:
        tuple: new size
    """

    # 从sample中的disparity图获取当前尺寸（假设disparity和image、mask的尺寸相同）。
    shape = list(sample["disparity"].shape)

    # 如果当前尺寸已经满足size指定的最小尺寸，则直接返回sample，不做任何调整。
    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    # 计算两个维度（高度和宽度）上需要的缩放比例，以确保图像至少达到指定的size。
    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    # 选择两个比例中较大的一个作为最终的缩放比例，以保持长宽比。
    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    # resize
    # 根据计算出的缩放比例和新的尺寸，使用cv2.resize函数调整sample中的image、disparity和mask的大小。

    # 对于image，使用用户指定的image_interpolation_method进行插值。
    sample["image"] = cv2.resize(
        sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method
    )

    # 对于disparity和mask，使用cv2.INTER_NEAREST进行插值，因为这两者通常包含类别标签或其他非连续性数据，最近邻插值可以避免引入不准确的值。

    sample["disparity"] = cv2.resize(
        sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST
    )
    sample["mask"] = cv2.resize(
        sample["mask"].astype(np.float32),
        tuple(shape[::-1]),
        interpolation=cv2.INTER_NEAREST,
    )

    # 在调整mask尺寸后，将其数据类型从float32转换回bool，因为掩码通常用布尔值表示像素是否有效或属于特定的类别
    sample["mask"] = sample["mask"].astype(bool)

    # 函数最后返回调整后的新尺寸，格式为(高度, 宽度)的元组。
    return tuple(shape)


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    # 将输入的数值x调整为最接近x的某个倍数（这个倍数由实例变量self.__multiple_of指定），同时确保调整后的值位于min_val和max_val指定的范围内。
    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        """
        x: 需要被调整的数值。
        min_val: 调整后的数值不得小于此值。默认为0。
        max_val: 调整后的数值不得大于此值。默认为None，表示没有上限。
        """

        # 计算最接近的倍数：使用np.round(x / self.__multiple_of)计算x除以倍数后最接近的整数，然后乘以倍数self.__multiple_of得到最接近x的该倍数的值y。
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        # 处理最大值限制：如果指定了max_val，且通过上述计算得到的y超过了max_val，则使用np.floor代替np.round来向下取整，以确保结果不会超过max_val。
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        # 处理最小值限制：如果通过上述任一步骤计算得到的y小于min_val，则使用np.ceil代替np.round或np.floor来向上取整，以确保结果不会小于min_val。
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    # 计算新的宽度和高度，以便调整图像的尺寸，同时满足一系列条件，如保持长宽比、符合特定的尺寸约束（比如最小或最大尺寸），并确保新尺寸是某个特定倍数。
    def get_size(self, width, height):
        # determine new height and width

        # self.__height和self.__width指定目标尺寸。
        scale_height = self.__height / height
        scale_width = self.__width / width

        # self.__keep_aspect_ratio指示是否需要保持原始图像的长宽比。
        if self.__keep_aspect_ratio:
            # self.__resize_method定义了调整尺寸的方法，如"lower_bound"、"upper_bound"或"minimal"。
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    # 定义了一个__call__方法，这使得定义了此方法的类的实例可以像函数一样被调用。
    # 这个特定的__call__方法用于调整给定样本中图像（以及可选的其他目标，如视差图、深度图、语义分割掩码和其他掩码）的尺寸。
    def __call__(self, sample):

        # 使用self.get_size方法，根据原始图像的宽度和高度计算新的尺寸。get_size方法可能会考虑保持长宽比、满足最小/最大尺寸限制等因素。
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        # 使用cv2.resize调整sample["image"]的尺寸到计算出的新宽度和高度。插值方法由self.__image_interpolation_method指定。
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        # 如果self.__resize_target为True，则对样本中存在的其他目标（如disparity、depth、semseg_mask、mask）也进行尺寸调整。
        if self.__resize_target:

            # 对于disparity和depth，使用最近邻插值（cv2.INTER_NEAREST）以避免引入插值误差。
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )

                # 使用的部分使用torch.nn.functional.interpolate进行调整，这适用于当semseg_mask已经是一个PyTorch张量的情况，允许直接在PyTorch张量上进行插值操作，然后将结果转换回NumPy数组。
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]

            # 对于mask，先将其数据类型转换为float32进行插值，然后可以选择性地将其转换回布尔类型（尽管示例中这部分代码被注释掉了）。
            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        # 方法返回调整尺寸后的样本。
        return sample


# 对图像数据进行归一化处理，即按通道（channel-wise）减去平均值后除以标准差
class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        # 接收两个参数：mean和std，分别是预定的平均值和标准差，这两个参数通常是基于训练数据集计算得到的，且分别对应图像数据的每个通道。
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        # 使得NormalizeImage实例可以被调用。它接收一个包含"image"键的字典sample，然后对sample["image"]进行归一化处理，并返回处理后的sample。
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

# 用于准备样本数据，使其适合作为神经网络的输入。这包括调整图像和其他相关数据的格式和数据类型，确保它们满足深度学习框架的要求。
class PrepareForNet(object):
    """Prepare sample for usage as network input.
    转换图像数据的维度顺序：

深度学习模型通常期望图像数据的维度顺序为(通道数, 高度, 宽度)，而标准图像格式通常是(高度, 宽度, 通道数)。
因此，需要使用np.transpose方法将图像数据的维度从(高度, 宽度, 通道数)转换为(通道数, 高度, 宽度)。
确保数据连续性并转换数据类型：

使用np.ascontiguousarray确保转换后的数组是连续的，这通常是深度学习框架处理数据的一个要求。
将图像和其他相关数据（如掩码、深度图等）转换为np.float32类型。深度学习模型通常在浮点数上进行计算，
而float32是一个常用的格式，因为它提供了足够的精度，同时又能节省内存和计算资源。
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample
