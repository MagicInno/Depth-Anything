import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


# 定义命令行参数
if __name__ == '__main__':
    
    # 通过创建ArgumentParser对象来定义你想要程序接受的命令行参
    parser = argparse.ArgumentParser()

    # 使用add_argument方法向这个解析器添加具体的命令参数定义。比如参数的名字类型默认值。

    # 定义了 --img-path ,用户需要通过这个参数来指定图像文件或包含图像文件的目录的路径。参数类型被指定为str，，意味着传递给--img-path的值会被解析为字符串。
    parser.add_argument('--img-path', type=str)
    # 用于指定输出目录的路径，其中处理后的图像将被保存。这个参数也是字符串类型，但与--img-path不同的是，这里提供了一个默认值'./vis_depth'。
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    # transformer 的 encoder 类型，small，base，large
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    # 是否只显示预测结果。
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    # 定义了命令行参数 --grayscale，当这个参数被指定时，程序不会应用彩色调色板渲染深度图。
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    # 负责解析这些输入，将每个参数的值转换为适当的类型，并将它们作为属性存储在返回的args对象中。
    args = parser.parse_args()
    # margin_width在合成图像时，原始图和深度图之间间隔。视觉上分隔两种不同的图像内容
    margin_width = 50
    # 在图像顶部添加标题栏的高度为60像素。这个空间用来放置诸如“Raw image”和“Depth Anything”之类的文本标签，用于标识合成图像中不同部分的内容。
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载预训练的深度估计模型‘DepthAnything’，并将其设置为评估模式。根据args.encoder选择不同的模型配置。
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(args.encoder)).to(DEVICE).eval()

    # 打印出深度学习模型，depth_anything 的总参数数量。
    total_params = sum(param.numel() for param in depth_anything.parameters())
    """
    param.numel()计算每个参数的元素数量（即，一个参数张量中所有元素的总数）。
    sum(param.numel() for param in depth_anything.parameters())遍历所有参数，累加它们的元素数量，得到模型的总参数数量。
    """
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    # 用于图像预处理的变换操作，使用torchvision.transforms.Compose来组合它们。
    # 在将图像输入到深度学习模型之前进行的，以确保图像数据符合模型的输入要求。
    transform = Compose([
        Resize(
            # 调整图像尺寸到指定的宽度和高度（在这个例子中是518x518像素）。
            width=518,
            height=518,
            # 不调整目标（如标签或标注）的尺寸
            resize_target=False,
            # 保持原图像的长宽比，避免因为强制缩放到指定尺寸而造成的形变。
            keep_aspect_ratio=True,
            # 确保调整后的尺寸是14的倍数
            ensure_multiple_of=14,
            # 指定了保持长宽比时调整尺寸的策略，这里的lower_bound可能意味着在保持长宽比的前提下，确保图像的最短边达到指定的尺寸。
            resize_method='lower_bound',
            # 指定图像插值方法为三次插值，这是一种比线性插值更高质量的插值方法，用于图像的缩放。
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        # 对图像进行归一化处理，使用指定的均值mean=[0.485, 0.456, 0.406]和标准差std=[0.229, 0.224, 0.225]对每个颜色通道进行归一化。
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    """
    读取一个文本文件，并将文件中的每一行作为一个元素存储到一个列表中。
    """
    # 如果args.img_path 是一个文件的路径
    if os.path.isfile(args.img_path):
        # 如果文件是txt格式
        if args.img_path.endswith('txt'):
            # with 是打开一个文件， args.img_path文件的路径，r 表示只读， as f 打开的文件会被赋给变量f。
            with open(args.img_path, 'r') as f:
                # f.read() 读取文件f的全部内容，返回一个字符串（包含了文件的所有文本）。 .splitlines()是字符串的方法，根据换行符分割字符串，存入filenames列表。
                filenames = f.read().splitlines()
        # 如果不是txt
        else:
            # 假定这个路径直接指向了一个图像文件。因此，程序将这个单一的文件路径作为列表的唯一元素存入filenames。
            filenames = [args.img_path]
    # 如果args.img_path 不是一个文件：
    else:
        # 使用os.listdir(args.img_path)列出该目录下的所有文件和子目录。
        filenames = os.listdir(args.img_path)
        # 使用列表推导式和os.path.join方法，结合条件过滤（忽略以.开头的文件或目录，通常这些是隐藏文件或特殊目录），生成完整的文件路径列表。
        filenames = [os.path.join(args.img_path, filename) for filename in filenames if not filename.startswith('.')]
        # 最后，对这个列表进行排序，以确保文件处理的顺序是确定的。
        filenames.sort()

    # 创建输出目录。如果目录已经存在，exist_ok=True参数会防止程序因尝试创建已存在的目录而报错。
    os.makedirs(args.outdir, exist_ok=True)

    
    # 对于每个图像，进行读取、颜色空间转换、预处理、模型推理，最后生成深度图。

    # 快速、扩展性强的python进度条库，可以在长循环中添加一个进度提示信息，用户可以看到大约的完成时间、当前进度条和已经过去的时间等信息。
    for filename in tqdm(filenames):
        # 使用cv2.imread函数从filename指定的路径读取图像。读取的图像raw_image存储为一个NumPy数组，其中图像的颜色通道顺序为BGR（蓝、绿、红）。
        raw_image = cv2.imread(filename)
        # 使用cv2.cvtColor函数将图像从BGR颜色空间转换到RGB颜色空间，除以255.0是为了将像素值归一化到0到1的范围内，这是深度学习模型通常期望的输入值范围。
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        # 获取图像尺寸，image.shape返回一个元组，包含图像的高度、宽度和通道数。通过[:2]获取前两个元素，即高度和宽度。
        h, w = image.shape[:2]

        # 将图像数据传递给之前定义的transform，这是一系列预处理步骤（如调整尺寸、归一化等）。
        # transform接受一个字典，并返回一个经过所有预处理步骤后的图像字典。这里，我们只关心处理后的图像，因此通过['image']获取结果。
        image = transform({'image': image})['image']

        # torch.from_numpy(image)：将NumPy数组image转换为PyTorch张量。
        # .unsqueeze(0)：在张量的第0维（即最前面）添加一个额外的维度，将图像张量从形状[高度, 宽度, 通道数]变为[1, 高度, 宽度, 通道数]。
        # 这是因为深度学习模型通常期望批次维度（即一次可以处理多个图像），即使我们只处理一个图像，也需要模拟这个批次维度。
        # .to(DEVICE)：将张量移动到之前定义的设备（如GPU或CPU）上，以便模型可以在适当的硬件上进行计算。
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        # with torch.no_grad():：这是一个上下文管理器，用于暂时禁用梯度计算。
        with torch.no_grad():
            # 调用了深度估计模型depth_anything来处理预处理后的图像image。输出对应的深度图。
            depth = depth_anything(image)

        # 处理了模型输出的深度图，使其适合于可视化或保存为图像文件。

        # 插值调整深度图尺寸
        """
        depth[None]：在深度图张量前添加一个批次维度，因为F.interpolate期望输入的是批次形式。等同于depth.unsqueeze(0)，但更简洁。
        F.interpolate：使用双线性插值（mode='bilinear'）调整深度图的尺寸。
        目标尺寸为(h, w)，即原始输入图像的高度和宽度。这一步确保深度图的尺寸与输入图像相匹配，便于将深度图覆盖或与原图对比显示。
        align_corners=False：在进行插值时，不对齐四角，这通常用于保持图像内容的比例。
        [0, 0]：移除添加的批次维度和通道维度，因为深度图是单通道的，得到一个与原图尺寸相匹配的二维数组。
        """
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        # 归一化深度值并转换为8位整数
        # 首先将深度图的像素值归一化到0到1的范围内（通过减去最小值后除以最大值和最小值的差），然后乘以255将值缩放到0到255的范围。这一步骤是为了将深度值转换为可以作为图像保存的8位格式。
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        # 转换为NumPy数组并改为8位无符号整型
        """
        depth.cpu()：确保深度图张量在CPU上，以便转换为NumPy数组。如果模型运行在GPU上，这一步是必要的。
        .numpy()：将PyTorch张量转换为NumPy数组。
        .astype(np.uint8)：将数组的数据类型转换为8位无符号整数（uint8），这是保存为图像文件常用的数据格式。
        """
        depth = depth.cpu().numpy().astype(np.uint8)

        # 如果指定了灰度输出：
        if args.grayscale:
            """
            当args.grayscale为True时，将深度图转换为灰度图。由于深度图原本是单通道的，使用np.repeat函数在最后一个维度（即通道维度）上复制三次，
            从而将其转换为三通道的灰度图，每个通道的像素值相同。这样做是为了保持图像格式的一致性，因为大多数图像显示和保存函数期望的是三通道的图像格式。
            """
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        # 如果没有指定灰度输出，则应用彩色调色板
        else:
            """
            使用cv2.applyColorMap函数将预先定义的彩色调色板（在这个例子中是COLORMAP_INFERNO）应用到单通道的深度图上。
            这个函数会根据深度值的不同将不同的颜色映射到深度图上，从而生成一个彩色的深度图。这种方式可以更直观地显示不同深度的变化，通常用于增强视觉效果和深度感知。
            """
            depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        # 更新文件名 os.path.basename(filename)获取原始文件名（filename）的基本名称部分，即去掉了路径的文件名。
        # 这是为了在保存处理后的图像时使用原始图像的文件名作为基础，可能会在后面加上一些后缀来区分不同的处理结果。
        filename = os.path.basename(filename)

        # 如果只展示预测结果：
        if args.pred_only:
            # 将处理后的深度图保存到指定的输出目录args.outdir中。文件名是原始图像文件名加上_depth.png后缀。
            """
            使用cv2.imwrite函数来保存图像，它接受一个完整的文件路径和图像数据。os.path.join用于构建保存路径，确保文件被保存在用户指定的目录下。
            """
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_depth.png'), depth)
        # 如果同时展示深度图和原图
        else:
            split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
            combined_results = cv2.hconcat([raw_image, split_region, depth])
            
            caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
            captions = ['Raw image', 'Depth Anything']
            segment_width = w + margin_width
            
            for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)
            
            final_result = cv2.vconcat([caption_space, combined_results])
            
            cv2.imwrite(os.path.join(args.outdir, filename[:filename.rfind('.')] + '_img_depth.png'), final_result)
        
