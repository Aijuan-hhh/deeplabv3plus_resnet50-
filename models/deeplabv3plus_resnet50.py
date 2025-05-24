# 加上点表示使用相对导入
from .resnet50 import resnet50
from .DeepLabV3Plus import DeepLabHeadV3Plus,DeepLabV3
from .utils import IntermediateLayerGetter


def _segm_resnet(num_classes, output_stride, pretrained_backbone):
    # 输出步长越小，特征图越大，保留的细节越多
    if output_stride == 8:
        replace_stride_with_dilation =[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation =[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet50(
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256


    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(num_classes, output_stride, pretrained_backbone):
    model = _segm_resnet(num_classes, output_stride=output_stride,pretrained_backbone=pretrained_backbone)
    return model

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


if __name__ == '__main__':
    net = deeplabv3plus_resnet50(num_classes=2, output_stride=8)
    print(net)