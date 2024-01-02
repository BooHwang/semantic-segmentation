import torch
from torch import Tensor
from torch.nn import functional as F
import sys
sys.path.append("/data4/face_parsing_task/val_test/semantic-segmentation")

from semseg.models.base import BaseModel
from semseg.models.heads import UPerHead


class CustomCNN(BaseModel):
    def __init__(self, backbone: str = 'ResNet-50', num_classes: int = 19):
        super().__init__(backbone, num_classes)
        self.decode_head = UPerHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

    
if __name__ == '__main__':
    # model = CustomCNN('ResNet-18', 19)
    # model.init_pretrained('checkpoints/backbones/resnet/resnet18.pth')
    # x = torch.randn(2, 3, 224, 224)
    # y = model(x)
    # print(y.shape)
    
    # model = CustomCNN('ConvNeXt-B', 12)
    # model.init_pretrained('/data1/face_parsing/semantic-segmentation/checkpoints/pretrained/convnext_base_1k_224_ema.pth')
    # x = torch.randn(2, 3, 224, 224)
    # y = model(x)
    # print(y.shape)
    
    # model = CustomCNN('PoolFormer-M36', 12)
    # model.init_pretrained('/data1/face_parsing/semantic-segmentation/checkpoints/pretrained/convnext_base_1k_224_ema.pth')
    # x = torch.randn(2, 3, 512, 512)
    # y = model(x)
    # print(y.shape)
    
    model = CustomCNN('ConvNeXt-L', 12)
    model.init_pretrained('/data4/face_parsing_task/val_test/semantic-segmentation/checkpoints/pretrained/convnext_base_1k_384.pth')
    x = torch.randn(2, 3, 384, 384)
    y = model(x)
    print(y.shape)
    # import time
    # t0 = time.time()
    # for _ in range(20):
    #     y = model(x)
    #     print(y.shape)
    # print(f"all use time: {time.time()-t0:.3f} s")