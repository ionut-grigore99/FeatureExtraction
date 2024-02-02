from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small
from torchvision.ops import FeaturePyramidNetwork
import lovely_tensors as lt


class KeypointDetector(nn.Module):
    def __init__(self, resolution=(640, 480), pretrained_backbone=True):
        super().__init__()
        self.res = resolution
        self.rgb = nn.Conv2d(1, 3, 1)
        self.backbone = mobilenet_v3_small(pretrained=pretrained_backbone).features
        self.fpn = FeaturePyramidNetwork([16, 24, 40, 576], out_channels=128, norm_layer=nn.InstanceNorm2d)
        # self.integrate = nn.Conv2d(128, 1,  1)
        self.integrate = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = self.rgb(x)
        feature_maps = OrderedDict()
        for i, feature in enumerate(self.backbone):
            x = feature(x)
            if i in {1, 2, 4, 12}:  # chosen layers
                feature_maps[i] = x
        fpn_feature_maps = self.fpn(feature_maps)
        up = F.interpolate(fpn_feature_maps[12], scale_factor=2, mode='bilinear') + fpn_feature_maps[4]
        print(f"First upscaled feature map: {up}")
        up = F.interpolate(up, scale_factor=2, mode='bilinear') + fpn_feature_maps[2]
        print(f"Second upscaled feature map: {up}")
        up = F.interpolate(up, scale_factor=2, mode='bilinear') + fpn_feature_maps[1]
        print(f"Third upscaled feature map: {up}")
        up = self.integrate(up)
        print(f"Integrated feature map: {up}")
        heatmaps = F.interpolate(up, size=self.res, mode='bicubic')
        print(f"Final heatmaps: {heatmaps}")
        print(58 * '-', 'END', 58 * '-', '\n')

        return heatmaps

    def print_model(self):
        from pytorch_model_summary import summary
        return summary(self, torch.rand(1, 1, *self.res), max_depth=2, show_parent_layers=True, print_summary=True)

    def from_pretrained(self, weights_path):
        ckp = torch.load(weights_path, map_location=torch.device('cpu'))
        state_dict = ckp['model_state_dict'].copy()

        # Remove 'module.' prefix if it's a DataParallel model state_dict
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'module.' in k}

        # Choose correct entries from your model's current state dict
        model_state_dict = self.state_dict()

        # Filter the loaded state_dict to only include keys that exist in the current model and have the same size.
        pretrained_dict = {k: v for k, v in new_state_dict.items() if
                           k in model_state_dict and model_state_dict[k].size() == v.size()}

        # Update the current model's state_dict with the filtered pretrained_dict
        model_state_dict.update(pretrained_dict)

        # Load the updated dictionary back into the model
        self.load_state_dict(model_state_dict)
        self.eval()


if __name__ == '__main__':
    lt.monkey_patch()
    net = KeypointDetector()
    net.print_model()
    net.from_pretrained('/home/ANT.AMAZON.COM/grigiono/Desktop/SP_FPN/src/weights/SP_FPN/20231103-095502/ckp-epoch=3-step=1852.pth')
    y = net(torch.rand(1, 1, 640, 480))
    assert y.shape == (1, 1, 640, 480)
    print('Success!')

