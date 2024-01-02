import os
import torch
import argparse
import yaml
import math
import cv2
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from glob import glob
from tqdm import tqdm
import sys
sys.path.append("/data4/face_parsing_task/val_test/semantic-segmentation")

from semseg.models import *
from semseg.datasets import *


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [100, 25, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255],[0,255,0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    l = [1,2,3,4,5,6,7,8,9,10,11]

    for pi in l:
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_parsing_anno_color = cv2.resize(vis_parsing_anno_color, (512, 512), cv2.INTER_NEAREST)
    vis_im = cv2.resize(vis_im, (512, 512))
    vis_im_merge = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.7, vis_parsing_anno_color, 0.3, 0)

    concat_img = np.concatenate([vis_im, vis_im_merge, vis_parsing_anno_color], axis=1)
    # Save result or not
    if save_im:
        parsing_anno = cv2.resize(parsing_anno.astype(np.uint8), (512, 512), cv2.INTER_NEAREST)
        cv2.imwrite(save_path.replace(".jpg", ".png"), parsing_anno)
        # np.save(save_path.replace(".jpg", ".npy"), parsing_anno)
        cv2.imwrite(save_path, concat_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return concat_img


class FaceParsing():
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], cfg["MODEL"]["NUM_CLASS"])
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)
        
        features = seg_map.squeeze().numpy()
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "xxxxxx.png")
        src_img = orig_img.permute(1, 2, 0).numpy()[:, :, ::-1]
        vis_im_merge = vis_parsing_maps(src_img, features, stride=1, save_im=False, save_path=output_path)
        
        return vis_im_merge

    @torch.inference_mode()
    # @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        if os.path.basename(img_fname) == "pornpics%pornpic_forced_selfidrec_close_eye_only_xt%aligned_512%AbigailMac%64198339%d1daee4e6fb259a5eae467808521521f_xt.jpg":
            from thop import profile
            flops, params = profile(self.model, (img, ))
            print(f"#params: {params/1000000:.4f}M, FLOPs: {flops/1000000000:.4f}G")
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)
    
    semseg = FaceParsing(cfg)
    
    img_paths = [y for x in os.walk(test_file) for y in glob(os.path.join(x[0], "*.jpg"))]
    for img_path in tqdm(img_paths):
        segmap = semseg.predict(img_path, cfg['TEST']['OVERLAY'])
        img_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(save_dir, img_name), segmap)