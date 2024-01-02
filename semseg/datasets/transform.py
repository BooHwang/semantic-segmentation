#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image, ImageFilter
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np
import torchvision.transforms.functional as tf
import copy
import math
import cv2
from scipy.io import loadmat


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        # assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            if lb is not None:
                lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        im_lb['im'] = im.crop(crop)
        if lb is not None:
            im_lb['lb'] = lb.crop(crop)
        return im_lb

class Pad(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im, lb):
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            padding = [(W - w)//2, (H - h)//2, W - w - (W- w)//2, H - h - (H - h)//2]
            im = tf.pad(im, padding, padding_mode='constant')
            lb = tf.pad(lb, padding, padding_mode='constant')
        return (im, lb)

class RandomCrop_Padding(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im, lb):
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            padding = [(w - W)//2, (h - H)//2, w - W - (w- W)//2, h - H - (h - H)//2]
            im = tf.pad(im, padding, padding_mode='constant')
            lb = tf.pad(lb, padding, padding_mode='constant')
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return (im.crop(crop), lb.crop(crop))

class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
            # [0'background', 1'skin', 2'nose', 3'eyeglass', 4'left_eye', 5'right_eye', 6'left_brow', 7'right_brow', 8'left_ear', 9'right_ear', 10'mouth', 11'upper_lip', 12'lower_lip', 13'hair', 14'hat', 15'earring', 16'necklace', 17'neck', 18'cloth']
            # [1'skin', 2'l_brow', 3'r_brow', 4'l_eye', 5'r_eye', 6'nose', 7'u_lip', 8'mouth', 9'l_lip', 10'hair']
            im_lb['im'] = im.transpose(Image.FLIP_LEFT_RIGHT)
            if lb is not None:
                flip_lb = np.array(lb)
                lb = copy.deepcopy(flip_lb)
                flip_lb[lb == 2] = 3
                flip_lb[lb == 3] = 2
                flip_lb[lb == 4] = 5
                flip_lb[lb == 5] = 4
                # flip_lb[lb == 8] = 9
                # flip_lb[lb == 9] = 8
                flip_lb = Image.fromarray(flip_lb)
                im_lb['lb'] = flip_lb.transpose(Image.FLIP_LEFT_RIGHT)
            return im_lb

class RandomRotate_V2(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, im_lb):
        img = im_lb['im']
        mask = im_lb['lb']
        rotate_degree = random.normalvariate(0, 1) / 3 * self.degree
        img =  tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                interpolation=tf.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
                shear=0.0,
            )
        if mask is not None:
            mask =  tf.affine(
                    mask,
                    translate=(0, 0),
                    scale=1.0,
                    angle=rotate_degree,
                    interpolation=tf.InterpolationMode.NEAREST,
                    fill=0,
                    shear=0.0,
                )
        im_lb['im'] = img
        im_lb['lb'] = mask
        return im_lb

class RandomTranslate_V2(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, im_lb):
        img = im_lb['im']
        mask = im_lb['lb']
        assert img.size == mask.size
        x_offset = int(random.normalvariate(0, 1) / 3 * self.offset[0])
        y_offset = int(random.normalvariate(0, 1) / 3 * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(
            img,
            y_crop_offset,
            x_crop_offset,
            img.size[1] - abs(y_offset),
            img.size[0] - abs(x_offset),
        )

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        img = tf.pad(cropped_img, padding_tuple, padding_mode="constant")
        mask = tf.affine(
                mask,
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fill=0,
            )
        im_lb['im'] = img
        im_lb['lb'] = mask
        return im_lb

class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        im_lb['im'] = im.resize((w, h), Image.BILINEAR)
        if lb is not None:
            im_lb['lb'] = lb.resize((w, h), Image.NEAREST)
        return im_lb

class RandomRangeScale(object):
    def __init__(self, min_scale1=-1, max_scale1=-1, min_scale2=-1, max_scale2=-1):
        self.min_scale1 = min_scale1
        self.max_scale1 = max_scale1
        self.min_scale2 = min_scale2
        self.max_scale2 = max_scale2

    def __call__(self, img, mask):
        w, h = img.size
        if max(h,w)> 380:
            scale = random.uniform(self.min_scale1, self.max_scale1)
        else:
            scale = random.uniform(self.min_scale2, self.max_scale2)

        return (img.resize((int(scale*w), int(scale*h)), Image.BILINEAR),mask.resize((int(scale*w), int(scale*h)), Image.NEAREST))

class RandomRangeScale_v2(object):
    def __init__(self, min_size,max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.size_list = np.arange(self.min_size,self.max_size, 25)
        self.size_list = self.size_list.tolist()

    def __call__(self, img, lb):
        target_size = random.choice(self.size_list)
        assert img.size == lb.size
        w, h = img.size
        scale = target_size / max(w,h)
        h_t = int(scale * h + 1)
        w_t = int(scale * w + 1)
        img = img.resize((w_t, h_t), Image.BILINEAR)
        lb = lb.resize((w_t, h_t), Image.NEAREST)
        return (img,lb)

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                interpolation=tf.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                interpolation=tf.InterpolationMode.NEAREST,
                fill=0,
                shear=0.0,
            ),
        )

class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        cropped_img = tf.crop(
            img,
            y_crop_offset,
            x_crop_offset,
            img.size[1] - abs(y_offset),
            img.size[0] - abs(x_offset),
        )

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)

        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)

        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

        return (
            # tf.pad(cropped_img, padding_tuple, padding_mode="reflect"),
            # tf.affine(
            #     mask,
            #     translate=(-x_offset, -y_offset),
            #     scale=1.0,
            #     angle=0.0,
            #     shear=0.0,
            #     fillcolor=255,
            # ),
            ### 255 is the ignore_index in CriterionAll
            tf.pad(cropped_img, padding_tuple, padding_mode="constant"),
            tf.affine(
                mask,
                translate=(-x_offset, -y_offset),
                scale=1.0,
                angle=0.0,
                shear=0.0,
                fill=0,
            ),
        )

class Pad(object):
    def __init__(self, imsize):
        self.imsize = imsize

    def __call__(self, img, mask):
        w, h = img.size
        W, H = self.imsize
        padding = [(W - w)//2, (H - h)//2, W - w - (W - w)//2, H - h - (H - h)//2]
        img = tf.pad(img, padding, padding_mode='constant')
        mask = tf.pad(mask, padding, padding_mode='constant')

        return (img, mask)

class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        im_lb['im'] = im
        im_lb['lb'] = lb
        return im_lb

class ColorJitter2(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im, lb):
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return (im,lb)


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs

class RandomAddGaussianNoise(object):
    
    def __init__(self, p=0.5, mean=0.0, variance=1.0, amplitude=15.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p = p

    def __call__(self, im_lb):
        img = im_lb['im']
        if random.random() < self.p:
            img = np.array(img)
            img = img.astype(np.int64)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, c))
            img = N + img
            img[img > 255] = 255                     # 避免有值超过255而反转
            img[img < 0] = 0
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        im_lb['im'] = img
        return im_lb

def motion_blur(image, degree=12, angle=45):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def motion_blur_mat(image, M):
    image = np.array(image)
    KName = './MotionBlurKernel/m_%02d.mat' % M
    k = loadmat(KName)['kernel']
    k = k.astype(np.float32)
    k /= np.sum(k)
    blurred = cv2.filter2D(image, -1, k)
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

class RandomGaussianBlur(object):
    
    def __init__(self, p=0.5, max_r=4, max_scale=4):
        self.max_r = max_r
        self.p = p
        self.scale_factor = math.log(max_scale, 2)

    def __call__(self, im_lb):
        img = im_lb['im']
        h, w = img.size
        n = random.random()
        if n<0.3:
            r = random.randint(2, self.max_r)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))
        elif n<0.5:
            scale_factor = random.randint(1,self.scale_factor)
            scale = 2 ** scale_factor
            img = img.resize((h//scale,w//scale),resample=Image.BILINEAR)
            img = img.resize((h,w),resample=Image.BILINEAR)
        elif n<0.7:
            degree = random.randint(2, 24)
            angle = random.randint(0, 90)
            img = np.array(img) 
            img = motion_blur(img, degree, angle)
            img = Image.fromarray(img)
            # M = random.randint(1,32)
            # img = motion_blur_mat(img, M)
            # img = Image.fromarray(img)
        im_lb['im'] = img
        return im_lb

class Sharpen(object):
    
    def __init__(self):
        a = 0

    def __call__(self, im_lb):
        img = im_lb['im']
        im_lb['im'] = img.filter(ImageFilter.SHARPEN)
        return im_lb

class RandomChangeBitDepth(object):
    def __init__(self, p1=0.2, p2=0.2):
        self.p1 = p1
        self.p2 = p1+p2

    def __call__(self, im_lb):
        img = im_lb['im']
        lb = im_lb['lb']
        n = random.random()
        if n < self.p1:
            img = img.convert("L")
            img = img.convert("RGB")
        elif n < self.p2:
            img = img.convert("P")
            img = img.convert("RGB")
        im_lb['im'] = img
        im_lb['lb'] = lb
        return im_lb

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb

class Compose2(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        image, mask = im_lb['im'], im_lb['lb']
        for comp in self.do_list:
            image, mask = comp(image, mask)
        im_lb = dict(im=image, lb=mask)
        return im_lb