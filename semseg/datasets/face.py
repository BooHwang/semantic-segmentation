#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from numpy import ma
import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2
import copy

from semseg.datasets.transform import *
from semseg.datasets import color_transformer

class FaceMask(Dataset):
    def __init__(self, imsize, rootpth, rootpth2='', rootpth3='', synthetics_pth='', cropsize=(640, 480), mode='train', hands_datasets_path='', add_other_class=False, muticlass_path="", extra_background_path='', num_class=70, add_edge=False, ratio=2, test=False, *args, **kwargs):
        super(FaceMask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.ratio = ratio
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth
        self.rootpth2 = rootpth2
        self.synthetics_pth = synthetics_pth
        self.extra_background_path = extra_background_path
        self.num = 0
        self.imgs = []
        self.masks = []
        self.imgs_synthetics = []
        self.masks_synthetics = []
        self.imsize = imsize
        self.celeb_num = 0
        # self.imgs = os.listdir(os.path.join(self.rootpth, mode+'_img3'))
        with open("/data4/face_parsing_task/val_test/faceparsing_training_data/cvpr_bad_eye_label.txt", "r") as f:
            cvpr_lines = f.readlines()
        cvpr_lines = [line.strip() for line in cvpr_lines]
        if not test:
            # if mode == "val":
            for p in os.listdir(os.path.join(self.rootpth, "images", mode)):
                self.imgs.append(os.path.join(self.rootpth, "images", mode, p))
                self.masks.append(os.path.join(self.rootpth, "annotations", mode, p.replace(".jpg", ".png")))
            if mode == "val":
                for p in os.listdir(os.path.join(self.rootpth2, "images", mode)):
                    self.imgs.append(os.path.join(self.rootpth2, "images", mode, p))
                    self.masks.append(os.path.join(self.rootpth2, "annotations", mode, p.replace(".jpg", ".png")))
                # bad_case_root = "/data4/hb/face_parsing/faceparsing_training_data/select_pornpics_occlusion_badcase_new_align_1221"
                # for p in os.listdir(os.path.join(bad_case_root, "images", mode)):
                #     self.imgs.append(os.path.join(bad_case_root, "images", mode, p))
                #     self.masks.append(os.path.join(bad_case_root, "annotations", mode, p.replace(".jpg", ".png")))
        else:
            for p in os.listdir(os.path.join(self.rootpth, 'Images')):
                self.imgs.append(os.path.join(self.rootpth, 'Images', p))
                self.masks.append(os.path.join(self.rootpth, 'Masks', p[:-3]+'png'))

        if mode == "train":
            # for p in os.listdir("/home/allan/Datasets/LAPA/images_crop_labelled_withouteyeg"):
            #     self.imgs.append(os.path.join("/home/allan/Datasets/LAPA/images_crop_labelled_withouteyeg", p))
            #     name = p.split(".")[0]
            #     self.masks.append(os.path.join("/home/allan/Datasets/LAPA/labels_crop_final", name+'.png'))
            # for p in os.listdir(os.path.join(self.rootpth2, mode+'_crop_no_glasses', 'Images')):
            #     self.imgs.append(os.path.join(self.rootpth2, mode+'_crop_no_glasses', 'Images', p))
            #     self.masks.append(os.path.join(self.rootpth2, mode+'_crop_no_glasses', 'Labels', p[:-3]+'png'))
            # CelebAMask的遮挡集合
            occulusion_path = "/data4/face_parsing_task/val_test/faceparsing_training_data/CelebMask-HQ-Occulusion_new_align_1221"
            for p in os.listdir(os.path.join(occulusion_path, "images", mode)):
                self.imgs.append(os.path.join(occulusion_path, "images", mode, p))
                self.masks.append(os.path.join(occulusion_path, "annotations", mode, p.replace(".jpg", ".png")))
            # CelebAMask的测试集
            for p in os.listdir(os.path.join(self.rootpth, "images", "test")):
                self.imgs.append(os.path.join(self.rootpth, "images", "test", p))
                self.masks.append(os.path.join(self.rootpth, "annotations", "test", p.replace(".jpg", ".png")))
            self.celeb_num = len(self.imgs)
            # cvpr数据集
            for p in os.listdir(os.path.join(rootpth3, "images", mode)):
                if p in cvpr_lines:
                    continue
                self.imgs.append(os.path.join(rootpth3, "images", mode, p))
                self.masks.append(os.path.join(rootpth3, "annotations", mode, p.replace(".jpg", ".png")))
            for p in os.listdir(os.path.join(rootpth3, "images", "val")):
                if p in cvpr_lines:
                    continue
                self.imgs.append(os.path.join(rootpth3, "images", "val", p))
                self.masks.append(os.path.join(rootpth3, "annotations", "val", p.replace(".jpg", ".png")))
            self.normal_sample_num = len(self.imgs)
            # pornpics以及douyin等数据
            for p in os.listdir(os.path.join(self.rootpth2, "images", mode)):
                self.imgs.append(os.path.join(self.rootpth2, "images", mode, p))
                self.masks.append(os.path.join(self.rootpth2, "annotations", mode, p.replace(".jpg", ".png")))
            # 多人同时存在的数据
            multiperson_root = "/data4/face_parsing_task/val_test/faceparsing_training_data/multiperson_new_align_1221"
            for p in os.listdir(os.path.join(multiperson_root, "images", mode)):
                self.imgs.append(os.path.join(multiperson_root, "images", mode, p))
                self.masks.append(os.path.join(multiperson_root, "annotations", mode, p.replace(".jpg", ".png")))
            
            # bad_case_root = "/data4/hb/face_parsing/faceparsing_training_data/select_pornpics_occlusion_badcase_new_align_1221"
            # for p in os.listdir(os.path.join(bad_case_root, "images", mode)):
            #     self.imgs.append(os.path.join(bad_case_root, "images", mode, p))
            #     self.masks.append(os.path.join(bad_case_root, "annotations", mode, p.replace(".jpg", ".png")))
            # bad_case_root2 = "/data4/hb/face_parsing/faceparsing_training_data/select_pornpics_occlusion_badcase2_new_align_1221"
            # for p in os.listdir(os.path.join(bad_case_root2, "images", mode)):
            #     self.imgs.append(os.path.join(bad_case_root2, "images", mode, p))
            #     self.masks.append(os.path.join(bad_case_root2, "annotations", mode, p.replace(".jpg", ".png")))
            
            # before_badcase_root = "/data4/hb/face_parsing/faceparsing_training_data/FM-basecases_1000_packages/train_data"
            # for p in os.listdir(os.path.join(before_badcase_root, 'Images')):
            #     self.imgs.append(os.path.join(before_badcase_root, 'Images', p))
            #     self.masks.append(os.path.join(before_badcase_root, 'Masks_P2_GT_FUSE', p[:-3]+'png'))
        self.realsamples_num = len(self.imgs)

        self.muticlass_path = muticlass_path
        self.muticlass_datasets = []
        self.extra_background_datasets = []
        self.add_other_class = add_other_class
        self.num_class = num_class
        self.add_edge = add_edge
        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            # RandomRotate_V2(45),
            # RandomTranslate_V2((40, 40)),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize),
            RandomGaussianBlur(p=0.5),
            RandomAddGaussianNoise(p=0.5)
            # RandomChangeBitDepth()
            ])
        self.trans_val = Compose([Sharpen()])

        self.trans_train_video = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            # RandomRotate_V2(45),
            # RandomTranslate_V2((40, 40)),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize),
            # RandomGaussianBlur(p=0.5),
            # RandomAddGaussianNoise(p=0.5)
            # RandomChangeBitDepth()
            ])
        
        self.hands_datasets_path_root = hands_datasets_path
        self.hands_datasets_images_path = osp.join(hands_datasets_path, 'Images/')
        self.hands_datasets = []
        if self.hands_datasets_path_root and self.mode == 'train':
            for i_hand in os.listdir(self.hands_datasets_images_path):
                # print(i_hand)
                # name = osp.split(i_hand)[1]
                f_name = i_hand.split(".")[0]
                if f_name:
                    i_path = osp.join(self.hands_datasets_images_path, f_name + '.jpg')
                    m_path = osp.join(self.hands_datasets_path_root, 'Masks/',f_name + '.png')
                    self.hands_datasets.append([i_path, m_path])

        if self.muticlass_path:
            for idx in range(2,81):
                path = os.path.join(self.muticlass_path,"Images/","{}/".format(idx))
                if os.path.exists(path) == False:
                    continue
                single_datasets = []
                if len(os.listdir(path))>0:
                    for i_name in os.listdir(path):
                        i_name = i_name.split(".")[0]
                        if i_name:
                            i_path = osp.join(self.muticlass_path,"Images/","{}/".format(idx), i_name + '.jpg')
                            m_path = osp.join(self.muticlass_path,"Masks/","{}/".format(idx), i_name + '.png')
                            # obj_image = Image.open(i_path).convert("RGB")
                            # obj_mask = Image.open(m_path).convert("L")
                            # single_datasets.append([obj_image, obj_mask])
                            single_datasets.append([i_path, m_path])
                    self.muticlass_datasets.append(single_datasets)
        # print(len(self.muticlass_datasets))
        if self.extra_background_path:
            for p in os.listdir(os.path.join(self.extra_background_path, 'Images')):
                self.extra_background_datasets.append([os.path.join(self.extra_background_path, 'Images', p), os.path.join(self.extra_background_path, 'Masks', p[:-3]+'png')])

    def __getitem__(self, idx):
        self.num += 1
        if self.mode == 'train' and idx >= self.normal_sample_num:
            idx = (idx - self.normal_sample_num) % (self.realsamples_num - self.normal_sample_num) + self.normal_sample_num
        impth = self.imgs[idx]
        mkpth = self.masks[idx]
        img = Image.open(impth)
        img = img.resize((self.imsize, self.imsize), Image.BILINEAR)
        label = Image.open(mkpth).convert('L')
        label = label.resize((self.imsize, self.imsize), Image.NEAREST)
        img = np.array(img)
        label = np.array(label)
        # label[label==255] = 0
        # print(np.unique(np.array(label)))
        if self.mode == 'train':
            if idx < self.celeb_num and random.random() > 0 and self.hands_datasets:
                hand_index = random.randint(0,len(self.hands_datasets)-1)
                hand_i_path, hand_m_path = self.hands_datasets[hand_index]
                hand_image = Image.open(hand_i_path).convert("RGB")
                hand_mask = Image.open(hand_m_path).convert("L")
                hand_im_lb = dict(im=hand_image, lb=hand_mask)
                hand_im_lb = Compose2([RandomRangeScale(self.imsize/512*0.8, self.imsize/512*1, self.imsize/512*1, self.imsize/512*1.1),Pad((self.imsize, self.imsize)),RandomTranslate((int(self.imsize/512*200),int(self.imsize/512*200)))])(hand_im_lb)
                # hand_im_lb = Compose2([RandomRangeScale(0.5, 0.8, 1, 1.8),RandomCrop_Padding((512, 512)),RandomRotate(180),RandomTranslate((128,128))])(hand_im_lb)

                hand_image, hand_mask = hand_im_lb['im'], hand_im_lb['lb']

                hand_image = np.array(hand_image)
                hand_mask = np.array(hand_mask)

                face_p = (label!=1)[:,:,np.newaxis].repeat(3,axis=2)
                hand_p = (hand_mask==0)[:,:,np.newaxis].repeat(3,axis=2)
                hand_image[hand_p] = 0
                face = copy.deepcopy(img)
                face[face_p] = 0
                face_p = face_p[:,:,0]
                hand_p = hand_p[:,:,0]
                hands = color_transformer.reinhard_color_transfer(hand_image,face,hand_p,face_p)
                hands = np.clip(hands , 0, 255)
               
                for c in range(3):
                    img[:,:,c] = np.where(hand_mask > 0, hands[:,:,c], img[:,:,c])

                label[hand_mask > 0] = 0

            if idx < self.celeb_num  and random.random() > 0.5 and self.add_other_class:
                class_index = random.randint(0,len(self.muticlass_datasets)-1)
                # print(len(self.muticlass_datasets))
                obj_index = random.randint(0,len(self.muticlass_datasets[class_index])-1)
                # print(len(self.muticlass_datasets[class_index]))
                obj_i_path, obj_m_path = self.muticlass_datasets[class_index][obj_index]
                # obj_image, obj_mask = self.muticlass_datasets[class_index][obj_index]
                obj_image = Image.open(obj_i_path).convert("RGB")
                obj_mask = Image.open(obj_m_path).convert("L")
                obj_im_lb = dict(im=obj_image, lb=obj_mask)
                obj_im_lb = Compose2([RandomRangeScale_v2(int(self.imsize/512*100), int(self.imsize/512*250)),Pad((self.imsize, self.imsize)),RandomRotate(180),RandomTranslate((int(self.imsize/512*160),int(self.imsize/512*160)))])(obj_im_lb)

                obj_image, obj_mask = obj_im_lb['im'], obj_im_lb['lb']
                obj_image = np.array(obj_image)
                obj_mask = np.array(obj_mask)
               
                for c in range(3):
                    img[:,:,c] = np.where(obj_mask > 0, obj_image[:,:,c], img[:,:,c])
                
                label[obj_mask > 0] = 0

            if idx < self.celeb_num  and random.random() < 0.1 and self.extra_background_path:
                bg_index = random.randint(0, len(self.extra_background_datasets) - 1)
                bg_i_path, bg_m_path = self.extra_background_datasets[bg_index]
                bg_image = Image.open(bg_i_path).convert("RGB")
                bg_mask = Image.open(bg_m_path).convert("L")
                bg_im_lb = dict(im=bg_image, lb=bg_mask)
                bg_im_lb = Compose2([RandomRangeScale_v2(int(self.imsize/512*160), int(self.imsize/512*350)),Pad((self.imsize, self.imsize)),RandomRotate(180),RandomTranslate((int(self.imsize/512*160),int(self.imsize/512*160)))])(bg_im_lb)

                bg_image, bg_mask = bg_im_lb['im'], bg_im_lb['lb']
                bg_image = np.array(bg_image)
                bg_mask = np.array(bg_mask)
               
                for c in range(3):
                    img[:,:,c] = np.where(bg_mask > 0, bg_image[:,:,c], img[:,:,c])
                
                label[bg_mask > 0] = 0

            img = Image.fromarray(img)
            label = Image.fromarray(label)
            im_lb = dict(im=img, lb=label)
            if idx < self.celeb_num:
                im_lb = self.trans_train(im_lb)
            else:
                im_lb = self.trans_train_video(im_lb)
            img, label = im_lb['im'], im_lb['lb']

            # if self.num<20:
            #     img.save("color{}.jpg".format(self.num))
            #     label.save("label{}.jpg".format(self.num))
            if self.add_edge:
                edge = np.array(label)
                edge = cv2.Laplacian(edge, cv2.CV_8U, ksize=3)
                edge[edge>0] = 1
                edge = Image.fromarray(edge)
                if self.num<20:
                    edge.save("edge{}.jpg".format(self.num))
                edge = np.array(edge).astype(np.int64)[np.newaxis, :]

        # im_edge = np.array(img)
        # im_edge = im_edge.astype(np.float32)
        # x = cv2.Sobel(im_edge, cv2.CV_32F, 1, 0, ksize=5)
        # y = cv2.Sobel(im_edge, cv2.CV_32F, 0, 1, ksize=5)
        # im_edge = np.concatenate((x*x, y*y), axis=2)
        # im_edge = np.sqrt(np.sum(im_edge, axis=2))
        # im_edge = (im_edge-im_edge.min())/(im_edge.max()-im_edge.min())
        # im_edge = torch.from_numpy(im_edge)
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        # label = convert_to_one_hot(label, self.num_class)
        if self.add_edge:return img, label, edge
        return img, label

    def __len__(self):
        if self.mode == 'train':
            # return len(self.imgs) + math.floor(self.ratio * len(self.imgs_synthetics) - 1)
            return len(self.imgs) + (self.ratio - 1) * (self.realsamples_num - self.normal_sample_num)
        else:
            return len(self.imgs)
        
        
if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    data_root = "/data4/hb/face_parsing/faceparsing_training_data/CelebAMask-HQ"
    data_root2 = "/data4/hb/face_parsing/faceparsing_training_data/douyin_pornpics_manual_anno"
    data_root3 = "/data4/hb/face_parsing/faceparsing_training_data/cvpr"
    synthetics_pth = ""
    hands_datasets_path = "/data4/hb/face_parsing/faceparsing_training_data/hands_datasets_v4"
    muticlass_path = "/data4/hb/face_parsing/faceparsing_training_data/coco/train2017"
    extra_background_path = "/data4/hb/face_parsing/faceparsing_training_data/FM-basecases_1000_packages/train_data/background"
    
    # 训练集
    dataset_lr_train = FaceMask(512, data_root, data_root2, data_root3, synthetics_pth, cropsize=448, mode='train', hands_datasets_path=hands_datasets_path, add_other_class=True, muticlass_path=muticlass_path, add_edge=False, extra_background_path=extra_background_path)
    print(len(dataset_lr_train))
    
    dataloader_train = DataLoader(
        dataset=dataset_lr_train,
        batch_size = 16,
        num_workers=8,
        pin_memory=True,
    )
    
    # for img, label in tqdm(dataloader_train):
    #     print(img.shape)
    #     print(label.shape)
    
    # # 验证集
    dataset_lr_valid = FaceMask(512, data_root, data_root2, data_root3, synthetics_pth, mode='val')
    print(len(dataset_lr_valid))
        