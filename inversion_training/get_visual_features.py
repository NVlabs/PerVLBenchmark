# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import numpy as np
from collections import Counter
import time
import os
import torch
import clip
from PIL import Image
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

data_augment_preprocess =  Compose([
        Resize(300, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


N = 1000 #1000 # number of objects
ENC_DIM = 512

coco_path = "../data/inversion_model_train_data/train2014"

file1 = open('../data/inversion_model_train_data/mscoco-train2014-svo-2016-02-18.txt', 'r')
lines = file1.readlines()

size = (len(lines))
img_ids = ['']*size
obj1 = ['']*size
obj2 = ['']*size
sentences = ['']*size

for i, line in enumerate(lines):
    line_split = (line.split("|"))
    img_ids[i] = line_split[0].split("=")[1]
    obj1[i] = line_split[1].split(",")[0].strip()
    obj2[i] = line_split[1].split(",")[2].strip()
    sentences[i] = line_split[3].strip()

joint_list_obj = obj1 + obj2
joint_list_img_ids = np.asarray(img_ids + img_ids)
joint_list_sentences = np.asarray(sentences + sentences)
most_commom_obj = Counter(joint_list_obj).most_common(N)

commom_obj_dict = dict()

all_image_inds = []

print("most_commom_obj",len(most_commom_obj))
#most_commom_obj = most_commom_obj[::100]
print("most_commom_obj",len(most_commom_obj))


for obj_entry in most_commom_obj:
    obj_name = obj_entry[0]
    inds = np.where(np.asarray(joint_list_obj)==obj_name)[0]
    all_image_inds = np.unique(np.asarray( list(all_image_inds) + list(joint_list_img_ids[inds])))
    print("all_image_inds",all_image_inds.shape)

file_names = ['']*len(all_image_inds)
for i, image_ind in enumerate(all_image_inds):
    file_names[i] = os.path.join(coco_path, "COCO_train2014_%s.jpg"%(str(image_ind).zfill(13)[:-1]))



### Get embedding
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("preprocess",preprocess)

model.eval()

print("file_names[0]",len(file_names))
#file_names = file_names[::1000]
print("file_names[0]",len(file_names))

image_features_all = torch.zeros((len(file_names), ENC_DIM))


visual_features_dict = dict()




for i, file_name in enumerate(file_names):
    with Image.open(file_name) as im:
        image = preprocess(im).unsqueeze(0).to(device)
        image = data_augment_preprocess(im).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features_all[i] = image_features

        print("image_ind[i]",all_image_inds[i])

        visual_features_dict[str(all_image_inds[i])] = [image_features.cpu().numpy()]

print("visual_features_dict",visual_features_dict.keys())

os.makedirs("../data/inversion_model_train_data/visual_features", exist_ok=True)
np.savez("../data/inversion_model_train_data/visual_features/visual_features_dict_center_crop_300_224", visual_features_dict)
