# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.
import os

import numpy as np
from collections import Counter
import time

N = 1000 # number of objects

file1 = open('../data/inversion_model_train_data/mscoco-train2014-svo-2016-02-18.txt', 'r')
obj_names_save_path = "../data/inversion_model_train_data/txt_for_training/"

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

print("most_commom_obj",len(most_commom_obj))
obj_names_only = [obj[0] for obj in most_commom_obj]

os.makedirs(obj_names_save_path, exist_ok=True)
np.save(os.path.join(obj_names_save_path,"obj_names_only.npy"), obj_names_only)

commom_obj_dict = dict()

for obj_entry in most_commom_obj:
    obj_name = obj_entry[0]
    inds = np.where(np.asarray(joint_list_obj)==obj_name)[0]

    joint_list_sentences_asterix = ['']*len(joint_list_sentences[inds])
    for i,joint_list_sentence in enumerate(joint_list_sentences[inds]):
        joint_list_sentences_asterix[i] = joint_list_sentences[inds][i].replace(obj_name, "*", 1)
    joint_list_sentences_asterix = np.asarray(joint_list_sentences_asterix)
    commom_obj_dict[obj_name] = [joint_list_img_ids[inds],joint_list_sentences[inds],joint_list_sentences_asterix]



np.save(os.path.join(obj_names_save_path,"commom_obj_dict.npy"), commom_obj_dict)
