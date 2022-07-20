# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.
import os

import numpy as np
from collections import Counter
import time
import torch
import clip
import faiss

natural_prompt_default: str = "This is a photo of a "
batch_size = 128
emb_dim = 512

obj_names = np.load("../data/inversion_model_train_data/txt_for_training/obj_names_only.npy")


#https://storage.googleapis.com/openimages/web/factsfigures.html
#Classes are identified by MIDs (Machine-generated Ids) as can be found in Freebase or Google Knowledge Graph API. A short description of each class is available in class-descriptions.csv.

open_images_object_list_path = "../data/inversion_model_train_data/oidv6-class-descriptions.csv"
file1 = open(open_images_object_list_path, 'r')
Lines = file1.readlines()
Lines = [line.split(",")[1:] for line in Lines]
aug_obj_names = [' '.join(line).strip() for line in Lines]

def clip_text_get_embeddings(obj_names):
    L = len(obj_names)

    out_features = torch.zeros((L, emb_dim))

    ind = 0

    for i in range(int(np.ceil(L/batch_size))):
        batch_data = obj_names[ind:ind + batch_size]
        batch_text = [natural_prompt_default + obj for obj in batch_data]

        batch_tokens = clip.tokenize(batch_text).to(device)
        with torch.no_grad():
            batch_feat = model.encode_text(batch_tokens)

        out_features[ind:ind + batch_size] = batch_feat

        ind = ind + batch_size

    return out_features





####### Find for each line from open images the most similar object:
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

obj_features = clip_text_get_embeddings(obj_names).cpu().numpy()
aug_obj_features = clip_text_get_embeddings(aug_obj_names).cpu().numpy()

print("obj_features",obj_features.shape)
print("aug_obj_features",aug_obj_features.shape)

####### Find nearest original object for each open object:

index = faiss.IndexFlatL2(obj_features.shape[1])
index.add(np.ascontiguousarray(obj_features.astype('float32')))

D, I = index.search(np.ascontiguousarray(aug_obj_features.astype('float32')), 1)

out_path = "../data/inversion_model_train_data/open_images"
os.makedirs(out_path,exist_ok=True)
np.savez(os.path.join(out_path,"open_images_obj_names.npz"), aug_obj_names)
np.savez(os.path.join(out_path,"open_images_to_mscoco_map.npz"), I)
