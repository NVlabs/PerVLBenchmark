# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import pandas as pd
import time
import os
import shutil

val_df_path = "../annotations/ytvos/cleaned_captions_ytvos_val.csv"
test_df_path = "../annotations/ytvos/cleaned_captions_ytvos_test.csv"
save_path = "../data/youtube_vos/retrival_sets"
eval_image_src_path = "../data/youtube_vos/ytvos_PerVL/fsl_seg/eval/retrieval"
folder_src = "../data/youtube_vos/ytvos_PerVL/fsl_seg/train/"

val_df = pd.read_csv(val_df_path)
test_df = pd.read_csv(test_df_path)

print("val_df",val_df.shape)
print("test_df",test_df.shape)

joint_caption_df = pd.concat([val_df, test_df], ignore_index=True)
print("joint_caption_df",joint_caption_df.shape)

joint_caption_df.to_csv("joint_captions.csv")

dfs = [val_df,test_df]
folder_name = ["validation","test"]

for i,df in enumerate(dfs):
    args_path = os.path.join(save_path, "%s" % (folder_name[i]), "codes_infer")
    args_eval_path = os.path.join(save_path, "%s" % (folder_name[i]), "eval","test")
    os.makedirs(args_path, exist_ok=True)
    os.makedirs(args_eval_path, exist_ok=True)
    classes = df["image_name"].to_numpy()
    for class_name in classes:
        image_name_str = class_name.split("/")[-1]
        class_name_str = "_".join(image_name_str.split("_")[:-1])
        dst_path = os.path.join(args_path, class_name_str)
        shutil.copytree(os.path.join(folder_src, class_name_str), dst_path)
        shutil.copy(os.path.join(eval_image_src_path, image_name_str), os.path.join(args_eval_path,class_name_str)+".jpg")



