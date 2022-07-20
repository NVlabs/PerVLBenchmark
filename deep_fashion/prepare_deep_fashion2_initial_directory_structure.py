# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

import os
import shutil
import pandas as pd


def prepare_files(set_files_path, out_path):

    with open(set_files_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    for line in lines:
        if line.startswith('codes_infer'):
            class_name = line.split("/")[1][:-1]
            class_path = os.path.join(out_path, class_name)
            os.makedirs(class_path, exist_ok=True)
        elif line.endswith('.jpg'):
            shutil.copy(os.path.join("../data/deep_fashion2/train/image", line),  os.path.join(class_path, line))

def prepare_eval_sets(caption_path, out_path):

    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(caption_path)
    eval_file_list = df["image_name"].to_numpy()

    for file in eval_file_list:
        file_path = os.path.join("../data/deep_fashion2", file)
        shutil.copy(file_path, os.path.join(out_path, file_path.split("/")[-1]))

prepare_files("validation_2022_02_07-16_42_59_fsl_train.txt", "../data/deep_fashion2/personalized_validation/codes_infer")
prepare_files("test_2022_02_07-16_42_59_fsl_train.txt", "../data/deep_fashion2/personalized_test/codes_infer")

prepare_eval_sets("captions_validation_2022_02_07-16_42_59.csv", "../data/deep_fashion2/personalized_validation/eval/test")
prepare_eval_sets("captions_test_2022_02_07-16_42_59.csv", "../data/deep_fashion2/personalized_test/eval/test")