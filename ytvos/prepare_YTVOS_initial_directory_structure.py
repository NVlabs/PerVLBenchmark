# Copyright (C) 2022 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file 
# located at the root directory.

import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import ruamel.yaml as ruamel_yaml
import yaml
import cv2 # pip3 install opencv-python
from dataclasses import dataclass
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm
#%%
from PIL import Image


@dataclass
class Args:
    output_dir: str = '../data/youtube_vos/ytvos_PerVL'
    """ Path to save the output directory structure """

    ytvos_official_dir: str = '../data/youtube_vos'
    """ Path of the official YTVOS data """

    pervl_meta_yaml: str = '../annotations/ytvos/ytvos_personalization_meta.yaml'
    """ full path filename to the yaml that contains the personalization metadata """




parser = ArgumentParser()
parser.add_arguments(Args, dest="cfg")
args = parser.parse_args().cfg

#%%
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

#%%


basedir = Path(args.ytvos_official_dir).expanduser()
with (basedir / 'train/meta.json').open('r') as f:
    meta = json.load(f)['videos']

#%%
def load_ordered_yaml(yaml_fname):
    # using ruamel to load a YAML while keeping the order and visual structure of the file
    with open(yaml_fname) as f:
        experiment_yaml = ruamel_yaml.load(f, Loader=ruamel_yaml.RoundTripLoader)
    return experiment_yaml

def load_image_to_np(fname):
    # return plt.imread(fname)
    im_frame = Image.open(fname)
    return np.array(im_frame)
#%%
def mask_to_bounding_box(mask):
    bbox = [None, None, None, None]
    bbox[1] = np.nonzero(mask)[0].min()
    bbox[0] = np.nonzero(mask)[1].min()
    bbox[3] = np.nonzero(mask)[0].max()
    bbox[2] = np.nonzero(mask)[1].max()
    return bbox

def write_ytvos_train_frames(output_dir, bbox, frame, img, name):
    # Write an image that crops the relevant object from the train frame.
    fname = str(output_dir / 'fsl_seg/train' / f'{name}/{name}_{frame}.jpg')
    Path(fname).parent.mkdir(exist_ok=True, parents=True)
    boxed_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    cv2.imwrite(fname, cv2.cvtColor(boxed_img, cv2.COLOR_RGB2BGR))


def write_ytvos_test_frames(output_dir, bbox, frame, gt_box_labels, mask, name, scene_id):

    # Write images for segmentation experiment. I.e. the full image, and the gt_mask of the target object.
    gt_box_labels[name] = tuple(bbox)
    img = load_image_to_np(basedir / 'train/JPEGImages' / scene_id / f'{frame}.jpg')
    fname = str(output_dir / 'fsl_seg/eval/images' / f'{name}_{frame}.jpg')
    Path(fname).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    fname = str(output_dir / 'fsl_seg/eval/gt_mask' / f'{name}_{frame}.png')
    Path(fname).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(fname, mask)

    # Write images for retrieval experiment. I.e. cropping only the relevant object from the test frame.
    fname = str(output_dir / 'fsl_seg/eval/retrieval' / f'{name}_{frame}.jpg')
    Path(fname).parent.mkdir(exist_ok=True, parents=True)
    dy = bbox[3] - bbox[1]
    dx = bbox[2] - bbox[0]
    r_coeff = 0.5
    retrieval_box = np.array((bbox[0] - r_coeff * dx, bbox[1] - r_coeff * dy, bbox[2] + r_coeff * dx,
                              bbox[3] + r_coeff * dy))
    retrieval_box = np.max((retrieval_box, (0, 0, 0, 0)), axis=0)
    retrieval_box[3] = min(retrieval_box[3], img.shape[0])
    retrieval_box[2] = min(retrieval_box[2], img.shape[1])
    retrieval_box = retrieval_box.astype(int)
    # print(bbox, retrieval_box)
    boxed_img = img[retrieval_box[1]:retrieval_box[3], retrieval_box[0]:retrieval_box[2], :]
    cv2.imwrite(fname, cv2.cvtColor(boxed_img, cv2.COLOR_RGB2BGR))

    # Write images_with_bbox just for illustration/annotation purpose
    bbox_color = (255, 255, 0)
    bbox_width = 2
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, bbox_width)
    fname = str(output_dir / 'fsl_seg/eval/images_with_bbox' / f'{name}_{frame}.jpg')
    Path(fname).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#%%

def load_frame(frame, concept, image_cache, seg_cache):
    scene_id = concept['scene_id']
    img = image_cache.setdefault((scene_id, frame), load_image_to_np(basedir / 'train/JPEGImages' / scene_id / f'{frame}.jpg'))
    seg = seg_cache.setdefault((scene_id, frame), load_image_to_np(basedir / 'train/Annotations' / scene_id / f'{frame}.png'))
    o_id = int(concept['instance_id'])
    mask = (seg == o_id).astype(np.uint8)
    if mask.sum() == 0:
        assert False, f'{scene_id}: missing test mask for {o_id} (frame={frame})'
    name = f"{concept['coarse_grained_class']}_{scene_id}_{o_id}"
    bbox = mask_to_bounding_box(mask)
    return bbox, img, mask, name

#%%

concepts_meta = load_ordered_yaml(args.pervl_meta_yaml)

#%%
image_cache, seg_cache = {}, {}
gt_box_labels = defaultdict(dict)

for concept in tqdm(list(concepts_meta.values())):
    for frame in concept['train_frames'] + [concept['test_frame']]:
        bbox, img, mask, name = load_frame(frame, concept, image_cache, seg_cache)
        if frame == concept["test_frame"]:
            write_ytvos_test_frames(output_dir, bbox, frame, gt_box_labels, mask, name, concept['scene_id'])
        else:
            write_ytvos_train_frames(output_dir, bbox, frame, img, name)

#%%
fname = str(output_dir / 'fsl_seg/eval/gt_box_labels.yaml')
with open(fname, 'w') as fp:
    yaml.dump(gt_box_labels, fp)

#%%
