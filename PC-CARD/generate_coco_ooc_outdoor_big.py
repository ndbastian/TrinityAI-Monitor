import cv2
# import ipdb
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle
import sys
from data.coco_loader import COCOLoader


def remove_bg(original, poly):
    poly = np.array(poly, dtype=np.int32)
    poly_e = np.expand_dims(poly, 0)
    # creat the mask
    mask = np.zeros(original.shape, dtype=np.uint8)
    cv2.fillPoly(mask, poly_e, (255, 255, 255))
    # and with the original
    final = cv2.bitwise_and(mask, original)
    return final, mask


def overlay_patch(bck_img, patch_img, patch_mask, location_rc):
    overlay_img = []
    dim_error_flag = False

    patch_mask = np.expand_dims(patch_mask, axis=2)
    patch_mask_3d = np.concatenate((patch_mask, patch_mask, patch_mask), axis=2)
    masked_img = np.multiply(patch_img, patch_mask_3d)

    r_end = location_rc[0] + patch_img.shape[0]
    c_end = location_rc[1] + patch_img.shape[1]

    if (r_end > bck_img.shape[0]) or (c_end > bck_img.shape[1]):
        dim_error_flag = True
        return overlay_img, dim_error_flag

    else:
        zero_img = np.zeros_like(bck_img)
        zero_img[location_rc[0]: location_rc[0] + patch_img.shape[0],
        location_rc[1]: location_rc[1] + patch_img.shape[1], :] = masked_img

        one_img = np.ones_like(bck_img)
        one_img[location_rc[0]: location_rc[0] + patch_img.shape[0],
        location_rc[1]: location_rc[1] + patch_img.shape[1], :] = (1 - patch_mask_3d)

        overlay_img = np.multiply(bck_img, one_img) + zero_img

    return overlay_img, dim_error_flag


coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
                67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
                38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
                81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
                42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
                80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
                7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
                46, 'zebra': 24}

# outdoor objects
sport_ids = {'tennis racket': 43, 'surfboard': 42, 'skateboard': 41, 'baseball glove': 40,
             'baseball bat': 39, 'kite': 38, 'sports ball': 37, 'snowboard': 36, 'skis': 35, 'frisbee': 34, }
animal_ids = {'giraffe': 25, 'zebra': 24, 'bear': 23, 'elephant': 22, 'cow': 21, 'sheep': 20,
              'horse': 19, 'dog': 18, 'cat': 17, 'bird': 16, }
outdoor_accessory_ids = {'suitcase': 33, 'tie': 32, 'handbag': 31, 'umbrella': 28, 'backpack': 27, }
outdoor_object_ids = {'bench': 15, 'parking meter': 14, 'stop sign': 13, 'fire hydrant': 11, 'traffic light': 10, }
vehicle_ids = {'boat': 9, 'truck': 8, 'train': 7, 'bus': 6, 'airplane': 5, 'motorcycle': 4, 'car': 3, 'bicycle': 2, }
person_ids = {'person': 1, }

# indoor objects
indoor_object_ids = {'toothbrush': 90, 'hair drier': 89, 'teddy bear': 88, 'scissors': 87, 'vase': 86, 'clock': 85,
                     'book': 84, }
appliance_ids = {'refrigerator': 82, 'sink': 81, 'toaster': 80, 'oven': 79, 'microwave': 78, }
electronic_ids = {'cell phone': 77, 'keyboard': 76, 'remote': 75, 'mouse': 74, 'laptop': 73, 'tv': 72, }
furniture_ids = {'toilet': 70, 'dining table': 67, 'bed': 65, 'potted plant': 64, 'couch': 63, 'chair': 62, }
food_ids = {'cake': 61, 'donut': 60, 'pizza': 59, 'hot dog': 58, 'carrot': 57, 'broccoli': 56,
            'orange': 55, 'sandwich': 54, 'apple': 53, 'banana': 52, }
kitchen_ids = {'bowl': 51, 'spoon': 50, 'knife': 49, 'fork': 48, 'cup': 47, 'wine glass': 46, 'bottle': 44, }

outdoor_ids = {**sport_ids, **outdoor_accessory_ids, **animal_ids, **outdoor_object_ids, **vehicle_ids, **person_ids}
indoor_ids = {**indoor_object_ids, **appliance_ids, **electronic_ids, **furniture_ids, **food_ids, **kitchen_ids}

outdoor_obj_name_list = list(outdoor_ids.keys())
indoor_obj_name_list = list(indoor_ids.keys())

outdoor_obj_id_list = list(outdoor_ids.values())
indoor_obj_id_list = list(indoor_ids.values())



# ------------------ main ----------------------------------------
# --------------------- Dataset --------------------------------

# Set up paths

DATASETS_ROOT = '/workspace/aroy/datasets'
root_path_prefix = '/workspace/aroy/datasets/coco/%s'
dataset_dirpath    = "/workspace/aroy/datasets/coco_ooc/OOC_images/outdoor_big/images"
annotation_dirpath = "/workspace/aroy/datasets/coco_ooc/OOC_images/outdoor_big/annotations"


split = "val2014"
root = root_path_prefix % (split)
annFile = '%s/coco/annotations/instances_%s.json' % (DATASETS_ROOT, split)
dataset = COCOLoader(root, annFile, included=[*range(1, 81)])

# ------------ scene and object selection ------------------------------
scene_obj_ids = outdoor_ids

sel_sport_ids = {'tennis racket': 43, 'baseball glove': 40, 
             'baseball bat': 39, 'sports ball': 37, 'snowboard': 36, 'skis': 35, 'frisbee': 34,}

sel_outdoor_accessory_ids = {'suitcase': 33, 'handbag': 31, 'umbrella': 28, 'backpack': 27, }


patch_obj_ids = {**sel_sport_ids, **sel_outdoor_accessory_ids}


# --------------------- Data generation --------------------------------
np.random.seed(1)

os.makedirs(dataset_dirpath, exist_ok=True)
os.makedirs(annotation_dirpath, exist_ok=True)

# hyper parameters
# patch size range
patch_width_range = (200, 240)

# location
location_r_range = (20, 60)
location_c_range = (20, 60)

min_obj_size = 150
min_obj_num = 6

keep_bg = False
count_i = 0

i_indx_max = 1000
j_indx_max = 2000
ooc_img_count = 0
for i, batch in enumerate(dataset):

    if i > i_indx_max:
        break

    _, data = batch
    image_id = int(data['image_id'].item())
    if not image_id in dataset.coco.imgs:
        continue
    fname = dataset.coco.loadImgs(image_id)[0]['file_name']
    fpath = os.path.join(dataset.root, fname)

    # number of objects
    ann_ids = dataset.coco.getAnnIds(imgIds=image_id, iscrowd=False)

    if len(ann_ids) < min_obj_num:  # filter images with less objects
        continue

    # check the object to determine whether to select the image
    i_select_flag = False
    # for each object
    for single_id in ann_ids:
        target = dataset.coco.loadAnns(single_id)
        ann = target[0]  # load only on annoation
        category_id = target[0]['category_id']

        if category_id in scene_obj_ids.values():  # indoor image
            i_select_flag = True
            category_name = [key for key in coco_ids.keys() if coco_ids[key] == category_id]
            break

    # -----------------if the image is selected-----------------------------
    if i_select_flag:

        # find another image object to overlay
        count_j = 0
        for j, batch_j in enumerate(dataset):

            if i == j:
                continue

            if j > j_indx_max:
                break

            _, data_j = batch_j
            image_id_j = int(data_j['image_id'].item())
            if not image_id_j in dataset.coco.imgs:
                continue
            fname_j = dataset.coco.loadImgs(image_id_j)[0]['file_name']
            fpath_j = os.path.join(dataset.root, fname_j)

            # number of objects
            ann_ids_j = dataset.coco.getAnnIds(imgIds=image_id_j, iscrowd=False)

            # for each object
            for ann_j_indx, single_id_j in enumerate(ann_ids_j):

                target_j = dataset.coco.loadAnns(single_id_j)
                ann_j = target_j[0]  # load only on annoation
                category_id_j = target_j[0]['category_id']
                category_name_j = [key for key in coco_ids.keys() if coco_ids[key] == category_id_j]

                # ------------- select the patch object ----------------
                if category_id_j in list(patch_obj_ids.values()):  # outdoor object
                    sel_fname_j = fname_j
                    sel_fpath_j = fpath_j

                    pil_j = Image.open(fpath_j).convert("RGB")
                    original_j = np.array(pil_j)

                    # get the selected crop and mask 
                    polys = []
                    if ann_j['iscrowd'] == 0:
                        for seg in ann_j['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polys.append(poly)
                    # if one objects has multiple polygone just ignore it to make things simple
                    if len(polys) > 1:
                        continue

                    xmin, ymin, w, h = ann_j['bbox']
                    xmax = int(xmin + w)
                    ymax = int(ymin + h)
                    xmin = int(xmin)
                    ymin = int(ymin)

                    bg_removed = np.zeros_like(original_j)

                    for p in polys:
                        p = np.array(p, dtype=np.int32)
                        bg_removed, mask = remove_bg(original_j, p)  # add each part back

                    crop_img = bg_removed[ymin:ymax, xmin:xmax, :]

                    if np.min([crop_img.shape[0], crop_img.shape[1]]) < min_obj_size:
                        continue

                    crop_mask = mask[ymin:ymax, xmin:xmax, 0]
                    crop_mask[crop_mask == 255] = 1

                    count_j = count_j + 1

                    # overlay

                    # manage the background img
                    pil = Image.open(fpath).convert("RGB")
                    original = np.array(pil)

                    # choose size of the patch
                    crop_width = int(np.random.uniform(patch_width_range[0], patch_width_range[1]))
                    crop_height = int(crop_img.shape[0] * (crop_width / crop_img.shape[1]))
                    crop_img_rz = cv2.resize(crop_img, dsize=(crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
                    crop_mask_rz = cv2.resize(crop_mask, dsize=(crop_width, crop_height),
                                              interpolation=cv2.INTER_NEAREST)

                    # choose the location to put the patch
                    location_r = int(np.random.uniform(location_r_range[0], location_r_range[1]))
                    location_c = int(np.random.uniform(location_c_range[0], location_c_range[1]))
                    location_rc = (location_r, location_c)

                    overlay_img, dim_error_flag = overlay_patch(original, crop_img_rz, crop_mask_rz, location_rc, )
                    if not dim_error_flag:
                        ooc_img_count = ooc_img_count + 1
                        print(i, j, ooc_img_count)
                        ori_img_name = os.path.splitext(fname)[0]
                        ooc_filename = f"{ori_img_name}_outdoor_big_var_{j}_{ann_j_indx}.jpg"
                        plt.imsave(os.path.join(dataset_dirpath, ooc_filename), overlay_img)

                        annotation = {}
                        annotation['original_ann_ids'] = ann_ids
                        annotation['image_id'] = image_id
                        bbox = [location_r, location_c, crop_width, crop_height] #xmin, ymin, w, h = bbox
                        annotation['ooc_annotation'] = {'image_id': image_id_j, 'coco_ann_id': single_id_j,
                                                        'bbox': bbox,
                                                        }

                        annotation_filename = ooc_filename.replace(".jpg", ".npy")
                        annotation_filepath = os.path.join(annotation_dirpath, annotation_filename)
                        np.save(annotation_filepath, annotation)
                        # print(f"{ooc_img_count}: {annotation_filepath}")