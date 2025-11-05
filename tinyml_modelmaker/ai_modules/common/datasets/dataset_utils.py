#################################################################################
# Copyright (c) 2023-2024, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#################################################################################

import base64
import collections
import copy
import glob
import json
import os
import random
import re
from itertools import accumulate
from random import seed, shuffle

# import PIL
# from PIL import ImageOps
from xml.etree.ElementTree import Element as ET_Element

import numpy as np

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
#
from typing import Any, Dict

from .... import utils


def create_filelist(input_data_path: str, output_dir: str, ignore_str_list=None) -> str:
    '''
    :param input_data_path: Just does a walkthrough of the dataset directory and creates a filelist
    :param ignore_str: RegEx String expression to ignore while globbing for files
    :return: file_list.txt which will later be used for creating validation_list.txt and teaining_list.txt
    '''
    filelist = []
    for root, dirs, files in os.walk(input_data_path):
        relative_root = os.path.relpath(root, input_data_path)
        for filename in files:
            tmp_name = os.path.join(relative_root, filename) if relative_root != "." else filename
            if not any(re.search(ignore_str, tmp_name) for ignore_str in ignore_str_list):
                filelist.append(tmp_name)

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'file_list.txt')
    with open(out_file, 'w') as fl_fh:
        fl_fh.write('\n'.join(filelist))

    return out_file


def create_inter_file_split(file_list: str, split_list_files: tuple, split_factor: float or list, shuffle_items=True, random_seed=42):
    '''
    Creates a simple split according to split factor for each of the classes.
    This function splits the list of files into train, val, test splits based on the split factor.

    :param file_list: file_list.txt file that contains all the links to the files for dataset
    :param split_list_files: training_list.txt and validation_list.txt and so on...
    :param split_factor: can be a float number or a list of splits e.g [0.2, 0.3]
    :return: out_files: List containing the paths of files that contain the dataset of the corresponding splits
    '''
    assert isinstance(split_list_files, (list, tuple)), "split_list_files should be passed as a tuple or list"
    number_of_splits = len(split_list_files)
    split_factors = []
    if type(split_factor) == float:
        assert split_factor < 1.0, "split_factor should be less than 1"
        # The default split factor is the fraction for training set.
        split_factors.append(split_factor)
        # The remainder of the set will be equally split between val or val/test
        remainder = 1 - split_factor

    elif isinstance(split_factor, (list, tuple)):
        assert sum(split_factor) <= 1, "The Sum of split factors should be <=1"
        assert len(split_factor) <= len(split_list_files), "The number of elements in split factors should be less than/equal to number of split names"
        split_factors.extend(split_factor)
        remainder = 1 - sum(split_factor)

    if number_of_splits > len(split_factor):
        remainder_fraction = remainder / (number_of_splits - len(split_factor))
        [split_factors.append(remainder_fraction) for _ in range(number_of_splits - len(split_factor))]
    assert len(split_factor) == len(split_list_files), f"Number of split files: {len(split_list_files)} should be same as length of split factors: {len(split_factor)}"

    with open(file_list) as fp:
        list_of_files = [x.strip() for x in fp.readlines()]  # Contains the list of files
    files_of_a_class = collections.defaultdict(list)  # Empty dict with values as list
    # Eg: file_path: class3_bearingFlaking/vibData_1.csv
    [files_of_a_class[os.path.dirname(file_path)].append(file_path) for file_path in list_of_files]

    if shuffle_items:
        seed(random_seed)
        shuffle(list_of_files)

    out_splits = collections.defaultdict(list)
    for class_name, class_files in files_of_a_class.items():
        # Normalise to split based on integer indices
        split_lengths = [int(split_factor * len(class_files)) for split_factor in split_factors]
        # To add the unused file to test data
        split_lengths[-1] = len(class_files) - sum(split_lengths[:-1])  # adjust the last split so that no element is left behind
        # To add the unused file to train data
        # split_lengths[0] = len(class_files) - sum(split_lengths[1:])  # adjust the last split so that no element is left behind
        split_file_list_for_class = [class_files[x - y: x] for x, y in zip(accumulate(split_lengths), split_lengths)]
        [out_splits[split_file].extend(split_file_list_for_class[i]) for i, split_file in enumerate(split_list_files)]

    for split_file, splits in out_splits.items():
        with open(split_file, 'w') as of_fh:
            of_fh.write('\n'.join(splits))

def create_intra_file_split(file_list: str, split_list_files: tuple, split_factor: float or list, data_dir, out_dir, split_names, shuffle_items=True, random_seed=42):
    '''
    Creates a simple split according to split factor for each of the classes. This utility splits each file as per the split factor
    Example: If the file has 100 lines --> 50 train, 30 val, 20 test

    :param file_list: file_list.txt file that contains all the links to the files for dataset
    :param split_list_files: training_list.txt and validation_list.txt and so on...
    :param split_factor: can be a float number or a list of splits e.g [0.2, 0.3]
    '''
    assert isinstance(split_list_files, (list, tuple)), "split_list_files should be passed as a tuple or list"
    number_of_splits = len(split_list_files)
    split_factors = []
    if type(split_factor) == float:
        assert split_factor < 1.0, "split_factor should be less than 1"
        # The default split factor is the fraction for training set.
        split_factors.append(split_factor)
        # The remainder of the set will be equally split between val or val/test
        remainder = 1 - split_factor

    elif isinstance(split_factor, (list, tuple)):
        assert sum(split_factor) <= 1, "The Sum of split factors should be <=1"
        assert len(split_factor) <= len(split_list_files), "The number of elements in split factors should be less than/equal to number of split names"
        split_factors.extend(split_factor)
        remainder = 1 - sum(split_factor)

    if number_of_splits > len(split_factor):
        remainder_fraction = remainder / (number_of_splits - len(split_factor))
        [split_factors.append(remainder_fraction) for _ in range(number_of_splits - len(split_factor))]
    assert len(split_factor) == len(split_list_files), f"Number of split files: {len(split_list_files)} should be same as length of split factors: {len(split_factor)}"

    with open(file_list) as fp:
        # list_of_files = [os.path.join(os.path.dirname(os.path.dirname(file_list)), data_dir, x.strip()) for x in fp.readlines()]  # Contains the list of files
        list_of_files = [x.strip() for x in fp.readlines()]  # Contains the list of files
    files_of_a_class = collections.defaultdict(list)  # Empty dict with values as list
    # Eg: file_path: class3_bearingFlaking/vibData_1.csv
    [files_of_a_class[os.path.dirname(file_path)].append(file_path) for file_path in list_of_files]

    if shuffle_items:
        seed(random_seed)
        shuffle(list_of_files)

    out_splits = collections.defaultdict(list)
    first_row_is_a_header = None
    for class_name, class_files in files_of_a_class.items():
        for class_file in class_files:
            with open(os.path.join(os.path.dirname(out_dir), data_dir, class_file)) as cfp:
                rows_in_file = cfp.readlines()  # Read the file as a list
            if re.search(r"[a-zA-Z]", rows_in_file[0]):
                first_row_is_a_header = rows_in_file[0]
                rows_in_file = rows_in_file[1:]

            # Normalise to split based on integer indices
            split_lengths = [int(split_factor * len(rows_in_file)) for split_factor in split_factors]
            # To add the unused file to test data
            split_lengths[-1] = len(rows_in_file) - sum(split_lengths[:-1])  # adjust the last split so that no element is left behind
            # To add the unused file to train data
            # split_lengths[0] = len(rows_in_file) - sum(split_lengths[1:])  # adjust the last split so that no element is left behind
            if first_row_is_a_header:
                file_split_by_rows = [[first_row_is_a_header] + rows_in_file[x - y: x] for x, y in zip(accumulate(split_lengths), split_lengths)]
            else:
                file_split_by_rows = [rows_in_file[x - y: x] for x, y in zip(accumulate(split_lengths), split_lengths)]
            # file_split_by_rows: Contains train, val, test split for each file
            out_file_paths = [f'{os.path.join(class_name, os.path.splitext(os.path.basename(class_file))[0])}_{split_name}{os.path.splitext(os.path.basename(class_file))[1]}' for split_name in split_names]
            for out_file_path, file_split in zip(out_file_paths, file_split_by_rows):
                os.makedirs(os.path.dirname(os.path.join(out_dir, out_file_path)), exist_ok=True)
                with open(os.path.join(out_dir, out_file_path), 'w') as ofp:
                    ofp.write(''.join(file_split))

            for i, split_file in enumerate(split_list_files):
                out_splits[split_file].append(out_file_paths[i])
            # [out_splits[split_file].extend(out_file_paths[i]) for i, split_file in enumerate(split_list_files)]

    for split_file, splits in out_splits.items():
        with open(split_file, 'w') as of_fh:
            of_fh.write('\n'.join(splits))


def parse_voc_xml_file(annotation_file_name: str) -> Dict[str, Any]:
    node = ET_parse(annotation_file_name).getroot()
    return parse_voc_xml(node)


def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def get_new_id(id_list):
    found_ids_max = max(id_list) if id_list else 0
    found_gaps = [id for id in range(1, found_ids_max+2) if id not in id_list]
    return found_gaps[0]


def get_category_names(categories_list):
    category_names = [d['name'] for d in categories_list]
    return category_names


def get_category_ids(categories_list):
    category_ids = [d['id'] for d in categories_list]
    return category_ids


def get_category_entry(categories_list, category_name):
    category_entries = [d for d in categories_list if d['name'] == category_name]
    category_entry = category_entries[0] if len(category_entries) > 0 else None
    return category_entry


def get_category_id(categories_list, category_name):
    category_ids = get_category_ids(categories_list)
    category_ids = [c for c in category_ids if c == category_name]
    category_id = category_ids[0] if len(category_ids) > 0 else None
    return category_id


def get_category_name(categories_list, category_id):
    category_names = get_category_names(categories_list)
    category_names = [c for c in category_names if c == category_id]
    category_name = category_names[0] if len(category_names) > 0 else None
    return category_name


def get_new_category_id(categories_list):
    category_ids = get_category_ids(categories_list)
    category_id = get_new_id(category_ids)
    return category_id


def add_missing_categories(categories_list, missing_category_name='undefined'):
    if len(categories_list) == 0:
        return categories_list
    #
    category_ids = [d['id'] for d in categories_list]
    category_ids_max = max(category_ids)
    category_ids_missing = [id for id in range(1,category_ids_max+1) if id not in category_ids]
    categories_list_out = copy.deepcopy(categories_list)
    for category_id in category_ids_missing:
        name = f'{missing_category_name}{category_id}'
        category_entry = dict(id=category_id, supercategory=name, name=name)
        categories_list_out.append(category_entry)
    #
    # sort
    categories_list_out = sorted(categories_list_out, key=lambda d:d['id'])
    return categories_list_out


def adjust_categories(categories_list, category_names_new, missing_category_name='undefined'):
    categories_list_out = []
    for category_name in category_names_new:
        category_entry = get_category_entry(categories_list, category_name)
        if category_entry is None:
            new_category_id = get_new_category_id(categories_list)
            category_entry = dict(id=new_category_id, supercategory=category_name, name=category_name)
        #
        categories_list_out.append(category_entry)
    #
    categories_list_out = add_missing_categories(categories_list_out, missing_category_name)
    return categories_list_out


def get_file_list(dataset_path):
    file_list = glob.glob(os.path.join(dataset_path, '*.*'))
    return file_list


def get_file_name_from_partial(dataset_file_partial, project_path):
    file_name = os.path.join(project_path, dataset_file_partial)
    return file_name


def get_file_name_partial(dataset_file, project_path):
    file_name_partial = dataset_file.replace(project_path + os.sep, '') if dataset_file else None
    return file_name_partial


def get_file_names_partial(dataset_files, project_path):
    file_names_partial = [get_file_name_partial(f, project_path) for f in dataset_files]
    return file_names_partial



def get_color_table(num_classes):
    num_classes_3 = np.power(num_classes, 1.0/3)
    delta_color = int(256/num_classes_3)
    colors = [(r, g, b) for r in range(0,256,delta_color)
                        for g in range(0,256,delta_color)
                        for b in range(0,256,delta_color)]
    # spread the colors list to num_classes
    color_step = len(colors) / num_classes
    colors_list = []
    to_idx = 0
    while len(colors_list) < num_classes:
        from_idx = round(color_step * to_idx)
        if from_idx < len(colors):
            colors_list.append(colors[from_idx])
        else:
            break
        #
        to_idx = to_idx + 1
    #
    shortage = num_classes-len(colors_list)
    if shortage > 0:
        colors_list += colors[-shortage:]
    #
    r_list = [c[0] for c in colors_list]
    g_list = [c[1] for c in colors_list]
    b_list = [c[2] for c in colors_list]
    max_color = (max(r_list), max(g_list), max(b_list))
    color_offset = ((255-max_color[0])//2, (255-max_color[1])//2, (255-max_color[2])//2)
    colors_list = [(c[0]+color_offset[0], c[1]+color_offset[1], c[2]+color_offset[2]) for c in colors_list]
    return colors_list


def get_color_palette(num_classes):
    colors_list = get_color_table(num_classes)
    if len(colors_list) < 256:
        colors_list += [(255,255,255)] * (256-len(colors_list))
    #
    assert len(colors_list) == 256, f'incorrect length for color palette {len(colors_list)}'
    return colors_list


def get_file_as_url(file_name):
    # streamlit can serve file content directly in base64 format
    # Note: in cases where it cannot, we will need to run an external fileserver
    file_url = None
    if file_name is not None:
        with open(file_name, 'rb') as fp:
            file_buffer = fp.read()
            file_content = base64.b64encode(file_buffer).decode('utf-8')
            file_ext = os.path.splitext(file_name)[-1]
            file_ext = file_ext[1:] if len(file_ext) > 0 else file_ext
            file_url = f'data:image/{file_ext};base64,{file_content}'
        #
    #
    return file_url

'''
def get_file_as_image(file_name):
    return PIL.Image.open(file_name) if file_name else None


def resize_image(image, output_width=None, output_height=None, with_pad=False):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    #
    border = (0,0,0,0)
    resize_width = output_width
    resize_height = output_height
    if resize_width is None and resize_height is None:
        return image, border
    #
    input_width, input_height = image.size
    input_ratio = input_width / input_height
    output_ratio = output_width / output_height
    if resize_width is None or (with_pad and output_ratio >= input_ratio):
        # pad width
        resize_width = round(input_width * resize_height / input_height)
    elif resize_height is None or (with_pad and output_ratio < input_ratio):
        # pad height
        resize_height = round(input_height * resize_width / input_width)
    #
    image = image.resize((resize_width, resize_height))
    wpad = round(output_width - resize_width)
    hpad = round(output_height - resize_height)
    top = hpad // 2
    bottom = hpad - top
    left = wpad // 2
    right = wpad - left
    border = (left, top, right, bottom)
    image = ImageOps.expand(image, border=border, fill=0)
    return image, border
'''

def pretty_json_dump(file_name, data):
    has_float_repr = False
    if hasattr(json.encoder, 'FLOAT_REPR'):
        has_float_repr = True
        float_repr_backup = json.encoder.FLOAT_REPR
    #
    json.encoder.FLOAT_REPR = lambda x: f'{x:g}'
    with open(file_name, 'w') as fp:
        json.dump(data, fp)
    #
    if has_float_repr:
        json.encoder.FLOAT_REPR = float_repr_backup
    else:
        encoder = json.encoder
        del encoder.FLOAT_REPR


def _find_annotations_info(dataset_store):
    image_id_to_file_id_dict = dict()
    file_id_to_image_id_dict = dict()
    annotations_info_list = []
    for file_id, image_info in enumerate(dataset_store['images']):
        image_id = image_info['id']
        image_id_to_file_id_dict[image_id] = file_id
        file_id_to_image_id_dict[file_id] = image_id
        annotations_info_list.append([])
    #
    for annotation_info in dataset_store['annotations']:
        if annotation_info:
            image_id = annotation_info['image_id']
            file_id = image_id_to_file_id_dict[image_id]
            annotations_info_list[file_id].append(annotation_info)
        #
    #
    return annotations_info_list


def dataset_split(dataset, split_factor, split_names, random_seed=1):
    random.seed(random_seed)
    if isinstance(dataset, str):
        with open(dataset) as fp:
            dataset = json.load(fp)
        #
    #
    dataset_train = dict(info=dataset['info'],
                         categories=dataset['categories'],
                         images=[], annotations=[])
    dataset_val = dict(info=dataset['info'],
                         categories=dataset['categories'],
                         images=[], annotations=[])
    dataset_splits = {split_names[0]:dataset_train, split_names[1]:dataset_val}

    annotations_info_list = _find_annotations_info(dataset)
    image_count_split = {split_name:0 for split_name in split_names}
    for image_id, (image_info, annotations) in enumerate(zip(dataset['images'], annotations_info_list)):
        if not annotations:
            # ignore images without annotations from the splits
            continue
        #
        image_info['file_name'] = os.path.basename(image_info['file_name'])
        if 'split_name' in image_info and image_info['split_name'] is not None:
            # print(f'file_name={image_info["file_name"]} split_name={image_info["split_name"]}')
            split_name = image_info['split_name']
            split_name = split_names[0] if 'train' in split_name else split_name #change trainval to tarin
            split_name = split_names[1] if 'test' in split_name else split_name  #change test to val
        else:
            # print(f'split_name was not found in {image_info["file_name"]}')
            split_name = split_names[0] if random.random() < split_factor else split_names[1]
        #
        dataset_splits[split_name]['images'].append(image_info)
        dataset_splits[split_name]['annotations'].extend(annotations)
        image_count_split[split_name] += 1
    #
    print('dataset split sizes', image_count_split)
    return dataset_splits


def dataset_split_limit(dataset_dict, max_num_files):
    if max_num_files is None:
        return dataset_dict
    #
    annotations_info_list = _find_annotations_info(dataset_dict)
    dataset_new = dict(info=dataset_dict['info'], categories=dataset_dict['categories'],
                       images=[], annotations=[])
    for image_id, (image_info, annotations) in enumerate(zip(dataset_dict['images'], annotations_info_list)):
        if image_id >= max_num_files:
            break
        #
        dataset_new['images'].append(image_info)
        dataset_new['annotations'].extend(annotations)
    #
    return dataset_new


def dataset_split_write(input_data_path, dataset_dict, input_data_path_split,
                        annotation_path_split):
    os.makedirs(os.path.dirname(annotation_path_split), exist_ok=True)
    pretty_json_dump(annotation_path_split, dataset_dict)
    return


def dataset_split_link(input_data_path, dataset_dict, input_data_path_split, annotation_path_split):
    utils.make_symlink(input_data_path, input_data_path_split)
    return


def dataset_load_coco(task_type, input_data_path, input_annotation_path):
    with open(input_annotation_path) as afp:
        dataset_store = json.load(afp)
    #
    for image_info in dataset_store['images']:
        image_info['file_name'] = os.path.basename(image_info['file_name'])
    #
    return dataset_store


def dataset_load_univ_ts_json(task_type, input_data_path, input_annotation_path):
    with open(input_annotation_path) as afp:
        dataset_store = json.load(afp)
    return dataset_store


def dataset_load(task_type, input_data_path, input_annotation_path, annotation_format='coco_json', is_dataset_split=False):
    if annotation_format == 'coco_json':
        dataset_store = dataset_load_coco(task_type, input_data_path, input_annotation_path)
    elif annotation_format == 'univ_ts_json':
        dataset_store = dataset_load_univ_ts_json(task_type, input_data_path, input_annotation_path)
    return dataset_store
