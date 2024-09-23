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

import json
import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd


def get_args_parser():
    DESCRIPTION = """This helper script uses the annotation file generated from Label Studio, parses it for the 
    corresponding csv's for the arc start and arc end and creates a directory structure that the 
    univariate time series dataset loader from tinyverse can pick up  
    """
    parser = ArgumentParser(description=DESCRIPTION)
    # parser.add_argument('--model_path', help="The model to compile (tflite/onnx)")
    parser.add_argument('--output_dir', default=os.getcwd(), help="Output directory to dump files in format as required by tinyverse")
    parser.add_argument('--annotation_json', required=True, help="Annotation.json file")
    parser.add_argument('--raw_data_dir', required=True, help="Raw data dir")
    parser.add_argument('--csv', action='store_true', help="Outputs csvs instead of npy files")
    # parser.add_argument('--cross_compiler', default='tiarmclang', type=str, choices=['tiarmclang'], help="Cross Compiler to compile the model")
    return parser


def main(args):
    # annotation_json = '/home/a0484689/Downloads/project-4-at-2023-09-08-15-27-12878fbe.json'
    # '/home/a0484689/Downloads/project-4-at-2023-09-08-15-04-d536b84a.json'
    # '/home/a0484689/Downloads/project-4-at-2023-09-08-12-40-0a8a4f48.json'
    with open(args.annotation_json) as aj:
        annotation_json_list = json.load(aj)

    print("Number of files annotated as per json: {}".format(len(annotation_json_list)))

    simple_annotation_dict = {}
    '''
     'filename1': [
             (label1, start_inst1, end_inst1),
             (label2, start_inst2, end_inst2),
             (label1, start_inst3, end_inst3),
             ...]
    '''
    # The below few lines uses the annotation.json file and creates a dict with just the required information
    for annotation_element in annotation_json_list:  # Each file corresponds to 1 annotation element
        data_file = annotation_element.get('file_upload')
        simple_annotation_dict[data_file] = []  # we get the basename of the file uploaded
        for annotations in annotation_element.get('annotations'):
            if annotations.get('was_cancelled'):
                continue
            results = annotations.get('result')  # result contains all the labels for the particular file
            for result in results:
                value = result.get('value')
                if not value.get('timeserieslabels'):
                    continue
                start_instant = value.get('start')
                end_instant = value.get('end')
                class_label = value.get('timeserieslabels')[0]  # Only first one is taken
                simple_annotation_dict[data_file].append((class_label, start_instant, end_instant))

    # The below set of lines uses the dict created above and then parses each data.csv file to push into its
    # class labels directory structure
    org_data_dir = args.raw_data_dir  # '../' # Arav Data
    # TODO: Can we download the csv updaded to LabelStudio
    out_dir = args.output_dir  # os.path.join(args.output_dir, 'processed_labelstudio')
    os.makedirs(out_dir, exist_ok=True)

    for filename in simple_annotation_dict.keys():
        csv_file = ''.join(filename.split('-')[1:])
        csv_abs_path = os.path.join(org_data_dir, csv_file)
        print("Reading from: {}".format(csv_abs_path))
        ts_df = pd.read_csv(csv_abs_path)
        for class_label, start_instant, end_instant in simple_annotation_dict[filename]:
            os.makedirs(os.path.join(out_dir, class_label), exist_ok=True)
            temp_df = ts_df[(ts_df["Time(sec)"] >= start_instant) & (ts_df["Time(sec)"] < end_instant)]
            if args.csv:
                out_array_file = os.path.join(out_dir, class_label,
                                              "currents_{}.csv".format(int(datetime.now().timestamp())))
                print("Saving file at: {}".format(out_array_file))
                temp_df.set_index("Time(sec)").to_csv(out_array_file)
            else:
                x_array = temp_df.values  # temp_df["I(amp)"].values
                out_array_file = os.path.join(out_dir, class_label, "currents_{}.npy".format(int(datetime.now().timestamp())))
                print("Saving file at: {}".format(out_array_file))
                np.save(out_array_file, x_array)


if __name__ == '__main__':
    """
    python labelstudio2tinyverse_univariate.py --output_dir processed_after_labelstudio \
    --annotation_json project-6-at-2023-09-09-12-33-64671231.json --raw_data_dir csvs_from_raw_data
    """
    args = get_args_parser().parse_args()
    main(args)
