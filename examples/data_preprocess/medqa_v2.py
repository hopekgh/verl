# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    #find the solution in the string between [end] and [end]
    solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
    assert solution is not None
    final_solution = solution.group(1)
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/medqa')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'HanyangMed/final_diagnosis_medqa'

    dataset = datasets.load_dataset(data_source, 'default')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "You are an medical AI that thinks by mimicking real medical process. Let's think step by step and output the final answer between [end] and [end]. Use exactly same words and language given in Options. English, Chinese or Taiwanese"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            option_raw = example.pop('options')

            question = question_raw + ' Options: '+ option_raw + 'Instruction: ' + instruction_following
            

            answer_raw = example.pop('answer')
            solution = answer_raw
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
