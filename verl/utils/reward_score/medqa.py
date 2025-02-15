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

import re
import random
from transformers.utils import logging
# def extract_solution(solution_str):
#     #find the solution in the string between [end] and [end]
#     solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     return final_solution
logger = logging.get_logger(__name__)

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']
    #only look at the solution after "<|im_start|>assistant"
    solution_str = solution_str.split("<|im_start|>assistant")[1]

    if method == 'strict':
        #if there is no [end] in the solution, return None
        if "[end]" not in solution_str:
            return None
        
        # this also tests the formatting of the model #     solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
        solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('[end]')[1]
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def check_too_many_end(solution_str):
    return solution_str.count("[end]") > 5

def normalize_string(text):
    return text.lower().strip()

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.1, partial_score=0.5):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        partial_score: score awarded if ground truth is contained in answer
    """
    # random between 0 to 64 int and if it is 0, do print
    do_print = random.randint(0, 32) == 0
    if do_print:
        print(f"Solution: {solution_str}, Ground Truth: {ground_truth}")
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        final_score = 0.0
        return final_score
    normalized_answer = normalize_string(answer)
    normalized_truth = normalize_string(ground_truth)
    is_correct = normalized_answer == normalized_truth
    final_score = 0.0
    if do_print:
        
        print(f"Answer: {normalized_answer}, Ground Truth: {normalized_truth}, Is Correct: {is_correct}")
        #logger.info(f"Answer: {normalized_answer}, Ground Truth: {normalized_truth}, Is Correct: {is_correct}")
    else:
        if is_correct:
            final_score = score
        elif str(ground_truth) in str(answer):
            final_score = partial_score
        else:
            final_score = format_score
    if check_too_many_end(solution_str):
        final_score = final_score - format_score
    if do_print:
        print(f"Final Score: {final_score}")
        #logger.info(f"Final Score: {final_score}")
    return final_score