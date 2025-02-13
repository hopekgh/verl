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

# def extract_solution(solution_str):
#     #find the solution in the string between [end] and [end]
#     solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     return final_solution

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model #     solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
        solution = re.search("\\[end\\](.*?)\\[end\\]", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(1)
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
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        elif str(ground_truth) in str(answer):
            return partial_score
        else:
            return format_score