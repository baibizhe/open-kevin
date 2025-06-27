"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict
import os
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from datetime import datetime
# from src import eval_kernel_against_ref
from .utils import is_e2b_available

from src.eval import eval_kernel_against_ref    # KernelBench evaluator
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
def _extract_submission(text: str) -> str:
    """
    Pull the python / triton code snip that the model wrapped in <answer> … </answer>.
    Falls back to the whole text if the tags are missing.
    """
    m = _ANS_RE.search(text)
    answer =  m.group(1).strip() if m else text.strip()
    answer = answer.replace('```python','')
    answer = answer.replace('```','')
    return answer
_PYBLOCK = re.compile(r"```python\s*\n(.*?)```", re.S | re.I)
def ref_from_prompt(txt: str) -> str | None:
    """
    Pull the reference PyTorch implementation that sits inside the
    ```python … ``` block of the *user* message.

    Returns the code string (with original indentation) or `None`
    if no block is found.
    """
    m = _PYBLOCK.search(txt)
    return m.group(1).strip() if m else None
def _safe_content(x):
    # works for both “raw string” and “[{role,content}]” chat style
    return x[0]["content"] if isinstance(x, list) else x

    
def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip().replace(',', '')
    
def extract_answer_from_model_output(text):
   """
   Extracts the value from the last <answer> tag in the text.

   Args:
       text (str): The model-generated text containing XML-style <answer> tags.

   Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

   Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
   """
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   if len(parts) < 2:  # No <answer> tag found
       return None
   last_part = parts[-1]

   # Extract content up to </answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip().replace(',', '')
   return None if answer == "..." else answer

def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.

   Explanation:
       1. Uses regex to find all numbers in the text (including negative numbers and decimals).
       2. If exactly one number is found, returns it as a float.
       3. If zero or multiple numbers are found, returns None.
   """
   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.

   Explanation:
       1. Removes dollar signs and percent symbols from the text.
       2. Uses regex to find a number that appears at the end of the text (possibly after whitespace).
       3. The pattern matches numbers that appear at the end of the string, with or without decimal points.
       4. Returns the found number as a float, or None if no match is found.
   """
   if text is None:
       return None
   text = text.replace('$', '').replace('%', '').replace(',', '')
   pattern = r'.*?(\d+\.?\d*)'
   matches = re.findall(pattern, text)
   return float(matches[-1]) if matches else None

def _prompt_text(p):
    # works for raw str **or** list-of-dict chat style
    return p if isinstance(p, str) else p[-1]["content"]
import multiprocessing as mp, signal, functools, time

_EVAL_TIMEOUT = 20          # seconds

def _call_eval(ref, sub, kwargs, q):
    try:
        num_perf_trials = kwargs.pop('num_perf_trials', 1000)
        res = eval_kernel_against_ref(ref, sub, 
        measure_performance=True,
        num_correct_trials=1,   # fewer trials for speed during RL
        verbose=False,
        num_perf_trials=num_perf_trials,
        measure_again_baseline=True)
        q.put(res)
    except BaseException as e:
        q.put(e)
def _safe_eval(ref, sub, **kwargs):
    ctx = mp.get_context("spawn")          
    q   = ctx.Queue()
    p   = ctx.Process(target=_call_eval, args=(ref, sub, kwargs, q))
    p.start()
    p.join(_EVAL_TIMEOUT)
    if p.is_alive():
        p.terminate(); p.join()
        raise RuntimeError("eval-timeout")
    out = q.get()
    if isinstance(out, BaseException):
        raise out
    return out
def accuracy_reward(prompts,completions, **kwargs):
    # print('completions',completions)
    if kwargs.get('step') < 30:
        return [(0) for i in range(len(prompts))]
    rewards = []
    # out = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for prm, cmp in zip(prompts, completions):
        single_reward =0
        res= 'tbd'
        good_exmaple =False
        ref_code = prm
        custom_code = _extract_submission(_safe_content(cmp))
        try:
            res = _safe_eval(
                ref_code,
                custom_code
            )
            # ## 22-15-48-38-767322 response of eval : compiled=True correctness=False metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'runtime_error': "module 'triton.language' has no attribute 'cbrt'"} runtime=-1.0 runtime_stats={} runtime_baseline=-1.0 runtime_stats_baseline={}
            
            single_reward +=0.3 if res.compiled else 0
            single_reward +=0.5 if 'runtime_error' not in str(res)  else 0
            single_reward +=0.5 if  res.runtime > 0.   else 0
            if res.runtime > 0.:
                accelerate_ratio_reward = max(0,res.metadata['accelerate_ratio']*10)
                single_reward+=accelerate_ratio_reward
                good_exmaple = res.metadata['accelerate_ratio']>0
        except Exception:
            pass
        rewards.append(single_reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if  good_exmaple:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} good reward :{single_reward} good custom_code {custom_code} -------------\n")
                    f.write(f"------------- {current_time} response of eval : {res} -------------\n")
            else:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} reward :{single_reward}  custom_code {custom_code} -------------\n")
                    f.write(f"------------- {current_time} response of eval : {res} -------------\n")
    # print('prompts',prompts)
    # if os.getenv("DEBUG_MODE") == "true":
    #     log_path = os.getenv("LOG_PATH")
    #     with open(log_path, "a") as f:
    #         for i in range(len(completions)):
    #             f.write(f"------------- {current_time} accuracy_reward reward: {rewards[i]} -------------\n")
    #             f.write(f"Content completions: {completions[i]}\n")
    return rewards
    # return rewards


def format_reward(prompts,completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards= [1.0 if match else 0.0 for match in matches]
    
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path, "a") as f:
            for i in range(len(completions)):
                f.write(f"------------- {current_time}  rewards: { rewards[i]} . completions: {completions[i]} -------------\n")
    return rewards




def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def extract_code(completion: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def code_reward(completions, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """
    if not is_e2b_available():
        raise ImportError(
            "E2B is not available and required for this reward function. Please install E2B with "
            "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
        )

    rewards = []
    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    try:
        """Returns a reward function that evaluates code snippets in a sandbox."""
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
        verification_info = kwargs["verification_info"]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"]))
            )
            for code, info in zip(code_snippets, verification_info)
        ]
        with Sandbox(timeout=30, request_timeout=3) as sbx:
            for script in scripts:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    output = float(execution.text)
                except (TypeError, ValueError):
                    output = 0.0
                rewards.append(output)
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)
    return rewards


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward
