# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any
from datasets import load_dataset, DatasetDict,concatenate_datasets
import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards_kevin import (
    accuracy_reward,
    format_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import  ModelConfig, ScriptArguments, TrlParser, get_peft_config
import random 
import numpy as np
from open_r1.grpo_kevin_trainer import GRPOTrainer
import ast
logger = logging.getLogger(__name__)
import time
import pathlib

def keep_relevant_nodes(tree: ast.Module, class_name="Model"):
    """只留 import / import-from / 指定 classdef 节点"""
    new_body = []
    for node in tree.body:
        # 顶层 import
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            new_body.append(node)
        # class Model
        elif isinstance(node, ast.ClassDef) and node.name == class_name:
            new_body.append(node)
    return ast.Module(body=new_body, type_ignores=[])

def extract_model(src_code: str, class_name="Model"):
    tree = ast.parse(src_code)
    model_tree = keep_relevant_nodes(tree, class_name=class_name)

    model_code = ast.unparse(model_tree)
    return model_code

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.

   Explanation:
       1. Takes a list of message dictionaries in the typical chat format.
       2. Extracts the 'content' field from each message and strips whitespace.
       3. Joins all content strings with newlines to create a single prompt.
       4. This preserves the training format while converting from structured messages to a string.
   """
   return "\n".join([msg["content"].strip() for msg in messages])




@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'format_deepseek', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
def build_kernelbench_dataset(
    levels:      List[int] | None = None,
    system_prompt:          str   = "You are a world-class GPU-kernel engineer.",
    split:                  str   = "main",
    data_path: str ="/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/datas/KernelBench/data"
) -> DatasetDict:
    """
    Load & convert KernelBench into GRPO-ready DatasetDict.

    Parameters
    ----------
    levels : list[int] | None
        Which benchmark levels to keep.  None ⇒ keep all four levels.
    system_prompt : str
        First message injected into every conversation.
    split : str
        'train' / 'validation' / 'all'.  The original dataset has no
        official splits, so we expose this arg for compatibility only.

    Returns
    -------
    datasets.DatasetDict
        With a single key equal to `split` containing
        `{prompt, level, problem_id, reference}` columns.
    """
    hf_ds = load_dataset(data_path)   # 270 rows:contentReference[oaicite:0]{index=0}

    # Merge the four parquet splits into one long table
    print(hf_ds)
    # all_rows = sum([hf_ds[k] for k in hf_ds])
    all_rows = concatenate_datasets(list(hf_ds.values()))
    print(all_rows)

    if levels is not None:
        all_rows = all_rows.filter(lambda ex: ex["level"] in levels)

    processed = all_rows.map(
        lambda ex: _row_to_conv(ex, system_prompt),
        remove_columns=all_rows.column_names,
        desc="Formatting KernelBench → chat prompts"
    )

    return DatasetDict({split: processed})
def _row_to_conv(ex: Dict[str, Any], sys_prompt: str) -> Dict[str, Any]:
    clean_code = extract_model(ex['code'])
    user_msg = (
        f'''
        Your job is to replace the forward pass of a given PyTorch “reference model” with a custom CUDA GPU kernel that is BOTH:
            • Correct - returns outputs numerically equivalent to the reference
            • Fast    - wall-clock runtime  lower than the reference code
        Mandatory format
        ────────────────
        1. First thinks about the reasoning process in the mind and then provides  with the code answer. The reasoning process and answer answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>\n<answer> code here </answer>.
        2. Inside the <answer> </answer> tags, start code with ```python and end code with ```, do not write comment for code, 
        3. It *must* define **class ModelNew(torch.nn.Module)** with a correct `__init__(...)` and `forward(...)` .
        4. You're not allowed to use built-in functions inside torch.nn and torch such as torch.relu, torch.norm or torch.nn.Conv2d ... etc .
        5. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. 
        6. Follow Triton best practices:\n
            • Ensure coalesced/global-memory access & avoid bank conflicts.\n
            • Exploit vectorization (`VEC_SIZE` ≥ 4) and warp primitives when helpful.\n
            • Minimize shared-memory usage and register pressure.\n
            • operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu)\n
            • .....some other techniques that  could accelerate.'''
        f"You are given a PyTorch reference implementation for problem "
        f"{ex['name']}⟩.\n\n"
        f"Task:  Re-implement the pytorch forward function inside the class {ex['name']} as a highly-optimised triton kernel.\n"
        "Reference PyTorch:\n"
        "```python\n"
        f"{clean_code.strip()}\n"
        "```"
        "Here is an example"
        """ 
        import torch
        import triton
        import triton.language as tl
        @triton.jit
        def softsign_kernel(
            x_ptr, 
            out_ptr, 
            n_elements, 
            BLOCK_SIZE: tl.constexpr, 
        ):
            pid = tl.program_id(0)  # 1D launch grid
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements  # Avoid out-of-bounds
            
            x = tl.load(x_ptr + offsets, mask=mask)
            abs_x = tl.abs(x)
            denominator = 1.0 + abs_x
            output = x / denominator
            
            tl.store(out_ptr + offsets, output, mask=mask)


        class ModelNew(torch.nn.Module):
            def __init__(self):
                super(ModelNew, self).__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not x.is_cuda or x.numel() < 1024:
                    return x / (1 + torch.abs(x))
                
                x_contig = x.contiguous()
                output = torch.empty_like(x_contig)
                n_elements = x_contig.numel()
                
                BLOCK_SIZE = triton.next_power_of_2(min(2048, n_elements))
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
                
                softsign_kernel[grid](
                    x_contig, output, n_elements,
                    BLOCK_SIZE=BLOCK_SIZE
                )
                return output
                """
       
        
    )
    # print('clean_code',clean_code)
    return {
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_msg},
        ],
        # Extra metadata that the reward fn needs
        "level":        ex["level"],
        "problem_id":   ex["problem_id"],
        "reference":    ex["code"],       # (optional) can be handy for debugging
    }
def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    
    if training_args.local_rank==0:
        log_level = logging.INFO
    else:
        log_level = logging.ERROR

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    system_prompt = """
    You are an expert in writing triton code.  
    """
    dataset = build_kernelbench_dataset(levels=[1, 2],data_path='/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/datas/KernelBench',system_prompt=system_prompt)          # DatasetDict

    ################
    # Load tokenizer
    ################
    # print('dataset',dataset)
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example["question"]})
        return {"prompt": prompt}

    if training_args.sample_num != 0:
        for split in dataset:
           dataset[split] = dataset[split].select(range(training_args.sample_num))

    # dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
        if "answer" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("answer","solution")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    from datetime import datetime
    
    all_datas = dataset[script_args.dataset_train_split]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # if os.getenv("DEBUG_MODE") == "true":
    #     log_path = os.getenv("LOG_PATH")
    #     with open(log_path, "a") as f:
    #         for i in range(len(all_datas)):
    #             f.write(f"dataset: {all_datas[i]}\n")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint
    

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if trainer.accelerator.is_main_process:  
        train_end_time = time.perf_counter()
        print("\nTraining + Eval time:", train_end_time - trainer.train_start_time)
        print("\nEval time:", trainer.eval_time)
        print("\nTraining time:", train_end_time - trainer.train_start_time - trainer.eval_time)
    
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    # Call the function to set random seed for reproducibility
    set_random_seed(42)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
