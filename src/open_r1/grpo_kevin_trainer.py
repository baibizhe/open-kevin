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
import re
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union, Dict, List
from unittest.mock import patch
import time
import math
from open_r1.rewards_kevin import (
    accuracy_reward,_safe_eval,_extract_submission,_safe_content
)
import traceback

from datetime import datetime
import base64, json
from dataclasses import dataclass, field
from src.eval import eval_kernel_against_ref    # KernelBench evaluator

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed, broadcast
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler, RandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    
)
from transformers import Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, is_deepspeed_available
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, is_torch_mlu_available, is_torch_mps_available, is_torch_musa_available, is_torch_npu_available, is_torch_xpu_available, is_accelerate_available, is_torch_xla_available
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_utils import speed_metrics, SaveStrategy

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.extras.profiling import profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

from tqdm import tqdm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb
@dataclass
class BestAttempt:
    reward: float = -1e9
    text: str = ""
    ids: torch.Tensor = None
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
_EVAL_TIMEOUT = 20          # seconds
import multiprocessing as mp, signal, functools, time

def _call_eval(ref, sub, kwargs, q):
    try:
        res = eval_kernel_against_ref(ref, sub, 
        measure_performance=True,
        num_correct_trials=1,   # fewer trials for speed during RL
        verbose=False,
        num_perf_trials=1000,
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
    # if isinstance(out, BaseException):
    #     raise out
    return out
def kernel_eval(prompts,completions, **kwargs): # do not 封装this into class, otherwise will cause program stop if _safe_eval thought error
# print('completions',completions)
    rewards = []
    responses = []
    for prm, cmp in zip(prompts, completions):
        single_reward =0
        res= 'tbd'
        good_exmaple =False
        ref_code = prm
        custom_code = _extract_submission(_safe_content(cmp))
        try:
            res = _safe_eval(
                ref_code,
                custom_code,
                num_perf_trials=1000
            )
            responses.append(res)
            # ## 22-15-48-38-767322 response of eval : compiled=True correctness=False metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'runtime_error': "module 'triton.language' has no attribute 'cbrt'"} runtime=-1.0 runtime_stats={} runtime_baseline=-1.0 runtime_stats_baseline={}
            single_reward =0.
            single_reward +=1 if res.compiled else 0
            single_reward +=1 if 'runtime_error' not in str(res)  else 0
            single_reward +=1 if  res.runtime > 0.   else 0
            if res.runtime > 0.:
                accelerate_ratio_reward = max(0,res.metadata['accelerate_ratio']*10)
                single_reward+=accelerate_ratio_reward
        except Exception:
            error_trace = traceback.format_exc()
            responses.append(error_trace)
            # print("捕获到错误：\n", error_trace)
        rewards.append(single_reward)
    return rewards,responses
class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm
        self.temperature = args.temperature

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon = args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle.
        self._step = 0
        # edit
        self.repeat = 1
        self.eval_time = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(1, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    if torch.cuda.device_count() == 1:
                        vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                    else:
                        vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=self.args.vllm_dtype,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=False,
                        max_model_len=self.args.vllm_max_model_len,
                    )

                # Guided decoding, if enabled
                if args.vllm_guided_decoding_regex is not None:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=args.vllm_guided_decoding_regex)
                else:
                    guided_decoding = None

                # Sampling parameters
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                    guided_decoding=guided_decoding,
                    n=args.num_generations,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
                


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generaations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            mini_repeat_count=1,
            repeat_count=1,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens

    @profiling_decorator
    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights(state_dict.items())
            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            if self.accelerator.is_main_process:
                self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs


    def _extract_think(self,txt: str) -> str:
        _THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
        
        m = _THINK_RE.search(txt)
        return m.group(1).strip() if m else ""

    def _summarise_think(self,think: str, tokenizer, model, max_new_tokens: int = 48) -> str:
        """
        One-sentence TL;DR produced deterministically (T=0).
        """
        prompt = f"Summarise in one sentence:\n{think}\nTL;DR:"
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, temperature=0.0)
        return
    def safe_bcast(self,obj, src=0):
        # 所有 rank 里 list 长度必须一致（这里统一为 1）
        obj_list = [obj] if torch.distributed.get_rank() == src else [None]
        torch.distributed.broadcast_object_list(obj_list, src=src)
        return obj_list[0]
    def strip_for_bcast(self,vllm_outs):
            # 只保留 token_ids（CPU list）和文本，去掉 logits / gpu tensor
            return [
                dict(text=o.text,
                    token_ids=o.token_ids.cpu().tolist())   # <= 2-3 KB
                for o in vllm_outs
            ]
    def _vllm_generate(self,prompts: list[str],
                       refinement_step,
        ) -> tuple[list[torch.Tensor], list[str]]:
        """
        One call to vLLM → returns **completion_ids** (already on `device`)
        and **completion_text** (decoded).
        Handles gather / broadcast so each GPU sees its slice.
        """
        self.accelerator.wait_for_everyone() 
        pure_prompts = [
            base64.b64encode(p).decode() if isinstance(p, bytes) else str(p)
            for p in prompts
        ]
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            sampling_params = self.sampling_params
            if  refinement_step==0:
                uniq = list(dict.fromkeys(all_prompts))   
                outs = self.llm.generate(uniq, sampling_params=sampling_params, use_tqdm=False)
            else:
                train_sampling_params = SamplingParams(
                        temperature=self.args.temperature,
                        max_tokens=self.max_completion_length,
                        n=1,
                    )
                outs = self.llm.generate(all_prompts, sampling_params=train_sampling_params, use_tqdm=False)
            flat_ids = [torch.tensor(o.token_ids, device=self.accelerator.device) for g in outs for o in g.outputs]
        else:
            flat_ids = [None] * len(all_prompts)
        self.accelerator.wait_for_everyone() 
        flat_ids = broadcast_object_list(flat_ids, from_process=0)
        flat_ids = broadcast_object_list(flat_ids, from_process=0)
        
        slice_ = slice(self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts))
        chunk = flat_ids[slice_]
        chunk = [t.to(self.accelerator.device, non_blocking=True) for t in chunk]
        chunk_padded = pad(chunk, padding_value=self.processing_class.pad_token_id)
        chunk_text = self.processing_class.batch_decode(chunk_padded, skip_special_tokens=True)
        return chunk_padded, chunk_text

    def _extract_kernel_and_thinking(self, completion_text):
        """Extract CUDA kernel and thinking process from completion."""
        import re
        
        # Extract thinking
        thinking_match = re.search(r'<think>(.*?)</think>', completion_text, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        sumamry_match = re.search(r'<summary>(.*?)</summary>', completion_text, re.DOTALL)
        summary = sumamry_match.group(1).strip() if sumamry_match else ""
        
        # Extract kernel from answer tags
        answer_match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL)
        if answer_match:
            kernel = answer_match.group(1).strip()
        else:
            code_match = re.search(r'```(?:cuda|cpp|python|c\+\+)?\n(.*?)\n```', completion_text, re.DOTALL)
            kernel = code_match.group(1).strip() if code_match else completion_text
        
        return kernel, thinking,summary

    # Helper method to add:
    def _construct_staged_feedback_prompt(self,stage: int, response_data: object) -> str:
        """
        Constructs a feedback prompt tailored to the current refinement stage.

        Args:
            stage: The current stage (0: compile, 1: correct, 2: speed).
            response_data: The evaluation object containing feedback.
                        (Assumes it has .message and .reward attributes)
        """

        if stage == 0:  # Compilation Failed
            error_trace = str(response_data)
            return (
                "The previous triton code attempt failed to compile. Your task is to fix the compilation errors.\n\n"
                "## Error Trace\n"
                "```\n"
                f"{error_trace}\n"
                "```\n\n"
                "Please provide a corrected version of the code that compiles successfully. Follow the original instruction's format."
            )
        elif stage == 1:  # runtime error Failed
            error_trace = str(response_data.metadata)
            return (
                "The previous triton code  compiles, but it produces runtime error. Your task is to fix the runtime error.\n\n"
                "## runtime error  Feedback\n"
                "```\n"
                f"{error_trace}\n"
                "```\n\n"
                "Follow the original instruction's format."
            )
        elif stage == 2:  # Correctness Failed
            error_trace = str(response_data.metadata)
            return (
                "The previous triton code  compiles, but it produces incorrect results compare to original pytorch code. Your task is to fix the correctness.\n\n"
                "## Correctness Feedback\n"
                "```\n"
                f"{error_trace}\n"
                "```\n\n"
                "Analyze the code's logic and provide a corrected version that is numerically accurate with original pytorch code. Follow the original instruction's format."
            )
        else:  # stage == 3, Speed Optimization
            return (
                "Excellent! The code is now correct. Your final task is to optimize it for maximum speed.\n\n"
                "## Performance Feedback\n"
                f"Current performance score: {response_data.metadata['accelerate_ratio']}. Analyze the code for potential bottlenecks (e.g., memory access, block sizes) and provide a more optimized version. "
                "Even small improvements are valuable.\n\n"
                "Please provide a more performant version of the code. Follow the original instruction's format."
            )
    def remove_think_tags(self,text):
        """
        删除字符串中所有 <think>...</think> 标签对及其之间的内容
        支持多行内容和多个标签对
        """
        pattern = r'<think\b[^>]*>.*?</think\s*>'
        return re.sub(pattern, '', text, flags=re.DOTALL)
    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        self.args.refinement_steps = 16
        device = self.accelerator.device
        reference_codes = [ex["reference"] for ex in inputs]
        base_prompts_struct = [ex["prompt"] for ex in inputs] # list[list[dict]]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        base_prompts_struct = [ex["prompt"] for ex in inputs]
        prompts_struct = base_prompts_struct
        prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
            for p in prompts_struct
        ]
        initial_comp_ids, initial_comp_txt = self._vllm_generate(
            prompts=prompts_text,
            refinement_step=0, # Initial generation
        )

        rewards_this_step, responses_this_step = kernel_eval(
            prompts=reference_codes,
            completions=initial_comp_txt,
        )
        if self.accelerator.process_index == 2:
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f'------------- "rewards_this_step",{rewards_this_step} responses_this_step {responses_this_step}-------------\n')
        reward = rewards_this_step[0]
        response = responses_this_step[0]
        
        if reward == 0: current_stage = 0   # Compilation failure
        elif reward == 1: current_stage = 1 # runtime error 
        elif reward == 2: current_stage = 2 # Correctness failure
        else: current_stage = 3             # Ready for speed optimization
        feedback_msg = self._construct_staged_feedback_prompt(current_stage, response)
        critic_turn = {"role": "user", "content": initial_comp_txt[0]+"\n"+feedback_msg} #我把这里变成单独assistant_content就会报错 ，很神奇
        prompt_with_fdback = [base_prompts_struct[0] + [critic_turn]]

        next_prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
            for p in prompt_with_fdback
        ]

        final_comp_ids, final_comp_txt = self._vllm_generate(
            prompts=next_prompts_text,
            refinement_step=1, # Pass step for logging/tracking
        )
        
        if self.accelerator.process_index == 2:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a") as f:
                    f.write(f"------------- base_prompts_struct {base_prompts_struct} -------------\n")
                    f.write(f"------------- next_prompts_text {next_prompts_text} -------------\n")
                    f.write(f"------------- final_comp_txt {final_comp_txt} -------------\n")


        prompts_text_final = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"]
            for p in prompt_with_fdback
        ]

        prompt_inputs = self.processing_class(
            prompts_text_final, return_tensors="pt",
            padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        # Use the best completion IDs gathered from the entire refinement process.
        completion_ids = final_comp_ids
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx_updates = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        eos_idx[is_eos.any(dim=1)] = eos_idx_updates
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        completion_mask_for_metric = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)  # We only need to compute logits for the completion tokens

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for msg_ctx, comp_txt in zip(prompts_text_final, completions_text):
                if isinstance(msg_ctx, list) and msg_ctx and isinstance(msg_ctx[-1], dict):
                    bootstrap = msg_ctx[-1]["content"] if msg_ctx[-1]["role"] == "assistant" else ""
                else:
                    bootstrap = ""
                completions.append([{"role": "assistant", "content": bootstrap + comp_txt}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts_text_final), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            reward_kwargs['step'] = self._step
            # edit
            output_reward_func = reward_func(prompts=reference_codes,completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

            
        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations ).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations ).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # if self.accelerator.process_index == 2:
        #     print('advantages',advantages)
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts_text_final),
            (self.accelerator.process_index + 1) * len(prompts_text_final),
        )
        advantages = advantages[process_slice]
        # if self.accelerator.process_index == 2:
        #     print('advantages',advantages) #torch.Size([1, 1405
        
        mode = "eval" if self.control.should_evaluate else "train"
        
        if advantages.shape[0] != 0:
            with torch.no_grad():
                if self.num_iterations > 1:
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    old_per_token_logps = None
                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
        else:
            old_per_token_logps = None
            ref_per_token_logps = None
        mode = "eval" if self.control.should_evaluate else "train"

        completion_mask_for_metric = self.accelerator.gather_for_metrics(completion_mask_for_metric.sum(1)).float().mean().item()
            
        self._metrics[mode]["completion_length"].append(completion_mask_for_metric)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        mode = "eval" if self.control.should_evaluate else "train"
        # Compute the per-token log probabilities for the model
        advantages = inputs["advantages"]
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        
        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            second_item = self.beta * (torch.exp(per_token_logps.detach() - ref_per_token_logps.detach())-1)
            self._metrics[mode]["second_item"].append(second_item.mean().item())
        else:
            second_item = torch.zeros_like(per_token_logps)

        # Compute the loss

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        self._metrics[mode]["first_item"].append(per_token_loss1.detach().mean().item())
        self._metrics[mode]["first_item_div_second_item"].append((per_token_loss1.detach()).mean().item()/(second_item.mean().item()+ 1e-6))
        self._metrics[mode]["total_sum"].append((per_token_loss1.detach() + second_item).mean().item())
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
    
    def evaluate(self, eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        metric = {}
        metric['eval_accuracy'] = 0
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        pbar = tqdm(eval_dataloader) if self.accelerator.is_main_process else eval_dataloader
        total = 0
        correct = 0
        device = self.accelerator.device
        accuracy = torch.tensor(0, device=device)
        for inputs in pbar:
            solution = [x['solution'] for x in inputs]
            solution = [extract_answer_from_dataset(a) for a in solution]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

            # Generate completions using either vLLM or regular generation
            if self.args.use_vllm:
                # First, have main process load weights if needed
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step

                # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
                all_prompts_text = gather_object(prompts_text)
                all_solution = gather_object(solution)
                
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                    total += len(ordered_set_of_prompts)
                    all_solution = all_solution[:len(ordered_set_of_prompts)]

                    # Guided decoding, if enabled
                    if self.args.vllm_guided_decoding_regex is not None:
                        guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.args.vllm_guided_decoding_regex)
                    else:
                        guided_decoding = None
                        
                    inference_sampling_params = SamplingParams(
                        temperature=0,
                        max_tokens=self.max_completion_length,
                        guided_decoding=guided_decoding,
                        n=1,
                    )
                    completion_ids = []
                    all_outputs = self.llm.generate(
                        ordered_set_of_prompts, sampling_params=inference_sampling_params, use_tqdm=False,
                    )
                    for outputs in all_outputs:
                        for output in outputs.outputs:
                            completion_ids.append(output.token_ids)
                            
                    completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                    completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                    predicted = [x for x in completions_text]

                    try:
                        # Extract answer and check correctness
                        for predicted, solution in zip(predicted, all_solution):
                            
                            # Try different matching methods
                            if predicted == solution:  # Exact match
                                is_correct = True
                            else:
                                # Try single number matching
                                pred_num = extract_single_number(predicted)
                                exp_num = float(solution)
                                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                                    is_correct = True
                                else:
                                    # Try last number matching
                                    pred_num = extract_last_number(predicted)
                                    exp_num = extract_last_number(solution)
                                    is_correct = (pred_num is not None and exp_num is not None and pred_num == exp_num)
                                    

                            predicted2 = extract_answer_from_model_output(predicted)
                            if predicted2 is not None:
                                if predicted2 == solution:  # Exact match
                                    is_correct = True
                                else:
                                    # Try single number matching
                                    pred_num = extract_single_number(predicted2)
                                    exp_num = float(solution)
                                    if pred_num is not None and exp_num is not None and pred_num == exp_num:
                                        is_correct = True
                                    else:
                                        # Try last number matching
                                        pred_num = extract_last_number(predicted2)
                                        exp_num = extract_last_number(solution)
                                        is_correct = (pred_num is not None and exp_num is not None and pred_num == exp_num)
                            
                            # Update counter for correct answers
                            if is_correct:
                                correct += 1
                            
                    except Exception as e:
                        print("\nFailed to parse model output for prompt:")
                        print("Error:", e)
                        print("-"*50)
           
                self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            accuracy = torch.tensor((correct / total) * 100, device=device)
            print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
            print("=" * 50)
            mode = "eval" if self.control.should_evaluate else "train"
            self._metrics[mode][f"accuracy"].append(accuracy.item())
        
        accuracy = broadcast(accuracy)
        metric['eval_accuracy'] += accuracy.item()
        
        return metric
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if self.accelerator.is_main_process:
            if self.state.global_step == 0:
                print('Begin Training')
                self.train_start_time = time.perf_counter()
                
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        if inputs["advantages"].shape[0] == 0:
            model.zero_grad()
            return torch.tensor(0.0).detach().to(self.args.device)
        
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)
            return loss.detach()

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()
        
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.accelerator.is_main_process:
            eval_start_time= time.perf_counter()

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
        if self.accelerator.is_main_process:
            eval_end_time= time.perf_counter()
            self.eval_time += eval_end_time - eval_start_time

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
