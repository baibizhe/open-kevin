# export WANDB_CONSOLE=off 
# export WANDB_MODE=offline
# accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
#     --num_processes=7  src/open_r1/grpo_gsm.py \
#     --config recipes/gsm8k/Qwen2.5-1.5B-Instruct.yaml  \
#     --output_dir=/data/GRPO \
#     --save_strategy='best' \
#     --model_name_or_path=/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/ckpts/llm/Qwen2.5-Coder-3B \
#     --dataset_name=/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/r1-reasoning/gsm8k  \
#     --num_generations=16 
export WANDB_CONSOLE=off 
export WANDB_MODE=offline
export DEBUG_MODE=true
timestamp="$(date '+%Y-%m-%d_%H-%M-%S')"
export LOG_PATH="debug_logs/kevin_multiturn_${timestamp}_debug.log"
accelerate launch  --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=7  src/open_r1/grpo_kevin.py \
    --config recipes/KEVIN/Qwen2.5-coder-7B-Instruct.yaml  \
    --output_dir='save_ckpts/KEVIN/Qwen-2.5-7B' \
    --save_strategy='steps' \
    --dataset_train_split='main' \
    --eval_strategy='no' \
    --save_steps=100 \
    --num_train_epochs=20 \
    --model_name_or_path=/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/ckpts/llm/Qwen2.5-Coder-7B-Instruct \
    --dataset_name=/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/datas/KernelBench/data  \
    --num_generations=7