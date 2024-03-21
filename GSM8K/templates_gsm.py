# Script to train on the data
TRAIN_TEMPLATE = """export CUDA_VISIBLE_DEVICES={cuda_devices}
export NCCL_P2P_LEVEL=NVL
torchrun --nproc_per_node={num_cuda_devices} --master_port={port} {base}/train.py \\
    --model_name_or_path {base_model} \\
    --model_max_length 1024 \\
    --data_path {data_path} \\
    --bf16 True \\
    --output_dir {output_path} \\
    --num_train_epochs {epochs} \\
    --per_device_train_batch_size 1 \\
    --per_device_eval_batch_size 1 \\
    --gradient_accumulation_steps {grad_acc_steps} \\
    --evaluation_strategy "no" \\
    --save_strategy "steps" \\
    --save_steps 2000 \\
    --save_total_limit 1 \\
    --learning_rate 2e-5 \\
    --weight_decay 0. \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --fsdp "full_shard auto_wrap offload" \\
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'"""

# Script that is used for evaluate on test and train
EVAL_SCRIPT_TEMPLATE = "CUDA_LAUNCH_BLOCKING=1 NCCL_P2P_LEVEL=NVL CUDA_VISIBLE_DEVICES={cuda_devices} python {base}/GSM8K/eval.py --mode {test_mode} {extra_flags} --model {model_path} --iterations 1 --batch_size 64 --output_path {eval_output_directory}"

# Script to filter out wrong answers
FILTER_SCRIPT_TEMPLATE = "python {base}/GSM8K/filter.py --seed_data_path {seed_data_path} --input_data_path {input_data_path} --output_data_path {output_data_path}"

# Script to generate new data from wrong answers
GENERATION_SCRIPT_TEMPLATE = "python {base}/GSM8K/generate_data.py generate_data --output_dir {generation_output_directory} --seed_tasks_path {seed_path} --instructions_per_seed_task {instructions_per_seed_task} --generated_instructions_per_seed_task {instructions_per_seed_task} --model_name {generation_model}"

# Script to process and combine data
PROCESS_SCRIPT_TEMPLATE = (
    "python {base}/process_data.py --output {process_output_path} convert {process_input_path}"
)
COMBINE_SCRIPT_TEMPLATE = "python {base}/process_data.py --output {combine_output_path} combine {combine_input_path} {combine_input_path_2}"
