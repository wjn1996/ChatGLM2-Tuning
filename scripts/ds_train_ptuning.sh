TASK_NAME=default_task
PRE_SEQ_LEN=128
LR=1e-4

CHAT_TRAIN_DATA=data/train.json
CHAT_VAL_DATA=data/dev.json

MODEL_NAME_OR_PATH=pre-trained-lm/chatglm-6b

NUM_GPUS=8

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_VERSION=v1 # v1 or v2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --num_gpus=$NUM_GPUS --master_port $MASTER_PORT chatglm_model_$MODEL_VERSION/run_ptuning.py \
--deepspeed deepspeed/deepspeed.json \
--do_train \
--train_file $CHAT_TRAIN_DATA \
--test_file $CHAT_VAL_DATA \
--prompt_column input \
--response_column output \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir ./output/deepspeed/adgen-chatglm-6b-ft-$LR \
--overwrite_output_dir \
--max_source_length 256 \
--max_target_length 256 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--predict_with_generate \
--max_steps 9000 \
--logging_steps 10 \
--save_steps 1000 \
--learning_rate $LR \
--task_name $TASK_NAME \
--base_cache_dir ./.cache \
--fp16
# --overwrite_cache \