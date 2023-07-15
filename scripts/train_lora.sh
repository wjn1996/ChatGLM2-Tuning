
PEFT_TYPE=lora
LORA_DIM=8
LR=2e-2
NUM_GPUS=1
TRAIN_DATA=./data/train.json
EVAL_DATA=./data/dev.json

MODEL_VERSION=v1 # v1 or v2

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS chatglm_model_$MODEL_VERSION/run_peft.py \
    --do_train \
    --train_file $TRAIN_DATA \
    --validation_file $EVAL_DATA \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path THUDM/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
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
    --peft_type $PEFT_TYPE \
    --lora_dim $LORA_DIM \
    --task_name $TASK_NAME \
    --base_cache_dir ./cache
    # --quantization_bit 4
    # --overwrite_cache \

