# ChatGLM2-Tuning


## 一、介绍

[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，进一步优化了模型，使得其具有更大的性能、更长的输入、更有效的部署和更开放的协议。ChatGLM2-6B也因此登顶C-Eval榜单。

本项目结合了 **ChatGLM-6B** 和 **ChatGLM2-6B** 进行微调，可进行全参数微调，也可以使用如下优化技术：
- Peft参数有效性训练：Ptuning、Prompt-tuning、Prefix-tuning、LoRA；
- DeepSpeed ZeRO训练；
- 量化感知训练&推理部署；

---

开发进程：
- 代码调试 ✅
- 全参数训练 ✅ 
- 参数有效性训练 ✅ 
- 量化感知训练 ✅ 
- 指令微调 ✅ 
- 多轮对话 ✅ 

---

## 二、开始使用
### 2.1 环境安装
首先需要下载本仓库：
```shell
git clone https://github.com/wjn1996/ChatGLM2-Tuning
cd ChatGLM2-Tuning
```

安装环境依赖：
```
pip install -r requirements.txt
```

### 2.2 数据集准备

##### （1）使用自定义的指令微调数据集

指令微调数据集中包括一个任务的指令（instruction），以及对应任务的输入（input）输出（output）。模型在训练时只会计算output的loss。

数据集格式样例：
```json
{
  "instruction": "请为下面的评论的情感类别进行分类，候选为【积极】和【消极】",
  "input": "《消失的她》这部电影很好看，但是我觉得大多数人看完后都会emo",
  "output": "消极",
}
```



##### （2）使用自定义的多轮对话数据集

多轮对话数据集在训练时有两种模式，一种是in-the-loop，另一种是session：
- **in-the-loop**：一个多轮对话根据对话轮次拆解成多个样本，在训练时每个样本视为独立，根据对话历史history和当前的prompt，计算response的loss；ChatGLM2-6B默认采用这种方式进行训练多轮对话。

```json
{
  "prompt": "长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", 
  "response": "用电脑能读数据流吗？水温多少", 
  "history": []
}
{
  "prompt": "95", 
  "response": "上下水管温差怎么样啊？空气是不是都排干净了呢？", 
  "history": [
    ["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"]
  ]
}
{
  "prompt": "是的。上下水管都好的", 
  "response": "那就要检查线路了，一般风扇继电器是由电脑控制吸合的，如果电路存在断路，或者电脑坏了的话会出现继电器不吸合的情况！", 
  "history": [
    ["长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线", "用电脑能读数据流吗？水温多少"], 
    ["95", "上下水管温差怎么样啊？空气是不是都排干净了呢？"]
  ]
}
```
> 以上面的一个多轮对话为例，in-the-loop设置中，数据处理时，一个多轮对话将会生成3个独立的样本，每个样本是一个序列，包含对话历史、当前的prompt以及输出response。
- **session**：将整个多轮对话当作一个样本，计算所有token（或每一轮对话的output）对应的loss；

```json
{
  "prompt": [
    "长城h3风扇不转。继电器好的。保险丝好的传感器新的风扇也新的这是为什么。就是继电器缺一个信号线",
    "95",
    "是的。上下水管都好的"
  ],
  "response": [
    "用电脑能读数据流吗？水温多少",
    "上下水管温差怎么样啊？空气是不是都排干净了呢？",
    "那就要检查线路了，一般风扇继电器是由电脑控制吸合的，如果电路存在断路，或者电脑坏了的话会出现继电器不吸合的情况！"
  ],
}
```
> 以上面的多轮对话为例，只会生成一个样本，每一轮对话的prompt和response拼接起来，所有轮次的对话拼接起来，形成类似“Q1 A1 Q2 A2 ...”格式的序列。


##### （3）获取开源评测数据集

TODO


### 2.3 模型训练

训练采用Causal LM进行训练，前向传播时只会计算指定token的loss，对于指令、对话历史、input和padding部分可以通过设置label为“-100”忽略对应的loss计算。

##### （1）P-tuning训练
```bash
TASK_NAME=default_task # 指定任务名称
PRE_SEQ_LEN=128 # prefix token数量
LR=1e-4 # 学习率

CHAT_TRAIN_DATA=data/train.json
CHAT_VAL_DATA=data/dev.json

MODEL_NAME_OR_PATH=pre-trained-lm/chatglm-6b

NUM_GPUS=8

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

MODEL_VERSION=v1 # V1:初始化为ChatGLM-6B，V2:初始化为ChatGLM2-6B

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
```

参考脚本：scripts/ds_train_ptuning.sh

##### （2）LoRA训练
```bash
TASK_NAME=default_task # 指定任务名称
# PRE_SEQ_LEN=128
PEFT_TYPE=lora # 指定参数有效性方法
LORA_DIM=8 # 指定LoRA Rank
LR=1e-4 # 学习率

CHAT_TRAIN_DATA=./data/train.json
CHAT_VAL_DATA=./data/dev.json

MODEL_NAME_OR_PATH=./pre-trained-lm/chatglm-6b

NUM_GPUS=8

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
deepspeed --num_gpus=$NUM_GPUS --master_port $MASTER_PORT chatglm_model_v1/run_peft.py \
--deepspeed deepspeed/deepspeed.json \
--do_train \
--train_file $CHAT_TRAIN_DATA \
--test_file $CHAT_VAL_DATA \
--prompt_column input \
--response_column output \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir ./output/deepspeed/chatglm-6b-$TASK_NAME-$PEFT_TYPE-$LORA_DIM-$LR \
--overwrite_output_dir \
--max_source_length 256 \
--max_target_length 1024 \
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
--base_cache_dir ./.cache/ \
--fp16
# --overwrite_cache \
```

参考脚本：scripts/ds_train_peft.sh

如果要使用INT4量化感知训练，添加参数

> --quantization_bit 4

即可。

### 2.4 模型推理与部署

可直接参考ChatGLM-6B或ChatGLM2-6B的部署即可。