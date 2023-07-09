# ChatGLM2-Tuning


## 一、介绍

[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B) 是开源中英双语对话模型 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，进一步优化了模型，使得其具有更大的性能、更长的输入、更有效的部署和更开放的协议。ChatGLM2-6B也因此登顶C-Eval榜单。

本项目基于 **ChatGLM2-6B** 进行微调，可进行全参数微调，也可以使用如下优化技术：
- Peft参数有效性训练：Ptuning、Prompt-tuning、Prefix-tuning、LoRA；
- DeepSpeed ZeRO训练；
- 量化感知训练&推理部署；

---

开发进程：
<input type="checkbox" unchecked disabled/> 代码调试；
<input type="checkbox" disabled/> 全参数训练
<input type="checkbox" disabled/> 参数有效性训练
<input type="checkbox" disabled/> 量化感知训练
<input type="checkbox" disabled/> 指令微调
<input type="checkbox" disabled/> 多轮对话

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

##### （1）全量参数训练
TODO

##### （2）参数有效性训练
TODO


##### （3）量化感知训练
TODO

### 2.4 模型推理与部署

##### （1）直接使用代码调用

可以通过如下代码调用 ChatGLM2-6B 模型来生成对话：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True, device='cuda')
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM2-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:

1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。

如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```

TODO