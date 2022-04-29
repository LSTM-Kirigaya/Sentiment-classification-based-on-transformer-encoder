# Sentiment-classification-based-on-transformer-encoder

基于`Bert`进行的情感分类，使用`colossalai`框架进行包装和优化

## 前置工作

数据安放在了`data/simplifyweibo_4_moods.csv`, 使用中文预训练`Bert`自带的词表进行嵌入。数据文件夹结构:
```
|-> data
    |-> simplifyweibo_4_moods.csv
    |-> stop_words.txt
```

### 安装相关的库
主要用到的库为`torch`, `colossalai`和 `transformers`，直接pip3 install 一下，需要注意一下，`co`

### 处理数据
一键运行`process.ipynb`即可，会在`data`下生成处理好的id序列数据`token_id_4_mood.json`



## 训练

### 准备训练
训练配置放在`config.py`中，以下是示例：

```python
from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 10  # for debug
DEBUG_MODE = False

BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE = 6e-5
LAST_MODEL = "model/2022-04-29 23-04-10.pth"        # path of model saved last time, please set None when you first run

fp16=dict(          
    mode=AMP_TYPE.TORCH
)
```
它会被注册到colossal的全局字典中，直接修改即可。混合精度配置根据选取的框架而定，本次使用使用

```bash
$torchrun --nproc_per_node 1 --master_addr localhost --master_port 8008 train_bert.py
```

### 训练策略

在训练之前请务必完成下属的测试:

1.  `dataset.py`中的数据加载器是否正常工作 
2.  `train_bert.py`是否可以跑通  
3. `train_bert.py`与本地磁盘的IO是否正常
4. `train_bert.py`是否可以在小批量数据上正常下降



使用如下配置和运行`dataset.py`完成测试1：

```python
from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 10  # for debug
DEBUG_MODE = True

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 6e-5
LAST_MODEL = None        # path of model saved last time

fp16=dict(          
    mode=AMP_TYPE.TORCH
)
```



使用如下配置和运行`dataset.py`完成测试2 & 4：

```python
from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 10  # for debug
DEBUG_MODE = True

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 6e-5
LAST_MODEL = None        # path of model saved last time

fp16=dict(          
    mode=AMP_TYPE.TORCH
)
```

> 测试完成后，应该确定大概的学习率



使用如下配置和运行`dataset.py`完成测试3：

```python
from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 10  # for debug
DEBUG_MODE = True

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 6e-5
LAST_MODEL = None        # path of model saved last time

fp16=dict(          
    mode=AMP_TYPE.TORCH
)
```

> 在第一次运行后，将LAST_MODEL改为`model/test.pth`



- [ ] 测试1
- [ ] 测试2
- [ ] 测试3
- [ ] 测试4



请在测试全部结束后进行大规模训练。