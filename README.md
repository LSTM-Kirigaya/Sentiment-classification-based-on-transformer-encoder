# Sentiment-classification-based-on-transformer-encoder

基于`Bert`进行的情感分类，使用`colossalai`框架进行包装和优化

## 前置工作

数据安放在了`data/simplifyweibo_4_moods.csv`, 使用中文预训练`Bert`自带的词表进行嵌入。

### 安装相关的库
主要用到的库为`torch`, `colossalai`和 `transformers`，直接pip3 install 一下，需要注意一下，`co`

### 处理数据


## 训练

### 准备训练
训练配置放在`config.py`中，以下是示例：

```python
from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 5  # for debug

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

fp16=dict(          
    mode=AMP_TYPE.TORCH
)
```
它会被注册到colossal的全局字典中，直接修改即可。混合精度配置根据选取的框架而定，本次使用使用

```bash
$torchrun --nproc_per_node 1 --master_addr localhost --master_port 8008 train_bert.py
```