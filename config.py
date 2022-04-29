from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 5  # for debug

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

fp16=dict(          
    mode=AMP_TYPE.TORCH
)