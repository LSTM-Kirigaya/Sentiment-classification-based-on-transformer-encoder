from colossalai.amp import AMP_TYPE

DEBUG_BATCH_NUM = 10  # for debug
DEBUG_MODE = True

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 6e-5
LAST_MODEL = "model/2022-04-29 23-04-10.pth"        # path of model saved last time

fp16=dict(          
    mode=AMP_TYPE.TORCH
)