# model
OB_RADIUS = 2  # observe radius, neighborhood radius
OB_HORIZON = 8  # number of observation frames
PRED_HORIZON = 12  # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
RNN_HIDDEN_DIM = 256

# training
LEARNING_RATE = 7e-4
BATCH_SIZE = 512
EPOCHS = 800  # total number of epochs for training
EPOCH_BATCHES = 100  # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 1  # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20  # best of N samples
FPC_SEARCH_RANGE = range(30, 50)  # FPC sampling rate

# evaluation
WORLD_SCALE = 1

# 自己添加
ROTATION = True
USE_AUGMENTATION = False
DIST_THRESHOLD = 2
DATASET_NAME = "hotel"
N_EMBD = 128
DROPOUT = 0.1
INT_NUM_LAYERS_LIST = [3, 2]
N_HEAD = 8
FORWARD_EXPANSION = 2
BLOCK_SIZE = 128
VOCAB_SIZE = 2
GOAL_NUM = 20
LAMBDA_DES = 100.0  # 目的地损失
LAMBDA_J = 0.0  # 多样性损失
LAMBDA_RECON = 1.0
