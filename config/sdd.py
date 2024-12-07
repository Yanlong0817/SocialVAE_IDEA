# model
OB_RADIUS = 5  # observe radius, neighborhood radius
OB_HORIZON = 8  # number of observation frames
PRED_HORIZON = 12  # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []

# training
LEARNING_RATE = 0.0015
LEARNING_RATE_MIN = 1e-5
BATCH_SIZE = 512
EPOCHS = 1000  # total number of epochs for training
EPOCH_BATCHES = 100  # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 1  # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20  # best of N samples
FPC_SEARCH_RANGE = range(40, 50)  # FPC sampling rate

# evaluation
WORLD_SCALE = 1  # in unit of meters

# 自己添加
ROTATION = True
USE_AUGMENTATION = False
DIST_THRESHOLD = 5
