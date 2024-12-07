# model
OB_RADIUS = 10000  # observe radius, neighborhood radius
OB_HORIZON = 8  # number of observation frames
PRED_HORIZON = 12  # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = ["PLAYER"]
RNN_HIDDEN_DIM = 512

# training
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
EPOCHS = 2000  # total number of epochs for training
EPOCH_BATCHES = 100  # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 1200  # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20  # best of N samples
FPC_SEARCH_RANGE = range(40, 50)  # FPC sampling rate

# evaluation
WORLD_SCALE = 1
