import sys
import time
from ConfigParser import ConfigParser

class Params():
    def __init__(self, config_file_path="../config/default.ini"):
        cf = ConfigParser()
        cf.read(config_file_path)

        # Info
        self.EXPERIMENT = cf.get('info','experiment')
        self.NAME = cf.get('info','name')
        self.MODEL_ID = str(int(time.time()))+" "+cf.get('info','name')

        # Dataset
        self.PIXELS = cf.get('dataset','pixels')
        self.CHANNELS = cf.get('dataset','channels')
        self.N_CLASSES = cf.get('dataset','n_classes')

        self.SUBSET = cf.get('dataset','subset')
        self.FILENAMES_TRAIN = cf.get('dataset','filenames_train')
        self.FILENAMES_VALIDATION = cf.get('dataset','filenames_validation')

        # Network
        self.ARCHITECTURE = cf.get('network', 'architecture')

        # Network - U-net specific
        self.INPUT_SIZE = cf.get('network', 'input_size')
        self.DEPTH = cf.get('network', 'depth')
        self.BRANCHING_FACTOR = cf.get('network', 'branching_factor')

        # Updates
        self.OPTIMIZATION = cf.get('updates', 'optimization')
        self.LEARNING_RATE = cf.get('updates', 'learning_rate')
        self.MOMENTUM = cf.get('updates', 'momentum')
        self.L2_LAMBDA = cf.get('updates', 'l2_lambda')

        self.BATCH_SIZE_TRAIN = cf.get('updates', 'batch_size_train')
        self.BATCH_SIZE_VALIDATION = cf.get('updates', 'batch_size_validation')

        # Normalization
        self.ZERO_CENTER = cf.get('normalization', 'zero_center')
        self.MEAN_PIXEL = cf.get('normalization', 'mean_pixel')

params = Params()
