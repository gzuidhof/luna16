import sys
import time
from ConfigParser import ConfigParser
import StringIO

class Params():
    def __init__(self, config_file_path="../config/default.ini"):
        cf = ConfigParser()
        cf.read(config_file_path)

        self.CONFIG = cf
        cf.set('info','config_file', config_file_path)
        cf.set('info','model_id', str(int(time.time()))+"_"+cf.get('info','name'))

        # Info
        self.EXPERIMENT = cf.get('info', 'experiment')
        self.NAME = cf.get('info', 'name')
        self.MODEL_ID = cf.get('info', 'model_id')

        # Dataset
        self.PIXELS = cf.getint('dataset','pixels')
        self.CHANNELS = cf.getint('dataset','channels')
        self.N_CLASSES = cf.getint('dataset','n_classes')

        self.SUBSET = None if cf.get('dataset','subset')=='None' else cf.getint('dataset','subset')

        self.FILENAMES_TRAIN = cf.get('dataset','filenames_train')
        self.FILENAMES_VALIDATION = cf.get('dataset','filenames_validation')

        # Network
        self.ARCHITECTURE = cf.get('network', 'architecture')

        # Network - U-net specific
        self.INPUT_SIZE = cf.getint('network', 'input_size')
        self.DEPTH = cf.getint('network', 'depth')
        self.BRANCHING_FACTOR = cf.getint('network', 'branching_factor')

        # Updates
        self.OPTIMIZATION = cf.get('updates', 'optimization')
        self.LEARNING_RATE = cf.getfloat('updates', 'learning_rate')
        self.MOMENTUM = cf.getfloat('updates', 'momentum')
        self.L2_LAMBDA = cf.getfloat('updates', 'l2_lambda')

        self.BATCH_SIZE_TRAIN = cf.getint('updates', 'batch_size_train')
        self.BATCH_SIZE_VALIDATION = cf.getint('updates', 'batch_size_validation')
        self.N_EPOCHS = cf.getint('updates', 'n_epochs')

        # Normalization
        self.ZERO_CENTER = cf.getboolean('normalization', 'zero_center')
        self.MEAN_PIXEL = cf.getfloat('normalization', 'mean_pixel')

    def to_string(self):
        output = StringIO.StringIO()
        self.CONFIG.write(output)
        val = output.getvalue()
        output.close()
        return val

    def write_to_file(self, filepath):
        with open(filepath, 'w') as f:
            self.CONFIG.write(f)

params = Params()
