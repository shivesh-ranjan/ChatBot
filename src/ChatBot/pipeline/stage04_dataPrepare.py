from ChatBot.components.dataPrepare import *
from ChatBot.config.configuration import ConfigurationManager
import random

config = ConfigurationManager()
config = config.get_data_prepare_config()

class DataPrepareTrainingPipeline:
    def __init__(self, voc, pairs):
        self.voc=voc
        self.pairs=pairs
    def main(self):
        # Example for validation
        small_batch_size = config.small_batch_size
        input_variable, lengths, target_variable, mask, max_target_len = batch2TrainData(self.voc, [random.choice(self.pairs) for _ in range(small_batch_size)])
        print("input_variable:", input_variable)
        print("lengths:", lengths)
        print("target_variable:", target_variable)
        print("mask:", mask)
        print("max_target_len:", max_target_len)
        return input_variable, lengths, target_variable, mask, max_target_len