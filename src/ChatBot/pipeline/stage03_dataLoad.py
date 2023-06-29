from ChatBot.config.configuration import ConfigurationManager
from ChatBot.components.dataLoad import *
from ChatBot.logging import logger

config = ConfigurationManager()
config = config.get_data_load_config()

class DataLoadTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        # Load/Assemble voc and pairs
        voc, pairs = loadPrepareData(config.corpus, config.corpus_name, config.datafile, config.save_dir)
        # Print some pairs to validate
        print("\npairs:")
        for pair in pairs[:10]:
            print(pair)

        # Trim voc and pairs
        pairs = trimRareWords(voc, pairs, config.min_count)
        return voc, pairs