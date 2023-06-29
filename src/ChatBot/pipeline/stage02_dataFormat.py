from ChatBot.config.configuration import ConfigurationManager
from ChatBot.components.dataFormat import DataFormat
from ChatBot.logging import logger
import codecs
import csv

class DataFormatTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_format_config = config.get_data_format_confg()
        data_format = DataFormat(config=data_format_config)

        delimiter = '\t'
        # Unescape the delimiter
        delimiter = str(codecs.decode(delimiter, "unicode_escape"))
        # Initialize lines dict and conversations dict
        lines = {}
        conversations = {}
        # Load lines and conversations
        print("\nProcessing corpus into lines and conversations...")
        lines, conversations = data_format.loadLinesAndConversations(data_format_config.utterances)
        # Write new csv file
        print("\nWriting newly formatted file...")
        with open(data_format_config.datafile, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
            for pair in data_format.extractSentencePairs(conversations):
                writer.writerow(pair)