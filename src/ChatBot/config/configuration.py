from ChatBot.constants import *
from ChatBot.utils.common import read_yaml, create_directories
from ChatBot.entity import DataIngestionConfig, DataFormatConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = Path('config/config.yaml'),
        params_filepath = Path('params.yaml')):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config
    
    def get_data_format_confg(self) -> DataFormatConfig:
        config = self.config.data_format
        data_format_config = DataFormatConfig(
            datafile=config.datafile, 
            utterances=config.utterances
        )