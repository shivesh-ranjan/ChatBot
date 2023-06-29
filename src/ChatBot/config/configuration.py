from ChatBot.constants import *
from ChatBot.utils.common import read_yaml, create_directories
from ChatBot.entity import DataIngestionConfig, DataFormatConfig, DataLoadConfig, DataPrepareConfig, TrainConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
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
        return data_format_config
    
    def get_data_load_config(self) -> DataLoadConfig:
        config = self.config.data_load
        params = self.params.data_load
        create_directories([config.save_dir])
        data_load_config = DataLoadConfig(
            save_dir=config.save_dir, 
            corpus_name=config.corpus_name, 
            corpus=config.corpus, 
            datafile=config.datafile,
            max_length=params.max_length,
            min_count=params.min_count
        )
        return data_load_config
    
    def get_data_prepare_config(self) -> DataPrepareConfig:
        params = self.params.data_prepare
        data_prepare_config = DataPrepareConfig(
            small_batch_size=params.small_batch_size
        )
        return data_prepare_config
    
    def get_train_config(self) -> TrainConfig:
        config = self.config.training 
        params = self.params.training
        train_config = TrainConfig(
            save_dir=config.save_dir,
            model_name=config.model_name,
            corpus_name=config.corpus_name,
            loadFilename=config.loadFilename,
            hidden_size=params.hidden_size, 
            attn_model=params.attn_model, 
            encoder_n_layers=params.encoder_n_layers, 
            dropout=params.dropout, 
            decoder_n_layers=params.decoder_n_layers, 
            clip=params.clip, 
            teacher_forcing_ratio=params.teacher_forcing_ratio, 
            learning_rate=params.learning_rate, 
            decoder_learning_ratio=params.decoder_learning_ratio, 
            n_iteration=params.n_iteration, 
            print_every=params.print_every, 
            save_every=params.save_every,
            batch_size=params.batch_size,
            checkpoint_iter=params.checkpoint_iter
        )
        return train_config