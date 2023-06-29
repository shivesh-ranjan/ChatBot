from dataclasses import dataclass
from pathlib import Path

# Not a class but a dataclass
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataFormatConfig:
    datafile: str
    utterances: str 

@dataclass(frozen=True)
class DataLoadConfig:
    save_dir: Path 
    corpus_name: str 
    corpus: Path
    datafile: str 
    max_length: int
    min_count: int

@dataclass(frozen=True)
class DataPrepareConfig:
    small_batch_size: int

@dataclass(frozen=True)
class TrainConfig:
    save_dir: Path 
    model_name: str 
    corpus_name: str 
    loadFilename: str
    hidden_size: int # Embedding Layer
    attn_model: str 
    encoder_n_layers: int
    dropout: float
    decoder_n_layers: int
    # Configure training/optimization
    clip: int
    teacher_forcing_ratio: float
    learning_rate: float
    decoder_learning_ratio: float
    n_iteration: int
    print_every: int
    save_every: int
    batch_size: int
    checkpoint_iter: int