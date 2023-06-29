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