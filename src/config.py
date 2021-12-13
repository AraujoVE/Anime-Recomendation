from dataclasses import dataclass
from classification_strategy import ClassificationStrategy

@dataclass
class DatasetConfig:
    anime_dataset_path: str
    anime_synopsis_path: str
    pass

@dataclass
class Config:
    datasetConfig: DatasetConfig
    classificationStrategy: ClassificationStrategy

