from ChatBot.pipeline.stage01_dataIngestion import DataIngestionTrainingPipeline
from ChatBot.pipeline.stage02_dataFormat import DataFormatTrainingPipeline
from ChatBot.pipeline.stage03_dataLoad import DataLoadTrainingPipeline
from ChatBot.pipeline.stage04_dataPrepare import DataPrepareTrainingPipeline
from ChatBot.pipeline.stage05_training import TrainingPipeline
from ChatBot.pipeline.stage06_evaluation import Evaluate
from ChatBot.logging import logger
'''
STAGE_NAME = 'Data Ingestion Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e

STAGE_NAME = 'Data Format Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    data_format = DataFormatTrainingPipeline()
    data_format.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e
'''
STAGE_NAME = 'Data Load Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    data_load = DataLoadTrainingPipeline()
    voc, pairs = data_load.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e

STAGE_NAME = 'Data Prepare Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    data_prepare = DataPrepareTrainingPipeline(voc, pairs)
    input_variable, lengths, target_variable, mask, max_target_len = data_prepare.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e
'''
STAGE_NAME = 'Training Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    train = TrainingPipeline(voc, pairs)
    train.initialize()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e
'''
STAGE_NAME = 'Evaluation Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    evaluate = Evaluate()
    encoder, decoder, searcher = evaluate.main(voc)
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e

from ChatBot.components.evaluation import evaluateInput
logger.info(f">>>> Prompt started <<<<")
evaluateInput(encoder, decoder, searcher, voc)