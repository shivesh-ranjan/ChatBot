from ChatBot.pipeline.stage01_dataIngestion import DataIngestionTrainingPipeline
from ChatBot.pipeline.stage02_dataFormat import DataFormatTrainingPipeline
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
'''
STAGE_NAME = 'Data Format Stage'
try:
    logger.info(f">>>> Stage {STAGE_NAME} started <<<<")
    data_format = DataFormatTrainingPipeline()
    data_format.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed <<<<\n\nX=============================X")
except Exception as e:
    logger.exception(str(e))
    raise e