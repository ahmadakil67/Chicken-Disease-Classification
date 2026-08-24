from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

from dotenv import load_dotenv


load_dotenv()


STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            # Load configuration
            config = ConfigurationManager()

            # Get Data Ingestion configuration
            data_ingestion_config = (
                config.get_data_ingestion_config()
            )

            # Create Data Ingestion component
            data_ingestion = DataIngestion(
                config=data_ingestion_config
            )

            # Download data.zip from Azure Blob Storage
            data_ingestion.download_file()

            # Extract downloaded ZIP
            data_ingestion.extract_zip_file()

        except Exception as e:
            logger.exception(e)
            raise e