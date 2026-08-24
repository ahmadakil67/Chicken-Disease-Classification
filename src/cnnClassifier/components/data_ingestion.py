import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


load_dotenv()


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        """
        Download data.zip from Azure Blob Storage
        into artifacts/data_ingestion/data.zip
        """

        # Create destination directory if it does not exist
        Path(self.config.local_data_file).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Skip download if file already exists
        if os.path.exists(self.config.local_data_file):
            logger.info(
                f"File already exists: {self.config.local_data_file}"
            )
            return

        connection_string = os.getenv(
            "AZURE_STORAGE_CONNECTION_STRING"
        )

        if not connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING "
                "not found in .env"
            )

        try:
            logger.info(
                f"Downloading {self.config.blob_name} "
                f"from Azure container "
                f"{self.config.container_name}"
            )

            blob_service_client = (
                BlobServiceClient.from_connection_string(
                    connection_string
                )
            )

            blob_client = blob_service_client.get_blob_client(
                container=self.config.container_name,
                blob=self.config.blob_name
            )

            with open(
                self.config.local_data_file,
                "wb"
            ) as file:

                download_stream = blob_client.download_blob(
                    max_concurrency=4
                )

                download_stream.readinto(file)

            logger.info("Azure Blob download completed successfully.")

        except AzureError as e:
            logger.error(
                f"Error downloading file from Azure Blob Storage: {e}"
            )
            raise


    def extract_zip_file(self):
        """
        Extract zip file into the configured unzip directory.
        """

        unzip_path = self.config.unzip_dir

        os.makedirs(
            unzip_path,
            exist_ok=True
        )

        logger.info(
            f"Extracting {self.config.local_data_file} "
            f"to {unzip_path}"
        )

        with zipfile.ZipFile(
            self.config.local_data_file,
            "r"
        ) as zip_ref:

            zip_ref.extractall(unzip_path)

        logger.info("ZIP extraction completed successfully.")