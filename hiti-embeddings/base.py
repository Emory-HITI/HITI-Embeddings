from abc import ABC, abstractmethod
from typing import Optional


class FoundationBase(ABC):
    # the name of the model should be defined
    model_name: str = "FoundationBase"

    # the input size for the model should be defined
    input_size: tuple = (1024, 768)

    def __init__(self):
        self.model: Optional[any] = None

    @abstractmethod
    def preprocess(self):
        """This function should take an input image and preprocess it as needed for the foundation model"""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """This function should load the foundation model and save it to self.model"""
        pass

    @abstractmethod
    def extract_embeddings(self):
        """This function should be run on a dataset of images and return a dataset of embeddings"""
        pass
