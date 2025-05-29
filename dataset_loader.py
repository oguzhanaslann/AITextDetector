from datasets import load_dataset
import util
from abc import ABC, abstractmethod
from datasets import Dataset

class DatasetLoader(ABC):
    def __init__(
            self,
            dataset_name,
            split_name = 'train',
        ):
        self.dataset_name = dataset_name
        self.split_name = split_name
        
    def load_dataset(self, size=None):
        if size is not None:
            ds = load_dataset(self.dataset_name, split=self.split_name, streaming=True)
            sample = []
            for i, example in enumerate(ds):
                if i >= size:
                    break
                sample.append(example)
            self._ds_master = Dataset.from_list(sample)
        else:
            self._ds_master = load_dataset(self.dataset_name, split=self.split_name)
        self._ds = self._ds_master
        return
    
    def use_percent(self, percent):
        self._ds = util.get_dataset_percent(self._ds_master, percent)
        return Dataset.from_dict(self._ds)
    
    def use_last_percent(self, percent):
        self._ds = util.get_dataset_percent_last(self._ds_master, percent)
        return Dataset.from_dict(self._ds)
    
    def get_dataset(self):
        return self._ds
    
    @abstractmethod
    def preprocess_dataset(self, row):
        pass 

    @abstractmethod
    def get_preprocessed_dataset(self):
        pass

    def getRowText(self, text):
        return util.preprocess_text(text)
    
    def getRowLabel(self, label):
        return util.getLabelText(label)