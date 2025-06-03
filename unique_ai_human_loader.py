from dataset_loader import DatasetLoader
from datasets import Dataset, Features, Value, ClassLabel
import util

class UniqueAIHumanLoader(DatasetLoader):
    def __init__(self):
        print("\n=== Initializing UniqueAIHumanLoader ===")
        super().__init__(
            dataset_name="humancert/unique_ai_human",
            split_name="train"
        )
        print(f"Dataset source: humancert/unique_ai_human")
        
    def preprocess_dataset(self, batch):
        print(f"Processing batch with {len(batch['human_text'])} pairs of texts")
        texts = []
        labels = []
        
        for human_text, ai_text in zip(batch["human_text"], batch["ai_text"]):
            # Add human text
            texts.append(self.getRowText(human_text))
            labels.append(util.HUMAN_LABEL)
            
            # Add AI text
            texts.append(self.getRowText(ai_text))
            labels.append(util.AI_LABEL)
        
        processed_batch = {
            "text": texts,
            "label": labels
        }
        print(f"Processed batch size: {len(texts)} examples")
        return processed_batch

    def get_preprocessed_dataset(self, size):
        print(f"\n=== Loading Unique AI Human Dataset ===")
        print(f"Requested dataset size: {size}")
        self.load_dataset(size=size)
        dataset = self.get_dataset()
        print(f"Raw dataset size: {dataset.num_rows} rows")
        print("Dataset features:", dataset.features)
        print("\n=== Starting Dataset Transformation ===")
        return self.transform_dataset(dataset)
    
    def transform_dataset(self, dataset):
        print("Transforming dataset with parallel processing (4 processes)")
        print("Original columns:", dataset.column_names)
        transformed_dataset = dataset.map(
            self.preprocess_dataset,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=4
        )
        transformed_dataset = self.cast_features(transformed_dataset)
        print(f"Transformed dataset size: {transformed_dataset.num_rows} rows")
        print("New columns:", transformed_dataset.column_names)
        print("Sample transformed row:", transformed_dataset[0])
        return transformed_dataset 