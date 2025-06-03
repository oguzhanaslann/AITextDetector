from dataset_loader import DatasetLoader
import util
from datasets import Features, Value, ClassLabel

class AIHumanGeneratedTextLoader(DatasetLoader):
    def __init__(self):
        print("\n=== Initializing AIHumanGeneratedTextLoader ===")
        super().__init__('Ateeqq/AI-and-Human-Generated-Text', 'train')
        print(f"Dataset source: Ateeqq/AI-and-Human-Generated-Text")

    def preprocess_dataset(self, batch):
        texts = []
        labels = []
        
        # Process each item in the batch
        for title, abstract, label in zip(batch['title'], batch['abstract'], batch['label']):
            # Combine title and abstract for more context
            full_text = f"Title: {title}\nAbstract: {abstract}"
            texts.append(self.getRowText(full_text))
            labels.append(self.getRowLabel(label))
        
        return {
            'text': texts,
            'label': labels
        }
    
    def get_preprocessed_dataset(self):
        print("\n=== Loading AI and Human Generated Text Dataset ===")
        self.load_dataset()
        dataset = self.get_dataset()
        print(f"Raw dataset size: {dataset.num_rows} rows")
        print("Dataset features:", dataset.features)
        print("\n=== Preprocessing Dataset ===")
        print("Applying preprocessing with 4 processes...")    
        processed_dataset = dataset.map(
            self.preprocess_dataset,
            remove_columns=dataset.column_names,
            num_proc=4,
            batched=True,
        )
        processed_dataset = self.cast_features(processed_dataset)
        print(f"Processed dataset size: {processed_dataset.num_rows} rows")
        print("Sample processed row:", processed_dataset[0])
        return processed_dataset

