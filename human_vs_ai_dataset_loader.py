from dataset_loader import DatasetLoader
import util

class HumanVsAiDatasetLoader(DatasetLoader):
    def __init__(self):
        print("\n=== Initializing HumanVsAiDatasetLoader ===")
        super().__init__('shahxeebhassan/human_vs_ai_sentences', 'train')
        print(f"Dataset source: shahxeebhassan/human_vs_ai_sentences")

    def preprocess_dataset(self, row):
        processed_row = {
            'text': self.getRowText(row['text']),
            'label': self.getRowLabel(row['label'])
        }
        return processed_row
    
    def get_preprocessed_dataset(self):
        print("\n=== Loading Human vs AI Dataset ===")
        self.load_dataset()
        dataset = self.get_dataset()
        print(f"Raw dataset size: {dataset.num_rows} rows")
        print("Dataset features:", dataset.features)
        print("\n=== Preprocessing Dataset ===")
        print("Applying preprocessing with 4 processes...")
        processed_dataset = dataset.map(self.preprocess_dataset, num_proc=4)
        print(f"Processed dataset size: {processed_dataset.num_rows} rows")
        print("Sample processed row:", processed_dataset[0])
        return processed_dataset