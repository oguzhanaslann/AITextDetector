from dataset_loader import DatasetLoader
from util import HUMAN_LABEL


class GutenbergHumanGeneratedTextLoader(DatasetLoader):
    def __init__(self):
        print("\n=== Initializing GutenbergHumanGeneratedTextLoader ===")
        super().__init__("AlekseyKorshuk/ai-detection-gutenberg-human", "train")
        print(f"Dataset source: AlekseyKorshuk/ai-detection-gutenberg-human")

    def preprocess_dataset(self, row):
        processed_row = {"text": self.getRowText(row["human"]), "label": HUMAN_LABEL}
        return processed_row

    def get_preprocessed_dataset(self):
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
        )
        processed_dataset = self.cast_features(processed_dataset)
        print(f"Processed dataset size: {processed_dataset.num_rows} rows")
        print("Sample processed row:", processed_dataset[0])
        return processed_dataset