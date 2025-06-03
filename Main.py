from human_vs_ai_dataset_loader import HumanVsAiDatasetLoader
from human_ai_generated_text import HumanAiGeneratedTextLoader
from datasets import concatenate_datasets, Dataset
import util
from ai_human_generated_text_loader import AIHumanGeneratedTextLoader
from gutenberg_human_generated_text import GutenbergHumanGeneratedTextLoader
from unique_ai_human_loader import UniqueAIHumanLoader

def create_train_dataset():
    print("\n=== Loading Human vs AI Dataset ===")
    human_vs_ai_dataset_loader = HumanVsAiDatasetLoader()
    processed_humanvsAIDataset = human_vs_ai_dataset_loader.get_preprocessed_dataset()
    print(f"Human vs AI Dataset Size: {processed_humanvsAIDataset.num_rows} rows")
    print("Sample of Human vs AI Dataset:")
    print(processed_humanvsAIDataset[:2])
    
    print("\n=== Loading Human AI Generated Text Dataset ===")
    human_ai_generated_text_loader = HumanAiGeneratedTextLoader()
    dsSize = human_vs_ai_dataset_loader.get_dataset().num_rows
    print(f"Target size for second dataset: {dsSize}")
    processed_dataset = human_ai_generated_text_loader.get_preprocessed_dataset(size=dsSize)
    print(f"Human AI Generated Text Dataset Size: {processed_dataset.num_rows} rows")
    print("Sample of Human AI Generated Text Dataset:")
    print(processed_dataset[:2])

    print("\n=== Combining Datasets ===")
    combined_dataset = concatenate_datasets([processed_humanvsAIDataset, processed_dataset])
    print(f"Combined Dataset Size: {combined_dataset.num_rows} rows")
    print("Sample of Combined Dataset:")
    print(combined_dataset[:2])
    print("\nDataset Features:", combined_dataset.features)

    print("\n=== Saving Combined Dataset ===")
    combined_dataset.to_csv("data/combined_dataset.csv")
    print("Dataset saved successfully to data/combined_dataset.csv")

def create_test_and_validation_dataset():
    validation_dataset_loader = HumanAiGeneratedTextLoader()
    validation_dataset_loader.load_dataset()
    test_and_validation_datasets = validation_dataset_loader.use_last_percent(0.01)
    transformed_test_and_validation_datasets = validation_dataset_loader.transform_dataset(test_and_validation_datasets)
    print(transformed_test_and_validation_datasets)
    half_percent = 0.5
    validation_dataset =  util.get_dataset_percent(transformed_test_and_validation_datasets, half_percent)
    validation_dataset = Dataset.from_dict(validation_dataset)
    print(validation_dataset)
    test_dataset =  util.get_dataset_percent_last(transformed_test_and_validation_datasets, half_percent)
    test_dataset = Dataset.from_dict(test_dataset)
    print(test_dataset)

    print("\n=== Saving Validation Dataset ===")
    validation_dataset.to_csv("data/validation_dataset.csv")
    print("Validation dataset saved successfully to data/validation_dataset.csv")

    print("\n=== Saving Test Dataset ===")
    test_dataset.to_csv("data/test_dataset.csv")
    print("Test dataset saved successfully to data/test_dataset.csv")

def create_test_dataset():
    print("\n=== Loading AI Human Generated Text Dataset ===")
    ai_human_loader = AIHumanGeneratedTextLoader()
    ai_human_dataset = ai_human_loader.get_preprocessed_dataset()
    print(f"AI Human Dataset Size: {len(ai_human_dataset['text'])} rows")

    print("\n=== Loading Gutenberg Dataset ===") 
    gutenberg_loader = GutenbergHumanGeneratedTextLoader()
    gutenberg_dataset = gutenberg_loader.get_preprocessed_dataset()
    gutenberg_dataset = util.get_dataset_percent(gutenberg_dataset, 15000/gutenberg_dataset.num_rows)
    print(f"Gutenberg Dataset Size: {len(gutenberg_dataset['text'])} rows")

    print("\n=== Loading Unique AI Human Dataset ===")
    unique_loader = UniqueAIHumanLoader()
    unique_dataset = unique_loader.get_preprocessed_dataset(size=15000)
    print(f"Unique AI Dataset Size: {unique_dataset.num_rows} rows")

    print("\n=== Combining Datasets ===")
    combined_dataset = concatenate_datasets([
        Dataset.from_dict(ai_human_dataset),
        Dataset.from_dict(gutenberg_dataset),
        unique_dataset
    ])
    print(f"Combined Dataset Size: {combined_dataset.num_rows} rows")
    print("Sample of Combined Dataset:")
    print(combined_dataset[:2])
    print("\nDataset Features:", combined_dataset.features)

    print("\n=== Saving Combined Dataset ===")
    combined_dataset.to_csv("data/test_15k_dataset.csv")
    print("Dataset saved successfully to data/test_15k_dataset.csv")

if __name__ == "__main__":
    create_test_dataset()