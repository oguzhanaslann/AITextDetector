import pandas as pd

def calculate_average_text_length():
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv('data/combined_dataset.csv')
    
    # Calculate text length for each row
    print("Calculating text lengths...")
    text_lengths = df['text'].str.len()
    
    # Calculate average length
    average_length = text_lengths.mean()
    
    # Calculate additional statistics
    min_length = text_lengths.min()
    max_length = text_lengths.max()
    median_length = text_lengths.median()
    
    # Print results
    print("\nText Length Statistics:")
    print(f"Average length: {average_length:.2f} characters")
    print(f"Minimum length: {min_length:.0f} characters")
    print(f"Maximum length: {max_length:.0f} characters")
    print(f"Median length: {median_length:.0f} characters")

if __name__ == "__main__":
    try:
        calculate_average_text_length()
    except Exception as e:
        print(f"An error occurred: {str(e)}") 