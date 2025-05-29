import csv
import json
import argparse

def convert_csv_to_jsonl(input_csv, output_jsonl):
    try:
        # Open the CSV file
        with open(input_csv, 'r', encoding='utf-8') as csv_file:
            # Read CSV file
            csv_reader = csv.DictReader(csv_file)
            
            # Open the JSONL file for writing
            with open(output_jsonl, 'w', encoding='utf-8') as jsonl_file:
                # Convert each row to JSON and write to the output file
                for row in csv_reader:
                    # Convert the row to JSON and write it as a line
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + '\n')
                    
        print(f"Successfully converted {input_csv} to {output_jsonl}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert CSV file to JSONL format')
    parser.add_argument('input_csv', help='Input CSV file path')
    parser.add_argument('output_jsonl', help='Output JSONL file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert the file
    convert_csv_to_jsonl(args.input_csv, args.output_jsonl)

if __name__ == "__main__":
    main() 