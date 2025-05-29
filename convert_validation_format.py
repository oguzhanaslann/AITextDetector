import json
import os
import util
def convert_line_to_format(input_data, include_system_prompt=False):
    """
    Convert input data to the specified format.
    """
    result = {}
    
    if include_system_prompt:
        result["systemInstruction"] = {
            "role": "system",
            "parts": [
                {
                    "text": util.get_system_prompt()
                }
            ]
        }
    
    result["contents"] = [
        {
            "role": "user", 
            "parts": [
                {"text": input_data["text"]}
            ]
        },
        {
            "role": "model",
            "parts": [
                {"text": input_data["label"]}
            ]
        }
    ]
        
    return result

def process_validation_dataset():
    input_file = "data/combined_dataset.jsonl"
    output_file = "data/combined_dataset_converted.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_number, line in enumerate(infile, 1):
                try:
                    # Parse the input JSON line
                    input_data = json.loads(line.strip())
                    
                    # Convert to new format
                    converted_data = convert_line_to_format(input_data)
                    
                    # Write to output file
                    json.dump(converted_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_number}: {e}")
                except KeyError as e:
                    print(f"Missing required key at line {line_number}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_number}: {e}")
        
        print(f"Conversion completed. Output saved to {output_file}")
    
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    process_validation_dataset() 