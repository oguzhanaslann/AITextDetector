import json
from typing import Any, Dict, List, Union

class JSONValidationError(Exception):
    """Custom exception for JSON validation errors."""
    pass

class JSONValidator:
    def __init__(self):
        self.errors = []

    def validate_json_string(self, json_string: str) -> tuple[bool, List[str]]:
        """
        Validate a JSON string and return whether it's valid and any error messages.
        
        Args:
            json_string (str): The JSON string to validate
            
        Returns:
            tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        try:
            # Try to parse the JSON string
            parsed_json = json.loads(json_string)
            return self.validate_json_data(parsed_json)
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON format: {str(e)}"]

    def validate_json_file(self, file_path: str) -> tuple[bool, List[str]]:
        """
        Validate a JSON file and return whether it's valid and any error messages.
        
        Args:
            file_path (str): Path to the JSON file
            
        Returns:
            tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_string = file.read()
                return self.validate_json_string(json_string)
        except FileNotFoundError:
            return False, [f"File not found: {file_path}"]
        except Exception as e:
            return False, [f"Error reading file: {str(e)}"]

    def validate_json_data(self, data: Any) -> tuple[bool, List[str]]:
        """
        Validate parsed JSON data and return whether it's valid and any error messages.
        
        Args:
            data: The parsed JSON data to validate
            
        Returns:
            tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        self.errors = []
        self._validate_data_structure(data)
        return len(self.errors) == 0, self.errors

    def _validate_data_structure(self, data: Any, path: str = "root") -> None:
        """
        Recursively validate the structure of JSON data.
        
        Args:
            data: The data to validate
            path: The current path in the JSON structure (for error reporting)
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(key, str):
                    self.errors.append(f"Invalid key type at {path}: keys must be strings")
                new_path = f"{path}.{key}" if path != "root" else key
                self._validate_data_structure(value, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                self._validate_data_structure(item, new_path)
        
        elif not isinstance(data, (str, int, float, bool, type(None))):
            self.errors.append(f"Invalid value type at {path}: {type(data)}")

def main():
    validator = JSONValidator()
    
    try:
        with open('data/combined_dataset.jsonl', 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                is_valid, errors = validator.validate_json_string(line)
                if not is_valid:
                    print(f"\nError in line {line_number}:")
                    print(f"Line content: {line}")
                    print("Validation errors:", errors)
                    break  # Stop processing on first error
                
                # Optional: Print progress every 1000 lines
                if line_number % 1000 == 0:
                    print(f"Processed {line_number} lines...")
                    
    except FileNotFoundError:
        print("Error: combined_dataset.jsonl file not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    #main()
    invalid_json = """
        {
            "text": "\"Another reason why all students should have to participate in at least one extracurricular activity is because it develops stronger social skills.\"",
            "label": "HUMAN"
    }
    """
    validator = JSONValidator()
    is_valid, errors = validator.validate_json_string(invalid_json)
    print(f"Valid: {is_valid}")
    print("Errors:", errors)