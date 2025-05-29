#!/usr/bin/env python3

import pandas as pd
import os
import sys
from typing import Dict, List, Tuple


class CSVValidator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.validation_results: Dict[str, bool] = {}
        self.error_messages: List[str] = []

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Perform all validation checks on the CSV file.
        Returns a tuple of (is_valid, list_of_error_messages)
        """
        # Check if file exists
        if not self._check_file_exists():
            return False, self.error_messages

        # Check if file is empty
        if not self._check_file_not_empty():
            return False, self.error_messages

        try:
            # Try to read the CSV file
            df = pd.read_csv(self.file_path)
            
            # Perform various checks
            self._check_column_names(df)
            self._check_missing_values(df)
            self._check_data_consistency(df)
            
            # Return overall validation result
            is_valid = all(self.validation_results.values())
            return is_valid, self.error_messages

        except pd.errors.EmptyDataError:
            self.error_messages.append("The CSV file is empty")
            return False, self.error_messages
        except pd.errors.ParserError:
            self.error_messages.append("Failed to parse CSV file - invalid format")
            return False, self.error_messages
        except Exception as e:
            self.error_messages.append(f"Unexpected error: {str(e)}")
            return False, self.error_messages

    def _check_file_exists(self) -> bool:
        """Check if the file exists and is readable"""
        exists = os.path.isfile(self.file_path)
        self.validation_results['file_exists'] = exists
        if not exists:
            self.error_messages.append(f"File does not exist: {self.file_path}")
        return exists

    def _check_file_not_empty(self) -> bool:
        """Check if the file is not empty"""
        is_not_empty = os.path.getsize(self.file_path) > 0
        self.validation_results['file_not_empty'] = is_not_empty
        if not is_not_empty:
            self.error_messages.append("File is empty")
        return is_not_empty

    def _check_column_names(self, df: pd.DataFrame) -> None:
        """Check if column names are valid"""
        has_valid_columns = all(isinstance(col, str) and col.strip() for col in df.columns)
        self.validation_results['valid_columns'] = has_valid_columns
        if not has_valid_columns:
            self.error_messages.append("Invalid column names detected")

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Check for missing values in the dataset"""
        missing_counts = df.isnull().sum()
        has_missing = missing_counts.any()
        self.validation_results['no_missing_values'] = not has_missing
        if has_missing:
            for column, count in missing_counts[missing_counts > 0].items():
                self.error_messages.append(f"Column '{column}' has {count} missing values")

    def _check_data_consistency(self, df: pd.DataFrame) -> None:
        """Check for data consistency (e.g., same number of fields in each row)"""
        expected_columns = len(df.columns)
        is_consistent = True
        
        # Check if all rows have the same number of columns
        with open(self.file_path, 'r') as file:
            for i, line in enumerate(file, 1):
                if line.strip():  # Skip empty lines
                    fields = line.count(',') + 1
                    if fields != expected_columns:
                        is_consistent = False
                        self.error_messages.append(
                            f"Inconsistent number of fields in row {i}: "
                            f"expected {expected_columns}, found {fields}"
                        )
                        break

        self.validation_results['data_consistency'] = is_consistent


def main():
    if len(sys.argv) != 2:
        print("Usage: python csv_validator.py <path_to_csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    validator = CSVValidator(csv_file)
    is_valid, errors = validator.validate()

    print(f"\nValidation results for: {csv_file}")
    print("-" * 50)
    
    if is_valid:
        print("✅ CSV file is valid!")
    else:
        print("❌ CSV file is invalid!")
        print("\nErrors found:")
        for error in errors:
            print(f"  - {error}")

    print("\nDetailed validation results:")
    for check, result in validator.validation_results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")


if __name__ == "__main__":
    main() 