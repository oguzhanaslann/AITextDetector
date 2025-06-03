# AITextDetector

AITextDetector is a machine learning tool that distinguishes between AI-generated and human-written text. Using advanced natural language processing and multiple datasets, it provides accurate content verification through sophisticated text analysis, making it essential for educators, publishers, and content moderators.

## Features

- Multiple dataset integration (Human vs AI, Gutenberg, Unique AI Human datasets)
- Advanced text preprocessing and tokenization
- Support for various text formats (CSV, JSON, JSONL)
- Watermark detection capabilities
- Comprehensive validation framework
- Training pipeline with Google Cloud Vertex AI support

## Project Structure

- `dataset_loader.py`: Base class for dataset loading functionality
- `Main.py`: Main entry point for dataset processing
- `watermark_detection.py`: Unicode watermark detection implementation
- `train.py` & `train_full.py`: Model training implementations
- `util.py`: Utility functions for text processing
- Various dataset loaders for different sources

## Setup

1. Clone the repository:
```bash
git clone https://github.com/oguzhanaslann/AITextDetector.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project provides various tools for text analysis and AI detection:

1. Dataset Processing:
```python
python Main.py
```

2. Training:
```python
python train.py
```

3. Validation:
```python
python convert_validation_format.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 