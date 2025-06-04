from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'datasets>=2.15.0',
    'torch>=2.1.0',
    'transformers>=4.36.0',
    'numpy>=1.24.0,<2.0.0',
    'pandas>=2.0.0',
    'accelerate>=0.23.0',
    'peft>=0.6.0'
]

setup(
    name='ai_text_detection_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='AI Text Detection training application for Vertex AI'
) 