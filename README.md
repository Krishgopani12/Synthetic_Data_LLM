## Overview
This project provides a tool for generating synthetic data based on sample CSV files using OpenAI's language models. It's designed to create realistic and varied synthetic data while maintaining the structure and patterns of the original dataset.

## Features
- Generate synthetic data based on sample CSV files
- Configurable number of synthetic rows
- Adjustable number of example rows for context
- Temperature control for output variation
- Automatic type inference for data fields
- Progress tracking during generation
- Sample data preservation

## Prerequisites
- Python 3.7+
- OpenAI API key
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - langchain
  - langchain-openai
  - pydantic

## Setup
1. Clone this repository
2. Create a `config.py` file in the root directory with your OpenAI API key:

## Usage
1. Place your sample CSV file in the `data/data2` directory
2. Run the generator:

```bash:README.md
python synthetic_data_generator.py
```

## Key Functions

### `infer_field_types(df)`
- Automatically infers Pydantic field types from DataFrame columns
- Supports int, float, string, boolean, and datetime types

### `create_examples(df, num_examples, random_seed)`
- Creates example dictionaries from the sample DataFrame
- Randomly selects specified number of examples
- Supports reproducibility via random seed

### `generate_synthetic_data(sample_csv_path, num_rows, num_examples, temperature, subject, extra, random_seed)`
- Main function for generating synthetic data
- Parameters:
  - `sample_csv_path`: Path to the sample CSV file
  - `num_rows`: Number of synthetic rows to generate
  - `num_examples`: Number of examples to use from sample data
  - `temperature`: Controls randomness (0.0-2.0)
  - `subject`: Description of the data type
  - `extra`: Additional instructions for data generation
  - `random_seed`: Optional seed for reproducibility

## Output
The generator creates two files:
1. `synthetic_data.csv`: Contains the generated synthetic data
2. `sample_data_used.csv`: Contains the sample data used for generation

## Example
```python
synthetic_df, sample_df = generate_synthetic_data(
    sample_csv_path="data2/sample_data.csv",
    num_rows=10,
    num_examples=3,
    temperature=1.0,
    subject="customer data",
    extra="make the data realistic and varied"
)
```

## Notes
- The quality of synthetic data depends on the quality and representativeness of the sample data
- Higher temperature values (closer to 2.0) result in more creative but potentially less accurate data
- Lower temperature values (closer to 0.0) result in more conservative and consistent data

## License
MIT License

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.