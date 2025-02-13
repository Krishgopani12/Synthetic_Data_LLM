
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)

try:
    from config import OPENAI_API_KEY
except ImportError:
    raise ImportError(
        "Could not find config.py. Please create this file with your OPENAI_API_KEY."
    )

def create_examples(df: pd.DataFrame, num_examples: int = 3, random_seed: int = None) -> tuple[List[Dict], pd.DataFrame]:
    """
    Create example dictionaries from the sample DataFrame with random selection
    
    Args:
        df: Input DataFrame containing sample data
        num_examples: Number of examples to select from the sample data
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple containing:
        - List of dictionaries containing the selected examples
        - DataFrame of the selected sample data
    """
    # Ensure we don't try to select more examples than available
    num_examples = min(num_examples, len(df))
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Randomly sample rows
    sample_df = df.sample(n=num_examples)
    examples = []
    
    for _, row in sample_df.iterrows():
        example_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
        examples.append({"example": example_str})
    
    return examples, sample_df

def generate_synthetic_data(
    sample_csv_path: str,
    num_rows: int,
    num_examples: int = 3,
    temperature: float = 1.0,
    subject: str = "data",
    extra: str = "make the data realistic and varied",
    random_seed: int = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data based on a sample CSV file
    """
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Read sample data
    df = pd.read_csv(sample_csv_path)
    
    # Create examples from sample data and get the sample DataFrame
    examples, sample_df = create_examples(df, num_examples=num_examples, random_seed=random_seed)
    
    # Create prompt template
    example_prompt = PromptTemplate(input_variables=["example"], template="{example}")
    prompt_template = FewShotPromptTemplate(
        prefix=SYNTHETIC_FEW_SHOT_PREFIX,
        examples=examples,
        suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
        input_variables=["subject", "extra"],
        example_prompt=example_prompt,
    )
    
    # Initialize ChatOpenAI
    llm = ChatOpenAI(temperature=temperature)
    
    # Generate synthetic data
    synthetic_data = []
    columns = df.columns.tolist()
    
    print(f"\nGenerating {num_rows} synthetic data rows:")
    for i in range(num_rows):
        # Format the prompt with current parameters
        prompt = prompt_template.format(subject=subject, extra=extra)
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        try:
            # Parse the response into a dictionary
            row_data = {}
            response_text = response.content.strip()
            
            # Split the response into key-value pairs
            pairs = response_text.split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    if key in columns:
                        row_data[key] = value
            
            synthetic_data.append(row_data)
            print(f"\rProgress: {i+1}/{num_rows} rows generated", end="", flush=True)
            
        except Exception as e:
            print(f"\nError processing row {i+1}: {e}")
            continue
    
    print("\nGeneration complete!")
    return pd.DataFrame(synthetic_data), sample_df

if __name__ == "__main__":
    # Get user input
    sample_csv_path = "data2/sample_data.csv"
    num_rows = int(input("Enter the number of synthetic rows to generate: "))
    num_examples = int(input("Enter the number of examples to use from sample data: "))
    
    # Generate synthetic data
    synthetic_df, sample_df = generate_synthetic_data(
        sample_csv_path=sample_csv_path,
        num_rows=num_rows,
        num_examples=num_examples
    )
    
    # Save the synthetic data to a new CSV file
    output_path = "data2/synthetic_data.csv"
    synthetic_df.to_csv(output_path, index=False)
    
    # Save the sample data used for generation
    sample_output_path = "data2/sample_data_used.csv"
    sample_df.to_csv(sample_output_path, index=False)
    
    print(f"\nGenerated {num_rows} rows of synthetic data using {num_examples} examples")
    print(f"Synthetic data saved to: {output_path}")
    print(f"Sample data used saved to: {sample_output_path}") 
