#!/usr/bin/env python3

import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Filter out rows with X or O in EmoClass column")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV/TSV file with emotion labels")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output filtered file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Read the file with the specified separator
        df = pd.read_csv(args.input)
        
        # Print initial data stats
        total_rows = len(df)
        print(f"Total rows in input file: {total_rows}")
        
        # Check if EmoClass column exists
        if 'EmoClass' not in df.columns:
            print("Error: 'EmoClass' column not found in the input file")
            return
        
        # Count rows with X or O
        x_count = len(df[df['EmoClass'] == 'X'])
        o_count = len(df[df['EmoClass'] == 'O'])
        print(f"Found {x_count} rows with 'X' and {o_count} rows with 'O' in EmoClass")
        
        # Filter out rows where EmoClass is X or O
        filtered_df = df[~df['EmoClass'].isin(['X', 'O'])]
        
        # Save the filtered dataframe
        filtered_df.to_csv(args.output, index=False)
        
        # Print results
        filtered_count = len(filtered_df)
        print(f"Filtered out {total_rows - filtered_count} rows")
        print(f"Saved {filtered_count} rows to {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()