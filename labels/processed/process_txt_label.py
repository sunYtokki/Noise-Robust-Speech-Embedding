#!/usr/bin/env python3
## skip X and O
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Filter out rows with X or O emotion labels")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input text file with emotion labels")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output filtered text file")
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
    
    # Process the file
    valid_count = 0
    skipped_count = 0
    
    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        for line in in_file:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Parse the line to extract the emotion label
            parts = line.split(';')
            if len(parts) < 2:
                # If the line format is invalid, write it to output anyway
                out_file.write(line)
                valid_count += 1
                continue
            
            # Get the emotion label (the second part after splitting by ';')
            emotion = parts[1].strip()
            
            # Skip X and O labels
            if emotion in ['X', 'O']:
                skipped_count += 1
                continue
            
            # Write valid lines to output
            out_file.write(line)
            valid_count += 1
    
    print(f"Processing complete.")
    print(f"Kept {valid_count} valid lines.")
    print(f"Filtered out {skipped_count} lines with X or O labels.")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()