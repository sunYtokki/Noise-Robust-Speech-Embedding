#!/usr/bin/env python3
import os
import argparse
import re

def extract_problematic_files(error_log_file):
    """
    Extract the list of problematic audio files from the error log.
    
    Args:
        error_log_file (str): Path to the error log file
        
    Returns:
        list: List of problematic audio file paths
    """
    problematic_files = []
    
    with open(error_log_file, 'r') as f:
        content = f.read()
        
        # Use regex to find all error messages about audio files
        pattern = r'Error loading audio file ([^\n:]+):'
        matches = re.findall(pattern, content)
        
        for match in matches:
            problematic_files.append(match.strip())
    
    return problematic_files

def save_problematic_files_list(problematic_files, output_file):
    """
    Save the list of problematic audio files to a file.
    
    Args:
        problematic_files (list): List of problematic audio file paths
        output_file (str): Path to the output file
    """
    with open(output_file, 'w') as f:
        for file_path in problematic_files:
            f.write(f"{file_path}\n")
    
    print(f"Saved list of {len(problematic_files)} problematic files to {output_file}")

def remove_from_label_file(label_file, output_label_file, problematic_files):
    """
    Remove entries with problematic audio files from the label file.
    
    Args:
        label_file (str): Path to the original label file
        output_label_file (str): Path to the output label file
        problematic_files (list): List of problematic audio file paths
        
    Returns:
        int: Number of entries removed
    """
    # Create a set of filenames (without path) for easy lookup
    problematic_filenames = set()
    for filepath in problematic_files:
        filename = os.path.basename(filepath)
        problematic_filenames.add(filename)
    
    removed_count = 0
    
    with open(label_file, 'r') as f_in, open(output_label_file, 'w') as f_out:
        # If the file has a header, preserve it
        first_line = f_in.readline()
        f_out.write(first_line)
        
        for line in f_in:
            # Check if line contains any of the problematic filenames
            skip_line = False
            for filename in problematic_filenames:
                if filename in line:
                    skip_line = True
                    removed_count += 1
                    break
            
            if not skip_line:
                f_out.write(line)
    
    return removed_count

def remove_symlinks(symlink_dir, problematic_files):
    """
    Remove symbolic links to problematic audio files.
    
    Args:
        symlink_dir (str): Directory containing symbolic links
        problematic_files (list): List of problematic audio file paths
        
    Returns:
        int: Number of symbolic links removed
    """
    # Create a set of filenames (without path) for easy lookup
    problematic_filenames = set()
    for filepath in problematic_files:
        filename = os.path.basename(filepath)
        problematic_filenames.add(filename)
    
    removed_count = 0
    
    for filename in os.listdir(symlink_dir):
        if filename in problematic_filenames:
            symlink_path = os.path.join(symlink_dir, filename)
            
            if os.path.islink(symlink_path):
                try:
                    os.unlink(symlink_path)
                    removed_count += 1
                    print(f"Removed symlink: {symlink_path}")
                except Exception as e:
                    print(f"Error removing symlink {symlink_path}: {e}")
    
    return removed_count

def main():
    parser = argparse.ArgumentParser(description='Remove problematic audio files from label file and symlinks')
    parser.add_argument('--error-log', '-e', required=True, help='Path to the error log file')
    parser.add_argument('--label-file', '-l', required=True, help='Path to the label file')
    parser.add_argument('--output-label-file', '-o', required=True, help='Path to the output label file')
    parser.add_argument('--symlink-dir', '-s', required=False, help='Directory containing symbolic links')
    parser.add_argument('--problem-list', '-p', required=False, help='Output file for the list of problematic files')
    
    args = parser.parse_args()
    
    # Extract problematic files from error log
    problematic_files = extract_problematic_files(args.error_log)
    print(f"Found {len(problematic_files)} problematic audio files")
    
    # Save the list of problematic files if requested
    if args.problem_list:
        save_problematic_files_list(problematic_files, args.problem_list)
    
    # Remove from label file
    removed_from_labels = remove_from_label_file(args.label_file, args.output_label_file, problematic_files)
    print(f"Removed {removed_from_labels} entries from label file")
    
    # Remove symlinks if directory provided
    if args.symlink_dir:
        removed_symlinks = remove_symlinks(args.symlink_dir, problematic_files)
        print(f"Removed {removed_symlinks} symbolic links")
    
    print("Done!")

if __name__ == '__main__':
    main()



# ####
# python remove_problematic_files.py \
#   --error-log /proj/speech/users/syk2145/nrse/labels/processed/error_logging.txt \
#   --label-file /proj/speech/users/syk2145/nrse/labels/processed/baselin_noisy_train.csv \
#   --output-label-file baselin_noisy_train_cleaned.csv \
#   --symlink-dir /proj/speech/users/syk2145/baseline/MSP-Podcast_Challenge/Audios/ \
#   --problem-list problematic_files.txt