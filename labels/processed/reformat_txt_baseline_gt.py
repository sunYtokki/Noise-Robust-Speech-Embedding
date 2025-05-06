#!/usr/bin/env python3
import os
import re
import argparse

def emotion_class_to_one_hot(emotion_class):
    """Convert emotion class code to one-hot encoding."""
    emotions = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']
    one_hot = [0.0] * len(emotions)
    
    if emotion_class == 'A':  # Angry
        one_hot[0] = 1.0
    elif emotion_class == 'S':  # Sad
        one_hot[1] = 1.0
    elif emotion_class == 'H':  # Happy
        one_hot[2] = 1.0
    elif emotion_class == 'U':  # Surprise
        one_hot[3] = 1.0
    elif emotion_class == 'F':  # Fear
        one_hot[4] = 1.0
    elif emotion_class == 'D':  # Disgust
        one_hot[5] = 1.0
    elif emotion_class == 'C':  # Contempt
        one_hot[6] = 1.0
    elif emotion_class == 'N':  # Neutral
        one_hot[7] = 1.0
    
    return one_hot

def get_base_filename(filename):
    """
    Extract base filename from augmented audio filename.
    Handles various MSP-PODCAST naming formats:
    - MSP-PODCAST_0251_0871.wav
    - MSP-PODCAST_0133_0459_snr2_IPb6u22cLIo.wav
    - MSP-PODCAST_3469_0114_0006_snr2_jtwYYpoqHcM.wav
    """
    # First, check if it's an augmented file with _snr pattern
    if '_snr' in filename:
        # Extract the part before _snr
        parts = filename.split('_snr')
        base_part = parts[0]
        
        # Check if the base part has the format MSP-PODCAST_XXXX_XXXX_XXXX or MSP-PODCAST_XXXX_XXXX
        segments = base_part.split('_')
        
        if len(segments) >= 3:  # We have at least MSP-PODCAST_XXXX_XXXX
            # For 3-segment IDs (MSP-PODCAST_XXXX_XXXX), keep all
            # For 4+-segment IDs (MSP-PODCAST_XXXX_XXXX_XXXX), try both the 3-segment and 4-segment versions
            base_name_3seg = f"{segments[0]}_{segments[1]}_{segments[2]}.wav"
            
            if len(segments) >= 4:
                base_name_4seg = f"{segments[0]}_{segments[1]}_{segments[2]}_{segments[3]}.wav"
                return [base_name_3seg, base_name_4seg]
            else:
                return [base_name_3seg]
    
    # If no _snr pattern or we couldn't parse it, just return the filename as is
    return [filename]

def reformat_labels(input_file, ground_truth_file, output_file, preserve_path=False):
    """
    Reformat labels from input file format to target format using information from ground truth file.
    
    Args:
        input_file (str): Path to the input label file
        ground_truth_file (str): Path to the ground truth label file
        output_file (str): Path to the output label file
        preserve_path (bool): Whether to preserve the full path in the output
    """
    # Build a dictionary of ground truth information
    ground_truth_dict = {}
    
    with open(ground_truth_file, 'r') as gt_file:
        header = gt_file.readline().strip().split(',')
        split_set_index = header.index('Split_Set') if 'Split_Set' in header else None
        
        for line in gt_file:
            if not line.strip():
                continue
            
            parts = line.strip().split(',')
            if len(parts) <= split_set_index:
                continue
            
            filename = parts[0]
            split_set = parts[split_set_index] if split_set_index is not None else "Unknown"
            
            # Store the split set information in the dictionary
            ground_truth_dict[filename] = split_set
    
    # Write the output file
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        # Write header
        out_file.write("FileName,Angry,Sad,Happy,Surprise,Fear,Disgust,Contempt,Neutral,Split_Set\n")
        
        for line in in_file:
            if not line.strip():
                continue
            
            # Parse the input line
            parts = line.strip().split(';')
            if len(parts) < 2:
                continue
            
            # Extract the file path and filename
            file_path = parts[0].strip()
            filename = os.path.basename(file_path)
            
            # Extract emotion class
            emotion_class = parts[1].strip() if len(parts) > 1 else "N"
            
            # Convert emotion class to one-hot encoding
            one_hot = emotion_class_to_one_hot(emotion_class)
            
            # Try to get the split set from ground truth using different key variations
            split_set = ground_truth_dict.get(filename, None)
            
            # If not found, try with possible base filenames
            if split_set is None:
                possible_base_filenames = get_base_filename(filename)
                for base_filename in possible_base_filenames:
                    if base_filename in ground_truth_dict:
                        split_set = ground_truth_dict[base_filename]
                        break
            
            # If still not found, use "Unknown"
            if split_set is None:
                split_set = "Unknown"
                print(f"Warning: No match found for {filename}")
            
            # Create the output line
            output_filename = file_path if preserve_path else filename
            one_hot_str = ",".join([str(val) for val in one_hot])
            output_line = f"{output_filename},{one_hot_str},{split_set}\n"
            
            out_file.write(output_line)
    
    print(f"Reformatted labels from {input_file} to {output_file} using information from {ground_truth_file}")

def main():
    parser = argparse.ArgumentParser(description='Reformat label files to target format')
    parser.add_argument('--input-file', '-i', required=True, help='Input label file')
    parser.add_argument('--ground-truth-file', '-g', required=True, help='Ground truth label file')
    parser.add_argument('--output-file', '-o', required=True, help='Output label file')
    parser.add_argument('--preserve-path', '-p', action='store_true', help='Preserve full path in output')
    
    args = parser.parse_args()
    
    reformat_labels(args.input_file, args.ground_truth_file, args.output_file, args.preserve_path)

if __name__ == '__main__':
    main()