#!/usr/bin/env python3
"""
Convert FAQ JSONL data to Harmony messages format for finetuning.
"""

import json
import sys
from pathlib import Path

def convert_to_harmony_messages(input_file, output_file):
    """
    Convert JSONL FAQ data to Harmony messages format.
    
    Harmony format:
    {
        "messages": [
            {
                "role": "user",
                "content": "instruction + input"
            },
            {
                "role": "assistant", 
                "content": "output"
            }
        ]
    }
    """
    
    harmony_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Extract fields
                instruction = data.get('instruction', '')
                input_text = data.get('input', '')
                output = data.get('output', '')
                meta = data.get('meta', {})
                
                # Combine instruction and input for user message
                user_content = instruction
                if input_text:
                    user_content = f"{instruction}\n\n{input_text}"
                
                # Create Harmony message format
                harmony_message = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_content
                        },
                        {
                            "role": "assistant",
                            "content": output
                        }
                    ]
                }
                
                # Add metadata if available
                if meta:
                    harmony_message["meta"] = meta
                
                harmony_data.append(harmony_message)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content: {line[:100]}...")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    
    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in harmony_data:
            json.dump(item, f, ensure_ascii=False, separators=(',', ':'))
            f.write('\n')
    
    print(f"Successfully converted {len(harmony_data)} FAQ entries to Harmony format")
    print(f"Output saved to: {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_to_harmony.py <input_file> <output_file>")
        print("Example: python convert_to_harmony.py faq_all.jsonl faq_harmony.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    try:
        convert_to_harmony_messages(input_file, output_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
