#!/usr/bin/env python3
"""
Filter markdown to include only main paper or only supplement.
Used to build separate Word files while maintaining cross-reference resolution.
"""

import sys

def filter_markdown(input_file, output_file, keep_supplement=False):
    """Filter markdown file to keep only main paper or only supplement."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    output_lines = []
    in_supplement = False
    
    for line in lines:
        # Check if we've reached the supplement marker
        stripped = line.strip()
        if stripped.startswith("# KCOR: Supplementary Material") or stripped.startswith("## Supplementary material"):
            in_supplement = True
        
        # Keep lines based on which section we want
        if keep_supplement:
            if in_supplement:
                output_lines.append(line)
        else:
            if not in_supplement:
                output_lines.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: filter_markdown.py <combined.md> <output.md> <main|supplement>", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    mode = sys.argv[3]
    
    if mode not in ['main', 'supplement']:
        print("ERROR: mode must be 'main' or 'supplement'", file=sys.stderr)
        sys.exit(1)
    
    filter_markdown(input_file, output_file, keep_supplement=(mode == 'supplement'))

