#!/usr/bin/env python3
import glob
import os

# Configuration
CSS_LINK = '    <link rel="stylesheet" type="text/css" href="../../assets/css/style_mobile.css">'
HEAD_END_TAG = '</head>'

def get_html_files():
    """Get HTML files in current dir and first-level subfolders"""
    files = []
    # Current directory
    files.extend(glob.glob('*.html'))
    # First-level subdirectories only
    for item in os.listdir('.'):
        if os.path.isdir(item):
            files.extend(glob.glob(f'{item}/*.html'))
    return files

def process_file(filepath):
    print(f"\n{'='*50}")
    print(f"Processing file: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    new_lines = []
    
    for i, line in enumerate(lines):
        if HEAD_END_TAG in line:
            print(f"\nFound closing head tag at line {i+1}:")
            print(f"Original line: {line.strip()}")
            
            if any(CSS_LINK.strip() in l for l in lines):
                print("Mobile CSS link already exists. Skipping.")
                return False
            
            new_lines.append(CSS_LINK + '\n')
            modified = True
            
            print("Added mobile CSS link:")
            print(CSS_LINK)
            print(f"New line becomes: {line.strip()}")
    
        new_lines.append(line)
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("\nFile modified successfully.")
        return True
    
    print("No changes made.")
    return False

def main():
    html_files = get_html_files()
    total_files = len(html_files)
    modified_count = 0
    
    print(f"\n{'*'*50}")
    print(f"Found {total_files} HTML files (current dir + first-level subfolders)")
    print(f"CSS link to add: {CSS_LINK}")
    print('*'*50)
    
    for filepath in html_files:
        if process_file(filepath):
            modified_count += 1
    
    print(f"\n{'#'*50}")
    print(f"Processing complete!")
    print(f"Files scanned: {total_files}")
    print(f"Files modified: {modified_count}")
    print('#'*50)

if __name__ == '__main__':
    main()