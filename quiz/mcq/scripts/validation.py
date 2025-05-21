import os
import json
import re

def validate_base_path_index_html(expected_base_path):
    """Validate base paths and file existence in index.html"""
    print("\nValidating index.html...")
    all_valid = True
    
    if not os.path.exists('index.html'):
        print("Error: index.html not found")
        return False

    with open('index.html', 'r', encoding='utf-8') as f:
        content = f.read()

    # Check favicon path
    favicon_pattern = r'<link\s+rel="icon"[^>]+href="([^"]+)"'
    favicon_match = re.search(favicon_pattern, content)
    
    if favicon_match:
        favicon_path = favicon_match.group(1)
        if not favicon_path.startswith(expected_base_path):
            print(f"Error: Favicon path should start with '{expected_base_path}', found '{favicon_path}'")
            all_valid = False
        
        # Check file existence - use path relative to current directory
        local_favicon_path = favicon_path.replace(expected_base_path, '').lstrip('/')
        if not os.path.exists(local_favicon_path):
            print(f"Error: Favicon file not found at '{local_favicon_path}'")
            all_valid = False
    else:
        print("Error: Could not find favicon link in index.html")
        all_valid = False

    # Check manifest path
    manifest_pattern = r'<link\s+rel="manifest"[^>]+href="([^"]+)"'
    manifest_match = re.search(manifest_pattern, content)
    
    if manifest_match:
        manifest_path = manifest_match.group(1)
        if not manifest_path.startswith(expected_base_path):
            print(f"Error: Manifest path should start with '{expected_base_path}', found '{manifest_path}'")
            all_valid = False
        
        # Check file existence - use path relative to current directory
        local_manifest_path = manifest_path.replace(expected_base_path, '').lstrip('/')
        if not os.path.exists(local_manifest_path):
            print(f"Error: Manifest file not found at '{local_manifest_path}'")
            all_valid = False
    else:
        print("Error: Could not find manifest link in index.html")
        all_valid = False

    if all_valid:
        print("✅ index.html validation passed")
    
    return all_valid

def validate_base_path_quiz_js(expected_base_path):
    """Validate base path in quiz.js"""
    print("\nValidating quiz.js...")
    all_valid = True
    
    if not os.path.exists('quiz.js'):
        print("Error: quiz.js not found")
        return False

    with open('quiz.js', 'r', encoding='utf-8') as f:
        content = f.read()

    base_path_pattern = r'const\s+basePath\s*=\s*isGitHubPages\s*\?\s*[\'"](/[^\'"]+)[\'"]\s*:\s*[\'"][^\'"]*[\'"]'
    match = re.search(base_path_pattern, content)
    
    if match:
        found_path = match.group(1)
        if found_path != expected_base_path:
            print(f"Error: basePath should be '{expected_base_path}', found '{found_path}'")
            all_valid = False
    else:
        print("Error: Could not find basePath definition in quiz.js")
        all_valid = False

    if all_valid:
        print("✅ quiz.js validation passed")
    
    return all_valid

def validate_manifest_json(expected_base_path):
    """Validate paths and file existence in manifest.json"""
    print("\nValidating manifest.json...")
    all_valid = True
    
    if not os.path.exists('manifest.json'):
        print("Error: manifest.json not found")
        return False

    try:
        with open('manifest.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in manifest.json")
        return False

    # Validate start_url
    start_url = data.get('start_url', '')
    if not start_url.startswith(expected_base_path):
        print(f"Error: start_url should start with '{expected_base_path}', found '{start_url}'")
        all_valid = False

    # Validate icons
    icons = data.get('icons', [{}])
    if not icons:
        print("Error: No icons defined in manifest.json")
        all_valid = False
    else:
        icon_src = icons[0].get('src', '')
        if not icon_src.startswith(expected_base_path):
            print(f"Error: icon src should start with '{expected_base_path}', found '{icon_src}'")
            all_valid = False
        
        # Check file existence - use path relative to current directory
        local_icon_path = icon_src.replace(expected_base_path, '').lstrip('/')
        if not os.path.exists(local_icon_path):
            print(f"Error: Icon file not found at '{local_icon_path}'")
            all_valid = False

    if all_valid:
        print("✅ manifest.json validation passed")
    
    return all_valid

def validate_file_paths_for_questions(root_dir='data'):
    """
    Validate that all file paths referenced in question JSON files across all topics exist.
    
    Args:
        root_dir (str): The root directory containing topic folders (default: 'data')
    
    Returns:
        dict: A dictionary with validation results, including missing files and error counts
    """
    validation_results = {
        'topics_checked': 0,
        'questions_checked': 0,
        'missing_files': [],
        'total_missing': 0
    }
    
    # Check if root directory exists
    if not os.path.exists(root_dir):
        return {'error': f"Root directory '{root_dir}' does not exist"}
    
    # Iterate through each topic folder in the root directory
    for topic in os.listdir(root_dir):
        topic_path = os.path.join(root_dir, topic)
        
        # Skip non-directories and the 'all_topics_questions_answers.md' file
        if not os.path.isdir(topic_path) or topic == 'all_topics_questions_answers.md':
            continue
            
        validation_results['topics_checked'] += 1
        questions_dir = os.path.join(topic_path, 'questions')
        
        # Check if questions directory exists
        if not os.path.exists(questions_dir):
            validation_results['missing_files'].append({
                'type': 'directory',
                'path': questions_dir,
                'error': 'Questions directory missing'
            })
            validation_results['total_missing'] += 1
            continue
            
        # Iterate through each question folder
        for qn_dir in os.listdir(questions_dir):
            qn_path = os.path.join(questions_dir, qn_dir)
            
            # Skip non-directories
            if not os.path.isdir(qn_path):
                continue
                
            qn_json_path = os.path.join(qn_path, f"{qn_dir}.json")
            
            # Check if question JSON file exists
            if not os.path.exists(qn_json_path):
                validation_results['missing_files'].append({
                    'type': 'file',
                    'path': qn_json_path,
                    'error': 'Question JSON file missing'
                })
                validation_results['total_missing'] += 1
                continue
                
            validation_results['questions_checked'] += 1
            
            try:
                # Load the question JSON file
                with open(qn_json_path, 'r', encoding='utf-8') as f:
                    qn_data = json.load(f)
                
                # Check all referenced files in the question data
                for field, file_path in qn_data.items():
                    # Skip non-file path fields (you might need to adjust this based on your JSON structure)
                    if not isinstance(file_path, str) or not any(ext in file_path for ext in ['.md', '.html', '.json', '.py', '.ipynb']):
                        continue
                    
                    # Handle relative paths (assuming paths are relative to the question directory)
                    full_path = os.path.join(qn_path, file_path)
                    
                    if not os.path.exists(full_path):
                        validation_results['missing_files'].append({
                            'type': 'file',
                            'path': full_path,
                            'error': f"Referenced file missing (from {qn_json_path})"
                        })
                        validation_results['total_missing'] += 1
                        
            except json.JSONDecodeError as e:
                validation_results['missing_files'].append({
                    'type': 'file',
                    'path': qn_json_path,
                    'error': f"Invalid JSON: {str(e)}"
                })
                validation_results['total_missing'] += 1
    
    return validation_results

def validate_all_files(parent_folder, working_folder):
    """Main validation function that runs all validations"""
    expected_base_path = f"/{parent_folder}/{working_folder}"
    print(f"\nStarting validation for base path: '{expected_base_path}'")
    
    results = [
        validate_base_path_index_html(expected_base_path),
        validate_base_path_quiz_js(expected_base_path),
        validate_manifest_json(expected_base_path),
        validate_file_paths_for_questions(expected_base_path)
    ]

    if all(results):
        print("\n🎉 All validations passed successfully!")
        return True
    else:
        print("\n❌ Validation failed. Please fix the reported issues.")
        return False


if __name__ == "__main__":
    # Configure these based on your folder structure
    PARENT_FOLDER = "quiz"
    WORKING_FOLDER = "mcq"
    
    if not validate_all_files(PARENT_FOLDER, WORKING_FOLDER):
        exit(1)