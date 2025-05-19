import os
import json
from pathlib import Path
import datetime

def format_category_name(folder_name):
    """Convert folder name (with underscores) to display name (with spaces)"""
    return folder_name.replace('_', ' ')

def validate_question_json(question_path, topic_folder):
    """Validate the structure of a question JSON file"""
    try:
        with open(question_path, 'r') as f:
            data = json.load(f)
        
        required_fields = ['id', 'tags', 'question', 'options', 'answer', 'explanation']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate tags is a list and contains the topic name
        if not isinstance(data['tags'], list):
            raise ValueError("Tags must be a list")
        
        expected_topic = format_category_name(topic_folder)
        if expected_topic not in data['tags']:
            raise ValueError(f"Tags should include the topic name: '{expected_topic}'")
        
        if not isinstance(data['options'], list) or len(data['options']) < 2:
            raise ValueError("Options must be a list with at least 2 items")
        
        if data['answer'] not in data['options']:
            raise ValueError("Answer must be one of the provided options")
        
        return True
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    except Exception as e:
        raise ValueError(str(e))

def create_topic_index(topic_dir, topic_folder):
    """Create the questions index file for a specific topic"""
    topic_name = format_category_name(topic_folder)
    questions_dir = os.path.join(topic_dir, 'questions')
    output_file = os.path.join(topic_dir, f"{topic_folder.lower()}_questions.json")
    
    if not os.path.exists(questions_dir):
        print(f"⚠️ No questions directory found for topic: {topic_name}")
        return None
    
    question_paths = []
    question_count = 0
    
    print(f"\n🔍 Scanning topic: {topic_name}")
    
    # Find all question folders (qn_*)
    for dir_name in sorted(os.listdir(questions_dir)):
        if dir_name.startswith('qn_'):
            qn_dir = os.path.join(questions_dir, dir_name)
            qn_json = os.path.join(qn_dir, f"{dir_name}.json")
            
            if not os.path.exists(qn_json):
                print(f"⚠️ Missing JSON file in question folder: {dir_name}")
                continue
            
            try:
                # Validate the question JSON
                validate_question_json(qn_json, topic_folder)
                
                # Create the full correct path starting with data/
                question_path = f"data/{topic_folder}/questions/{dir_name}/{dir_name}.json"
                question_paths.append(question_path)
                question_count += 1
                print(f"✅ Valid question found: {dir_name}")
                
            except ValueError as e:
                print(f"❌ Invalid question {dir_name}: {str(e)}")
                continue
    
    if not question_paths:
        print("⛔ No valid questions found for this topic")
        return None
    
    # Create the output data structure
    output_data = {
        "questions": question_paths,
        "metadata": {
            "topic": topic_name,
            "question_count": question_count
        }
    }
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n🎉 Created topic index: {output_file}")
    print(f"Example question path: {question_paths[0]}")
    return output_file

def create_main_index(data_dir, topics):
    """Create or update the main index.json file"""
    main_index_path = os.path.join(data_dir, 'index.json')
    main_index = {
        "categories": {},
        "metadata": {
            "version": "1.0",
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d")
        }
    }
    
    # Build the categories dictionary
    for topic_folder in topics:
        topic_name = format_category_name(topic_folder)
        index_file = f"data/{topic_folder}/{topic_folder.lower()}_questions.json"
        main_index["categories"][topic_name] = index_file
    
    # Write to file
    with open(main_index_path, 'w') as f:
        json.dump(main_index, f, indent=2)
    
    print(f"\n📋 Created main index: {main_index_path}")
    return main_index_path

def main():
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if not os.path.exists(data_dir):
        print("⛔ Error: 'data' directory not found")
        return
    
    print("🚀 Starting index creation...")
    print(f"📂 Data directory: {os.path.abspath(data_dir)}")
    
    # Find all topic directories (direct subdirectories of data/)
    topics = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    if not topics:
        print("⛔ No topics found in data directory")
        return
    
    print(f"\nFound {len(topics)} topics: {', '.join(format_category_name(t) for t in topics)}")
    
    # Create index files for each topic
    topic_index_files = []
    for topic_folder in topics:
        topic_dir = os.path.join(data_dir, topic_folder)
        index_file = create_topic_index(topic_dir, topic_folder)
        if index_file:
            topic_index_files.append(index_file)
    
    # Create the main index.json
    create_main_index(data_dir, topics)
    
    print(f"\n🏁 Completed: Created {len(topic_index_files)} topic index files")

if __name__ == "__main__":
    main()