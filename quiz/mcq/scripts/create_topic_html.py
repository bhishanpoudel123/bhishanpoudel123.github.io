import os
import json
from bs4 import BeautifulSoup
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension

# Default theme
DEFAULT_THEME = 'solarized_light'

# Theme configuration
THEMES = {
    'solarized_dark': {
        'template': 'html_template_solarized_dark.html',
        'description': 'Solarized Dark Theme'
    },
    'solarized_light': {
        'template': 'html_template_solarized_light.html',
        'description': 'Solarized Light Theme'
    },
    'one_dark': {
        'template': 'html_template_one_dark.html',
        'description': 'One Dark Theme'
    },
    'dracula': {
        'template': 'html_template_dracula.html',
        'description': 'Dracula Theme'
    },
    'github': {
        'template': 'html_template_github.html',
        'description': 'GitHub Theme'
    }
}

def check_dependencies():
    """Check for required Python packages"""
    required = ['markdown', 'bs4']
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("⛔ Error: Missing required Python packages. Please install them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def format_category_name(folder_name):
    """Convert folder name to display name"""
    return folder_name.replace('_', ' ')

def read_file_content(file_path):
    """Read content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def render_markdown(content):
    """Convert markdown to HTML with proper code block handling"""
    extensions = [
        CodeHiliteExtension(
            linenums=False,
            css_class='language-',
            guess_lang=True,
            use_pygments=False
        ),
        FencedCodeExtension(),
        'tables',
        'nl2br'
    ]
    
    # Ensure code blocks have proper language classes
    replacements = [
        ('```python', '```python'),
        ('```javascript', '```javascript'),
        ('```bash', '```bash'),
        ('```sql', '```sql'),
        ('```html', '```html'),
        ('```json', '```json'),
        ('```yaml', '```yaml'),
        ('```markdown', '```markdown')
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Convert markdown to HTML
    html = markdown.markdown(content, extensions=extensions)
    
    # Post-process to ensure proper code block structure
    soup = BeautifulSoup(html, 'html.parser')
    
    for pre in soup.find_all('pre'):
        if not pre.code:
            code = soup.new_tag('code')
            code.string = pre.get_text()
            pre.clear()
            pre.append(code)
        else:
            code = pre.code
        
        # Ensure code has language class
        if not any(cls.startswith('language-') for cls in code.get('class', [])):
            code['class'] = code.get('class', []) + ['language-text']
    
    return str(soup)

def load_html_template(theme=DEFAULT_THEME):
    """Load the HTML template for the specified theme"""
    if theme not in THEMES:
        theme = DEFAULT_THEME
    
    template_path = os.path.join(os.path.dirname(__file__), THEMES[theme]['template'])
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading template: {str(e)}")
        raise

def create_topic_html(topic_dir, topic_folder, theme=DEFAULT_THEME):
    """Create HTML file for a topic with specified theme"""
    topic_name = format_category_name(topic_folder)
    questions_dir = os.path.join(topic_dir, 'questions')
    output_file = os.path.join(topic_dir, f"{topic_folder.lower()}.html")
    
    if not os.path.exists(questions_dir):
        print(f"⚠️ No questions directory found for topic: {topic_name}")
        return None
    
    try:
        # Load and prepare HTML template
        html_template = load_html_template(theme).replace('{{TOPIC_NAME}}', topic_name)
        soup = BeautifulSoup(html_template, 'html.parser')
        
        # Verify questions container exists
        questions_container = soup.find('div', {'id': 'questions-container'})
        if questions_container is None:
            raise ValueError("Missing questions-container div in template")
        
        # Process each question
        question_folders = sorted([d for d in os.listdir(questions_dir) if d.startswith('qn_')])
        
        for index, qn_folder in enumerate(question_folders):
            qn_dir = os.path.join(questions_dir, qn_folder)
            qn_json = os.path.join(qn_dir, f"{qn_folder}.json")
            
            if not os.path.exists(qn_json):
                continue
            
            try:
                with open(qn_json, 'r', encoding='utf-8') as f:
                    question_data = json.load(f)
                
                question_div = soup.new_tag('div', attrs={'class': 'question'})
                question_div['id'] = f'q-{index + 1}'
                
                # Add question title
                qn_title = soup.new_tag('h2', attrs={'class': 'question-title'})
                qn_title.string = f"Qn {question_data['id']}: {question_data['question']}"
                question_div.append(qn_title)
                
                # Add answer
                answer_div = soup.new_tag('div', attrs={'class': 'answer'})
                answer_title = soup.new_tag('h3')
                answer_title.string = "Answer"
                answer_div.append(answer_title)
                answer_div.append(BeautifulSoup(render_markdown(question_data['answer']), 'html.parser'))
                question_div.append(answer_div)
                
                # Add explanation if exists
                if 'explanation' in question_data and question_data['explanation']:
                    explanation_div = soup.new_tag('div', attrs={'class': 'explanation'})
                    explanation_title = soup.new_tag('h3')
                    explanation_title.string = "Explanation"
                    explanation_div.append(explanation_title)
                    explanation_div.append(BeautifulSoup(render_markdown(question_data['explanation']), 'html.parser'))
                    question_div.append(explanation_div)
                
                # Add learning resources if they exist
                if 'learning_resources' in question_data and question_data['learning_resources']:
                    resources_title = soup.new_tag('h3')
                    resources_title.string = "Additional Resources"
                    question_div.append(resources_title)
                    
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    
                    for resource in question_data['learning_resources']:
                        resource_path = resource['path']
                        if resource_path.startswith('data/'):
                            full_path = os.path.normpath(os.path.join(project_root, resource_path))
                        else:
                            full_path = os.path.normpath(os.path.join(qn_dir, resource_path))
                        
                        details = soup.new_tag('details')
                        summary = soup.new_tag('summary')
                        summary.string = f"{resource['title']} ({resource['type']})"
                        details.append(summary)
                        
                        resource_div = soup.new_tag('div', attrs={'class': 'resource'})
                        
                        try:
                            content = read_file_content(full_path)
                            
                            if resource['type'] == 'markdown':
                                md_container = soup.new_tag('div', attrs={'class': 'markdown-content'})
                                md_container.append(BeautifulSoup(render_markdown(content), 'html.parser'))
                                resource_div.append(md_container)
                            elif resource['type'] == 'html':
                                html_container = soup.new_tag('div', attrs={'class': 'html-content'})
                                html_container.append(BeautifulSoup(content, 'html.parser'))
                                resource_div.append(html_container)
                            elif resource['type'] == 'code':
                                pre = soup.new_tag('pre', **{'class': 'line-numbers'})
                                ext = resource_path.split('.')[-1].lower()
                                lang_map = {
                                    'py': 'python',
                                    'js': 'javascript',
                                    'sh': 'bash',
                                    'sql': 'sql',
                                    'md': 'markdown',
                                    'html': 'markup',
                                    'css': 'css',
                                    'yml': 'yaml',
                                    'yaml': 'yaml',
                                    'json': 'json'
                                }
                                language = lang_map.get(ext, ext)
                                code = soup.new_tag('code', **{'class': f'language-{language}'})
                                code.string = content
                                pre.append(code)
                                resource_div.append(pre)
                            
                            details.append(resource_div)
                            question_div.append(details)
                        
                        except Exception as e:
                            error_msg = soup.new_tag('p')
                            error_msg.string = f"Error loading resource: {str(e)} - Path: {full_path}"
                            error_msg['style'] = "color: red;"
                            resource_div.append(error_msg)
                            details.append(resource_div)
                            question_div.append(details)
                
                questions_container.append(question_div)
            
            except Exception as e:
                error_div = soup.new_tag('div')
                error_div.string = f"Error processing question {qn_folder}: {str(e)}"
                error_div['style'] = "color: red; padding: 10px; margin: 10px 0; border: 1px solid red;"
                questions_container.append(error_div)
        
        # Write the final HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        
        print(f"🎉 Created topic HTML ({THEMES[theme]['description']}): {output_file}")
        return output_file
    
    except Exception as e:
        print(f"Error creating HTML for {topic_name}: {str(e)}")
        return None

def main():
    if not check_dependencies():
        return
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if not os.path.exists(data_dir):
        print("⛔ Error: 'data' directory not found")
        return

    print("🚀 Starting HTML creation for topics...")
    print("Available themes:")
    for theme, config in THEMES.items():
        print(f"  {theme}: {config['description']}")

    # Get theme choice from user
    theme_choice = DEFAULT_THEME #input(f"Select theme (default: {DEFAULT_THEME}): ").strip().lower()
    theme = theme_choice if theme_choice in THEMES else DEFAULT_THEME
    
    topics = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    if not topics:
        print("⛔ No topics found in data directory")
        return
    
    print(f"Found {len(topics)} topics: {', '.join(format_category_name(t) for t in topics)}")
    print(f"Using theme: {THEMES[theme]['description']}")
    
    for topic_folder in topics:
        topic_dir = os.path.join(data_dir, topic_folder)
        html_file = create_topic_html(topic_dir, topic_folder, theme)
        if html_file:
            print(f"✅ Created HTML for {format_category_name(topic_folder)}")
    
    print("\n🏁 Completed HTML generation for all topics")

if __name__ == "__main__":
    main()