import os
import json
from pathlib import Path
import datetime

def format_category_name(folder_name):
    """Convert folder name (with underscores) to display name (with spaces)"""
    return folder_name.replace('_', ' ')

def read_file_content(file_path):
    """Read content from a file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def render_markdown(content):
    """Convert markdown to HTML with proper code block handling"""
    import markdown
    from markdown.extensions.codehilite import CodeHiliteExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
    
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
    
    # Ensure code blocks have language classes
    content = content.replace('```python', '```python')
    content = content.replace('```javascript', '```javascript')
    content = content.replace('```bash', '```bash')
    content = content.replace('```sql', '```sql')
    
    return markdown.markdown(content, extensions=extensions)

def create_topic_html(topic_dir, topic_folder):
    """Create a comprehensive HTML file for a topic"""
    topic_name = format_category_name(topic_folder)
    questions_dir = os.path.join(topic_dir, 'questions')
    output_file = os.path.join(topic_dir, f"{topic_folder.lower()}.html")
    
    if not os.path.exists(questions_dir):
        print(f"⚠️ No questions directory found for topic: {topic_name}")
        return None
    
    # HTML template with styling and structure
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic_name} Learning Resources</title>
    <!-- PrismJS CSS -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css" rel="stylesheet" />
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/toolbar/prism-toolbar.min.css" rel="stylesheet" />
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .toc {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .question {{
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }}
        .question-title {{
            color: #3498db;
        }}
        .answer {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .explanation {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        details {{
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        summary {{
            font-weight: bold;
            cursor: pointer;
        }}
        pre[class*="language-"] {{
            margin: 1em 0;
            border-radius: 5px;
            background: #2d2d2d;
            padding: 1em;
            overflow: auto;
        }}
        code[class*="language-"] {{
            font-size: 14px;
            font-family: 'Fira Code', 'Consolas', monospace;
        }}
        .token.comment,
        .token.prolog,
        .token.doctype,
        .token.cdata {{
            color: #999;
        }}
        .token.punctuation {{
            color: #ccc;
        }}
        .token.property,
        .token.tag,
        .token.boolean,
        .token.number,
        .token.constant,
        .token.symbol,
        .token.deleted {{
            color: #f92672;
        }}
        .token.selector,
        .token.attr-name,
        .token.string,
        .token.char,
        .token.builtin,
        .token.inserted {{
            color: #a6e22e;
        }}
        .token.operator,
        .token.entity,
        .token.url,
        .language-css .token.string,
        .style .token.string {{
            color: #67cdcc;
        }}
        .token.atrule,
        .token.attr-value,
        .token.keyword {{
            color: #66d9ef;
        }}
        .token.function,
        .token.class-name {{
            color: #e6db74;
        }}
        .token.regex,
        .token.important,
        .token.variable {{
            color: #fd971f;
        }}
        .resource {{
            margin-top: 15px;
        }}
        .resource-title {{
            font-weight: bold;
            color: #2c3e50;
        }}
        :not(pre) > code {{
            padding: 2px 5px;
            background: #f0f0f0;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        details[open] {{
            padding-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{topic_name} Learning Resources</h1>
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul id="toc-list">
                <!-- TOC will be generated by JavaScript -->
            </ul>
        </div>
        <div id="questions-container">
            <!-- Questions will be inserted here -->
        </div>
    </div>

    <!-- PrismJS Core -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <!-- PrismJS Autoloader -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <!-- PrismJS Toolbar -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/toolbar/prism-toolbar.min.js"></script>
    <!-- PrismJS Copy to Clipboard -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js"></script>
    <!-- PrismJS Line Numbers -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.js"></script>
    <!-- Language Components -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markdown.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Generate TOC
            const questions = document.querySelectorAll('.question');
            const tocList = document.getElementById('toc-list');
            
            questions.forEach(function(question, index) {{
                const questionId = 'q-' + index;
                question.id = questionId;
                const title = question.querySelector('.question-title').textContent;
                
                const tocItem = document.createElement('li');
                const tocLink = document.createElement('a');
                tocLink.href = '#' + questionId;
                tocLink.textContent = title;
                tocItem.appendChild(tocLink);
                tocList.appendChild(tocItem);
            }});
            
            // Initialize Prism for all code blocks
            Prism.highlightAll();
            
            // Special handling for details elements
            document.querySelectorAll('details').forEach(detail => {{
                detail.addEventListener('toggle', function() {{
                    if (this.open) {{
                        Prism.highlightAllUnder(this);
                    }}
                }});
            }});
        }});
    </script>
</body>
</html>
"""
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_template, 'html.parser')
    questions_container = soup.find('div', {'id': 'questions-container'})
    
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
            question_div['id'] = f'q-{index}'
            
            qn_title = soup.new_tag('h2', attrs={'class': 'question-title'})
            qn_title.string = f"Qn {question_data['id']}: {question_data['question']}"
            question_div.append(qn_title)
            
            answer_div = soup.new_tag('div', attrs={'class': 'answer'})
            answer_title = soup.new_tag('h3')
            answer_title.string = "Answer"
            answer_div.append(answer_title)
            answer_div.append(BeautifulSoup(render_markdown(question_data['answer']), 'html.parser'))
            question_div.append(answer_div)
            
            if 'explanation' in question_data and question_data['explanation']:
                explanation_div = soup.new_tag('div', attrs={'class': 'explanation'})
                explanation_title = soup.new_tag('h3')
                explanation_title.string = "Explanation"
                explanation_div.append(explanation_title)
                explanation_div.append(BeautifulSoup(render_markdown(question_data['explanation']), 'html.parser'))
                question_div.append(explanation_div)
            
            if 'learning_resources' in question_data and question_data['learning_resources']:
                resources_title = soup.new_tag('h3')
                resources_title.string = "Additional Resources"
                question_div.append(resources_title)
                
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                
                for resource in question_data['learning_resources']:
                    resource_path = resource['path']
                    
                    # Handle both absolute and relative paths
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
                        if resource['type'] == 'markdown':
                            content = read_file_content(full_path)
                            resource_div.append(BeautifulSoup(render_markdown(content), 'html.parser'))
                        elif resource['type'] == 'html':
                            content = read_file_content(full_path)
                            resource_div.append(BeautifulSoup(content, 'html.parser'))
                        elif resource['type'] == 'code':
                            content = read_file_content(full_path)
                            pre = soup.new_tag('pre', **{'class': 'line-numbers'})
                            file_extension = resource_path.split('.')[-1].lower()
                            language = {
                                'py': 'python',
                                'js': 'javascript',
                                'sh': 'bash',
                                'sql': 'sql',
                                'md': 'markdown'
                            }.get(file_extension, file_extension)
                            
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
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f"\n🎉 Created topic HTML: {output_file}")
    return output_file

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    if not os.path.exists(data_dir):
        print("⛔ Error: 'data' directory not found")
        return
    
    print("🚀 Starting HTML creation for topics...")
    print(f"📂 Data directory: {os.path.abspath(data_dir)}")
    
    topics = [d for d in os.listdir(data_dir) 
              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    if not topics:
        print("⛔ No topics found in data directory")
        return
    
    print(f"\nFound {len(topics)} topics: {', '.join(format_category_name(t) for t in topics)}")
    
    for topic_folder in topics:
        topic_dir = os.path.join(data_dir, topic_folder)
        html_file = create_topic_html(topic_dir, topic_folder)
        if html_file:
            print(f"✅ Created HTML for {format_category_name(topic_folder)}")
    
    print("\n🏁 Completed HTML generation for all topics")

if __name__ == "__main__":
    try:
        from bs4 import BeautifulSoup
        import markdown
        from markdown.extensions.codehilite import CodeHiliteExtension
        from markdown.extensions.fenced_code import FencedCodeExtension
    except ImportError:
        print("Installing required packages...")
        import subprocess
        # subprocess.call(['pip', 'install', 'beautifulsoup4', 'markdown', 'pygments'])
    main()