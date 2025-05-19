# Instruction
- To edit any topic, edit the json file for the given topic. e.g. `data/Data_Analysis/data_analysis_questions.json`.
- Only one json is parsed for the given topic, but that json can have many questions along with given markdown and html paths.
- In learning section, the topic can have multiple html files to learn from.
- Once, the files are changed, do following:
  + if a question in topic json is changed, no need to do anything.
  + if a new markdown or html is added for the question in topic, no need to change.
  + if a new html is added for the learning material, we need to create new index_html.json file.
  + if a new topic is added, we need to create new index.json and index_html.json.
  + for a given topic, from the json we can create learning material html using `create_topic_html.py`.

# Run web app locally
```bash
python -m http.server 8000
# then open page: localhost:8000
```

# Tree command in windows git bash
To see file structure in windows:
```bash
# tree # works in Macos
cmd //c tree
```

Alternative:
- download binary from: https://fabiochen.hashnode.dev/install-tree-command-in-windows-10
- put it in `C:\Program Files\Git\usr\bin`
- Then terminal command `tree` works.

# Debub web app in google chrome using developer tools
- press `F12` or `ctrl-shift-i` or `top right 3 dots > more tools > developer tools`, then look at console.