# Blogs — Working Guide

This document explains how the Blogs folder is organized, how to add new topics, and the recommended git workflow for contributing changes back to the upstream repo.

## Folder Structure

    Blogs/
    ├── index_blogs.html
    ├── README.md
    └── topics/
        ├── databricks-snippets/
        ├── python-snippets/
        ├── notes-caveats/
        ├── troubleshoots/
        ├── pandas-useful/
        ├── dec-2021/
        └── jan-2022/

Key principle: one folder per topic. Everything related to a topic lives together.

## Adding a New Topic

### Step 1 — Create the folder

    cd Blogs
    mkdir -p topics/my-new-topic

### Step 2 — Create the fragment file

Fragments must be barebones HTML — no DOCTYPE, html, head, body, style, or script tags. Just a section wrapper.

Template for topics/my-new-topic/my-new-topic.html:

    <section>
      <h1>My New Topic</h1>
      <p>Some explanation.</p>
      <pre><code class="language-python">
    print('hello')
      </code></pre>
    </section>

### Step 3 — Register it in index_blogs.html

3a. Add a div inside main:

    <main>
      <div id="my-new-topic"></div>
      <div id="pandas-useful"></div>
    </main>

3b. Add includeHTML() call:

    includeHTML('topics/my-new-topic/my-new-topic.html', 'my-new-topic');

## Testing Locally

    cd ~/github/bhishanpoudel123.github.io
    python -m http.server 8080

Open http://localhost:8080/Blogs/index_blogs.html

## Git Workflow

    # Sync
    git checkout main
    git fetch upstream
    git merge upstream/main
    git push origin main

    # Feature branch
    git checkout -b my-feature-name

    # Commit + push
    git add .
    git commit -m "Describe your change"
    git push origin my-feature-name

    # Open PR on GitHub, then after merge:
    git checkout main
    git fetch upstream
    git merge upstream/main
    git push origin main
    git branch -d my-feature-name
    git push origin --delete my-feature-name

## TL;DR

    git checkout main && git fetch upstream && git merge upstream/main && git push origin main
    mkdir -p Blogs/topics/my-topic
    python -m http.server 8080
    git checkout -b add-my-topic
    git add . && git commit -m "Add my-topic" && git push origin add-my-topic

---