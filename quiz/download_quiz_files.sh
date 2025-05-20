#!/bin/bash

# Base URL
base="https://raw.githubusercontent.com/bhishanpoudel123/bhishanpoudel123.github.io/refs/heads/main/quiz"

# File list
files=(
	"mcq/styles.css"
	"mcq/index.html"
	"mcq/quiz.js"
	"mcq/data/index.json"
	"mcq/data/Data_Analysis/questions/qn_01/qn_01.json"
)

# Loop through each file
for file in "${files[@]}"; do
	url="$base/$file"
	filename=$(basename "$file")

	# Check if file exists
	if curl --head --silent --fail "$url" >/dev/null; then
	    echo ""
		echo "Downloading $url -> $filename"
		curl -s "$url" -o "$filename"
	else
		echo "❌ File not found: $url"
	fi
done
