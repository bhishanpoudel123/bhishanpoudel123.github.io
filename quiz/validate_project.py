#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import markdown
import warnings


class QuizValidator:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "data"
        self.errors = []
        self.warnings = []

    def validate_all(self):
        """Run all validation checks"""
        print("=== Starting Quiz Data Validation ===")

        # Structural validations
        self.validate_directory_structure()
        self.validate_data_folders_json()

        # Content validations
        self.validate_tags_json()
        self.validate_all_question_files()
        self.validate_index_files()

        # Cross-referencing validations
        self.validate_tag_coverage()
        self.validate_question_references()

        self.report_results()
        return len(self.errors) == 0

    def validate_directory_structure(self):
        """Check basic project structure"""
        required_dirs = [
            "data",
            "data/interview_drfirst",
            "data/linear_regression",
            "css",
            "js",
        ]

        required_files = [
            "data/data_folders.json",
            "data/tags.json",
            "index.html",
            "css/styles.css",
            "js/quiz.js",
        ]

        for dir_path in required_dirs:
            if not (self.root_dir / dir_path).is_dir():
                self.errors.append(f"Missing directory: {dir_path}")

        for file_path in required_files:
            if not (self.root_dir / file_path).is_file():
                self.errors.append(f"Missing file: {file_path}")

    def validate_data_folders_json(self):
        """Validate data_folders.json structure and content"""
        try:
            with (self.data_dir / "data_folders.json").open() as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self.errors.append("data_folders.json should be a JSON object")
                return

            if "folders" not in data:
                self.errors.append("data_folders.json missing 'folders' key")
                return

            if not isinstance(data["folders"], list):
                self.errors.append("'folders' should be an array in data_folders.json")
                return

            for folder in data["folders"]:
                folder_path = self.data_dir / folder
                if not folder_path.is_dir():
                    self.errors.append(
                        f"Folder listed in data_folders.json does not exist: {folder}"
                    )

        except Exception as e:
            self.errors.append(f"Error reading data_folders.json: {str(e)}")

    def validate_tags_json(self):
        """Validate tags.json structure and content"""
        try:
            with (self.data_dir / "tags.json").open() as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self.errors.append("tags.json should be a JSON object")
                return

            required_keys = ["tags", "tag_details", "total_questions", "unique_tags"]
            for key in required_keys:
                if key not in data:
                    self.errors.append(f"tags.json missing required key: {key}")

            if not isinstance(data["tags"], list):
                self.errors.append("'tags' should be an array in tags.json")

            if not isinstance(data["tag_details"], dict):
                self.errors.append("'tag_details' should be an object in tags.json")

        except Exception as e:
            self.errors.append(f"Error reading tags.json: {str(e)}")

    def validate_index_files(self):
        """Validate all index.json files in question folders"""
        for index_file in self.data_dir.glob("*/index.json"):
            try:
                with index_file.open() as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    self.errors.append(f"{index_file} should be a JSON object")
                    continue

                if "files" not in data:
                    self.errors.append(f"{index_file} missing 'files' key")
                    continue

                if not isinstance(data["files"], list):
                    self.errors.append(f"'files' should be an array in {index_file}")
                    continue

                folder_path = index_file.parent
                for qn_file in data["files"]:
                    qn_path = folder_path / qn_file
                    if not qn_path.is_file():
                        self.errors.append(
                            f"Question file listed in {index_file} does not exist: {qn_file}"
                        )

            except Exception as e:
                self.errors.append(f"Error reading {index_file}: {str(e)}")

    def validate_all_question_files(self):
        """Validate all question JSON files"""
        question_files = list(self.data_dir.glob("*/*.json"))
        question_files = [f for f in question_files if f.name != "index.json"]

        for qn_file in question_files:
            self.validate_question_file(qn_file)

    def validate_question_file(self, filepath):
        """Validate a single question JSON file"""
        try:
            with filepath.open() as f:
                data = json.load(f)

            # Handle both array and single question formats
            questions = data if isinstance(data, list) else [data]

            for q in questions:
                if not isinstance(q, dict):
                    self.errors.append(f"{filepath} contains invalid question format")
                    continue

                self.validate_question_structure(filepath, q)
                self.validate_question_content(filepath, q)

        except Exception as e:
            self.errors.append(f"Error reading {filepath}: {str(e)}")

    def validate_question_structure(self, filepath, question):
        """Validate question JSON structure"""
        required_fields = ["id", "question_short", "answer_short", "tags"]

        for field in required_fields:
            if field not in question:
                self.errors.append(f"{filepath} missing required field: {field}")

        # Validate array fields
        array_fields = ["answer_long_md", "answer_long_html", "tags"]
        for field in array_fields:
            if field in question and not isinstance(question.get(field), list):
                self.errors.append(f"{filepath} field '{field}' should be an array")

    def validate_question_content(self, filepath, question):
        """Validate question content and referenced files"""
        # Check referenced files exist
        for md_file in question.get("answer_long_md", []):
            abs_path = self.root_dir / md_file.lstrip("/")
            if not abs_path.is_file():
                self.errors.append(
                    f"{filepath} references missing markdown file: {md_file}"
                )
            else:
                self.validate_markdown_file(abs_path)

        for html_file in question.get("answer_long_html", []):
            abs_path = self.root_dir / html_file.lstrip("/")
            if not abs_path.is_file():
                self.errors.append(
                    f"{filepath} references missing HTML file: {html_file}"
                )

        # Check image paths if they exist
        if "question_image" in question and question["question_image"]:
            img_path = self.root_dir / question["question_image"].lstrip("/")
            if not img_path.is_file():
                self.errors.append(
                    f"{filepath} references missing question image: {question['question_image']}"
                )

        if "answer_image" in question and question["answer_image"]:
            img_path = self.root_dir / question["answer_image"].lstrip("/")
            if not img_path.is_file():
                self.errors.append(
                    f"{filepath} references missing answer image: {question['answer_image']}"
                )

    def validate_markdown_file(self, filepath):
        """Basic markdown file validation"""
        try:
            with filepath.open("r") as f:
                content = f.read()
            markdown.markdown(content)  # Try to parse it
        except Exception as e:
            self.errors.append(f"Invalid markdown in {filepath}: {str(e)}")

    def validate_tag_coverage(self):
        """Verify all tags in tags.json are used in questions"""
        try:
            with (self.data_dir / "tags.json").open() as f:
                tags_data = json.load(f)

            all_tags = set(tags_data["tags"])
            found_tags = set()

            # Collect all tags from all questions
            question_files = list(self.data_dir.glob("*/*.json"))
            question_files = [f for f in question_files if f.name != "index.json"]

            for qn_file in question_files:
                with qn_file.open() as f:
                    data = json.load(f)
                questions = data if isinstance(data, list) else [data]
                for q in questions:
                    if "tags" in q:
                        found_tags.update(t.lower() for t in q["tags"])

            # Check for tags defined but not used
            unused_tags = set(t.lower() for t in all_tags) - found_tags
            for tag in unused_tags:
                self.warnings.append(
                    f"Tag '{tag}' is defined but not used in any question"
                )

            # Check for tags used but not defined
            undefined_tags = found_tags - set(t.lower() for t in all_tags)
            for tag in undefined_tags:
                self.warnings.append(
                    f"Tag '{tag}' is used in questions but not defined in tags.json"
                )

        except Exception as e:
            self.errors.append(f"Error validating tag coverage: {str(e)}")

    def validate_question_references(self):
        """Verify all questions are referenced in index files"""
        try:
            # Get all question files
            all_question_files = set()
            for qn_file in self.data_dir.glob("*/*.json"):
                if qn_file.name != "index.json":
                    rel_path = qn_file.relative_to(self.data_dir)
                    all_question_files.add(str(rel_path).replace("\\", "/"))

            # Get all referenced files from indexes
            referenced_files = set()
            for index_file in self.data_dir.glob("*/index.json"):
                with index_file.open() as f:
                    data = json.load(f)
                folder = index_file.parent.name
                for qn_file in data.get("files", []):
                    rel_path = f"{folder}/{qn_file}"
                    referenced_files.add(rel_path)

            # Find unreferenced questions
            unreferenced = all_question_files - referenced_files
            for qn_file in unreferenced:
                self.warnings.append(
                    f"Question file not referenced in any index.json: {qn_file}"
                )

        except Exception as e:
            self.errors.append(f"Error validating question references: {str(e)}")

    def report_results(self):
        """Print validation results"""
        print(
            f"\nValidation complete. Found {len(self.errors)} errors and {len(self.warnings)} warnings."
        )

        if self.errors:
            print("\n=== ERRORS ===")
            for error in self.errors:
                print(f"❌ {error}")

        if self.warnings:
            print("\n=== WARNINGS ===")
            for warning in self.warnings:
                print(f"⚠️ {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed successfully!")


if __name__ == "__main__":
    validator = QuizValidator()
    if not validator.validate_all():
        exit(1)
