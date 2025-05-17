import json

topic = "Modelling"
tp = topic.lower().replace(" ", "_")
fname = f'{tp}_questions.json'
ofile = f'{tp}_questions_v2.json'

# Load the original JSON file
with open(fname, 'r') as file:
    questions = json.load(file)

# Update each question with the answer markdown path
for question in questions:
    question_id = question["id"]
    question["answer_long_md"] = [f"data/{topic}/qn_{question_id:02}_long_answer_01.md"]

# Save the modified data to a new JSON file
with open(ofile, 'w') as file:
    json.dump(questions, file, indent=4)

print(f"Updated JSON file saved as {ofile}")
