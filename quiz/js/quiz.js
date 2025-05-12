let currentTag = "";
let currentQuestions = [];
let currentIndex = 0;
let correctCount = 0;
let userResponses = {};

async function loadTags() {
	const res = await fetch('data/tags.json');
	const data = await res.json();
	const container = document.getElementById('tag-selector');
	data.tags.forEach(tag => {
		const btn = document.createElement('button');
		btn.textContent = tag;
		btn.onclick = () => loadQuestions(tag);
		container.appendChild(btn);
	});
}

async function loadQuestions(tag) {
	document.getElementById('quiz-container').innerHTML = '';
	document.getElementById('score-section').style.display = 'none';
	currentTag = tag;
	correctCount = 0;
	userResponses = {};

	const files = ['linear_regression.json', 'machine_learning.json'];
	let questions = [];

	for (let file of files) {
		const res = await fetch(`questions/${file}`);
		const qns = await res.json();
		qns.forEach(q => {
			if (q.tags.includes(tag)) {
				questions.push(q);
			}
		});
	}

	currentQuestions = questions;
	currentIndex = 0;

	document.getElementById('summary').textContent = `Loaded ${questions.length} questions for tag "${tag}"`;
	showQuestion();
}

async function fetchMarkdown(path) {
	const res = await fetch(path);
	return await res.text();
}

async function showQuestion() {
	const q = currentQuestions[currentIndex];
	const container = document.getElementById('quiz-container');
	container.innerHTML = '';

    let questionLong = '';
    let answerLong = '';

    if (q.question_long_path) {
        try {
            questionLong = await fetchMarkdown(`content/${q.question_long_path}`);
        } catch (error) {
            console.warn(`Could not load question markdown: ${error}`);
        }
    }

    if (q.answer_long_path) {
        try {
            answerLong = await fetchMarkdown(`content/${q.answer_long_path}`);
        } catch (error) {
            console.warn(`Could not load answer markdown: ${error}`);
        }
    }

	const div = document.createElement('div');
	div.className = 'question-block';

	div.innerHTML = `
    <strong>Q${currentIndex + 1}: ${q.question_short}</strong><br>
    <details><summary>Show Full Question</summary>
      <div class="markdown">${marked.parse(questionLong)}</div>
      ${q.question_image ? `<img src="${q.question_image}" width="300">` : ''}
    </details>
    <details><summary>Show Answer</summary>
      <p><strong>Short:</strong> ${q.answer_short}</p>
      <div class="markdown">${marked.parse(answerLong)}</div>
    </details>
    <button onclick="markAnswer(true)">✔️ Correct</button>
    <button onclick="markAnswer(false)">❌ Wrong</button>
  `;

	container.appendChild(div);
}

function markAnswer(isCorrect) {
	userResponses[currentQuestions[currentIndex].id] = isCorrect;
	if (isCorrect) correctCount++;
	currentIndex++;
	if (currentIndex < currentQuestions.length) {
		showQuestion();
	} else {
		showSummary();
	}
}

function showSummary() {
	const container = document.getElementById('score-section');
	container.style.display = 'block';

	const total = currentQuestions.length;
	container.innerHTML = `
    <h2>Test Complete</h2>
    <p>You answered ${correctCount} out of ${total} correctly.</p>
    <button onclick="showReview()">🔍 Review</button>
  `;
}

function showReview() {
	const container = document.getElementById('quiz-container');
	container.innerHTML = '<h3>📘 Review</h3>';

	currentQuestions.forEach((q, i) => {
		const div = document.createElement('div');
		div.className = 'question-block';
		const correct = userResponses[q.id];
		div.innerHTML = `
      <strong>Q${i + 1}: ${q.question_short}</strong><br>
      <p>${q.answer_short}</p>
      <p class="${correct ? 'correct' : 'incorrect'}">
        You were ${correct ? 'Correct ✅' : 'Wrong ❌'}
      </p>
    `;
		container.appendChild(div);
	});
}

loadTags();
