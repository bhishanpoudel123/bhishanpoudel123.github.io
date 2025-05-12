let currentTag = "";
let currentQuestions = [];
let currentIndex = 0;
let correctCount = 0;
let userResponses = {};

// DOM Elements
const tagSelector = document.getElementById('tag-selector');
const startQuizBtn = document.getElementById('start-quiz');
const quizContainer = document.getElementById('quiz-container');
const scoreSection = document.getElementById('score-section');
const summaryElement = document.getElementById('summary');
const quizControls = document.querySelector('.quiz-controls');
const shuffleBtn = document.getElementById('shuffle');
const previousBtn = document.getElementById('previous');
const nextBtn = document.getElementById('next');
const endQuizBtn = document.getElementById('end-quiz');

// Initialize the quiz
document.addEventListener('DOMContentLoaded', () => {
	loadTags();
	setupEventListeners();
});

async function loadTags() {
	try {
		const res = await fetch('data/tags.json');
		const data = await res.json();

		// Add options to dropdown
		data.tags.forEach(tag => {
			const option = document.createElement('option');
			option.value = tag;
			option.textContent = tag;
			tagSelector.appendChild(option);
		});
	} catch (error) {
		console.error('Error loading tags:', error);
		summaryElement.textContent = "Error loading quiz topics. Please refresh the page.";
	}
}

function setupEventListeners() {
	tagSelector.addEventListener('change', () => {
		startQuizBtn.disabled = tagSelector.value === "";
	});

	startQuizBtn.addEventListener('click', startQuiz);
	shuffleBtn.addEventListener('click', shuffleQuestions);
	previousBtn.addEventListener('click', showPreviousQuestion);
	nextBtn.addEventListener('click', showNextQuestion);
	endQuizBtn.addEventListener('click', showSummary);
}

async function startQuiz() {
	const selectedTag = tagSelector.value;
	if (selectedTag) {
		await loadQuestions(selectedTag);
	}
}

async function loadQuestions(tag) {
	try {
		quizContainer.innerHTML = '';
		scoreSection.style.display = 'none';
		quizControls.style.display = 'flex';

		currentTag = tag;
		correctCount = 0;
		userResponses = {};
		currentIndex = 0;

		const files = ['questions/linear_regression.json']; // Add more files as needed
		let questions = [];

		for (let file of files) {
			const res = await fetch(file);
			const qns = await res.json();
			questions.push(...qns.filter(q => q.tags.includes(tag)));
		}

		currentQuestions = questions;

		if (questions.length === 0) {
			summaryElement.textContent = `No questions found for "${tag}". Please select another topic.`;
			quizControls.style.display = 'none';
			return;
		}

		summaryElement.textContent = `Selected topic: "${tag}" | ${questions.length} question${questions.length !== 1 ? 's' : ''}`;
		showQuestion();
	} catch (error) {
		console.error('Error loading questions:', error);
		summaryElement.textContent = "Error loading questions. Please try again.";
	}
}

function shuffleQuestions() {
	// Fisher-Yates shuffle algorithm
	for (let i = currentQuestions.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[currentQuestions[i], currentQuestions[j]] = [currentQuestions[j], currentQuestions[i]];
	}
	currentIndex = 0;
	showQuestion();
	summaryElement.textContent = `Selected topic: "${currentTag}" | ${currentQuestions.length} questions (shuffled)`;
}

function showPreviousQuestion() {
	if (currentIndex > 0) {
		currentIndex--;
		showQuestion();
	}
}

function showNextQuestion() {
	if (currentIndex < currentQuestions.length - 1) {
		currentIndex++;
		showQuestion();
	} else {
		showSummary();
	}
}


// Configure marked.js to preserve code blocks
marked.setOptions({
	highlight: function (code, lang) {
		if (Prism.languages[lang]) {
			return Prism.highlight(code, Prism.languages[lang], lang);
		}
		return code;
	},
	langPrefix: 'language-', // This matches Prism's class pattern
});

// Then modify your showQuestion() function to add line numbers:
async function showQuestion() {
	const q = currentQuestions[currentIndex];
	quizContainer.innerHTML = '';

	let questionLong = '';
	let answerLong = '';

	if (q.question_long_path) {
		questionLong = await fetchMarkdown(q.question_long_path);
	}

	if (q.answer_long_path) {
		answerLong = await fetchMarkdown(q.answer_long_path);
	}

	const div = document.createElement('div');
	div.className = 'question-block';

	div.innerHTML = `
        <strong>Q${currentIndex + 1}/${currentQuestions.length}: ${q.question_short}</strong><br>
        ${questionLong ? `<details><summary>Show Full Question</summary>
            <div class="markdown">${marked.parse(questionLong)}</div>
            ${q.question_image ? `<img src="${q.question_image}" width="300">` : ''}
        </details>` : ''}
        <details><summary>Show Answer</summary>
            <p><strong>Short Answer:</strong> ${q.answer_short}</p>
            ${answerLong ? `<div class="markdown">${marked.parse(answerLong)}</div>` : ''}
        </details>
        <div class="answer-buttons">
            <button onclick="markAnswer(true)">✅ Correct</button>
            <button onclick="markAnswer(false)">❌ Wrong</button>
        </div>
    `;

	quizContainer.appendChild(div);

	// Apply Prism highlighting to all code blocks
	const codeBlocks = div.querySelectorAll('pre code');
	codeBlocks.forEach((block) => {
		Prism.highlightElement(block);
		block.parentElement.classList.add('line-numbers');
	});

	updateNavButtons();
}


function updateNavButtons() {
	previousBtn.disabled = currentIndex === 0;
	nextBtn.disabled = currentIndex === currentQuestions.length - 1;
	nextBtn.textContent = currentIndex === currentQuestions.length - 1 ? "Finish Quiz ➡️" : "Next ➡️";
}

async function fetchMarkdown(path) {
	try {
		if (!path) return '';
		const res = await fetch(path.startsWith('content/') ? path : `content/${path}`);
		if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
		return await res.text();
	} catch (error) {
		console.error('Error loading markdown:', error);
		return '';
	}
}

function markAnswer(isCorrect) {
	userResponses[currentQuestions[currentIndex].id] = isCorrect;
	if (isCorrect) correctCount++;

	if (currentIndex < currentQuestions.length - 1) {
		currentIndex++;
		showQuestion();
	} else {
		showSummary();
	}
}

function showSummary() {
	quizContainer.innerHTML = '';
	quizControls.style.display = 'none';
	scoreSection.style.display = 'block';

	const total = currentQuestions.length;
	const percentage = Math.round((correctCount / total) * 100);

	scoreSection.innerHTML = `
        <h2>Quiz Complete!</h2>
        <p>You answered ${correctCount} out of ${total} correctly (${percentage}%).</p>
        <button onclick="showReview()">🔍 Review Answers</button>
        <button onclick="location.reload()">🔄 Start New Quiz</button>
    `;
}

function showReview() {
	quizContainer.innerHTML = '<h3>📝 Quiz Review</h3>';

	currentQuestions.forEach((q, i) => {
		const div = document.createElement('div');
		div.className = 'question-block review';
		const correct = userResponses[q.id];

		div.innerHTML = `
            <strong>Q${i + 1}: ${q.question_short}</strong>
            <p class="answer"><strong>Answer:</strong> ${q.answer_short}</p>
            <p class="${correct ? 'correct' : 'incorrect'}">
                You marked this as ${correct ? 'correct ✅' : 'incorrect ❌'}
            </p>
        `;

		quizContainer.appendChild(div);
	});
}