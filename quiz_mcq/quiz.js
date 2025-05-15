// Quiz State Management
const quizState = {
	questions: [],
	filteredQuestions: [],
	currentIndex: 0,
	userAnswers: {},
	showAnswer: false,
	categories: [],
	timerInterval: null,
	quizDuration: 20 * 60,
	timeLeft: 20 * 60,
	startTime: null,
	endTime: null,
	quizStarted: false,
	indexMap: {},
	markedInstance: marked.marked.setOptions({
		highlight: (code, lang) => hljs.highlightAuto(code).value
	})
};

// DOM Elements
const dom = {
	categorySelector: document.getElementById('category-selector'),
	durationSelector: document.getElementById('duration-selector'),
	quizContainer: document.getElementById('quiz-container'),
	questionTitle: document.getElementById('question-title'),
	questionText: document.getElementById('question-text'),
	optionsContainer: document.getElementById('options-container'),
	answerSection: document.getElementById('answer-section'),
	correctAnswer: document.getElementById('correct-answer'),
	explanation: document.getElementById('explanation'),
	timerDisplay: document.getElementById('quiz-timer'),
	startTime: document.getElementById('start-time'),
	endTime: document.getElementById('end-time'),
	questionJump: document.getElementById('question-jump'),
	previousBtn: document.getElementById('previous'),
	nextBtn: document.getElementById('next'),
	showAnswerBtn: document.getElementById('show-answer'),
	restartBtn: document.getElementById('restart-quiz'),
	endBtn: document.getElementById('end-quiz'),
	scoreSection: document.getElementById('score-section'),
	jumpButton: document.getElementById('jump-button'),
	startQuizBtn: document.getElementById('start-quiz')
};

// Initialize Quiz
document.addEventListener('DOMContentLoaded', async () => {
	await loadIndexJson();
	populateDurationOptions();
	setupEventListeners();
});

// Load index.json and fetch questions
async function loadIndexJson() {
	try {
		const response = await fetch('data/index.json');
		if (!response.ok) throw new Error('Failed to fetch index.json');
		const data = await response.json();
		quizState.indexMap = data.files;
		await loadAllQuestions(data.files);
	} catch (err) {
		console.error('Error loading index.json:', err);
		dom.questionText.innerHTML = `<div class="error">Error loading index.json</div>`;
	}
}

async function loadAllQuestions(files) {
	const allQuestions = [];
	const categories = [];

	for (const [category, filename] of Object.entries(files)) {
		const categoryFolder = category.replace(/\s+/g, '_');
		const path = `data/${categoryFolder}/${filename}`;

		try {
			const response = await fetch(path);
			if (!response.ok) throw new Error(`Failed to load ${path}`);
			const questions = await response.json();
			allQuestions.push(...questions);
			categories.push(category);
		} catch (err) {
			console.warn(`Skipped ${path}:`, err.message);
		}
	}

	quizState.questions = allQuestions;
	quizState.categories = [...new Set(categories)];
	populateCategories();
}

// Category dropdown
function populateCategories() {
	quizState.categories.forEach(cat => {
		const opt = document.createElement('option');
		opt.value = cat;
		opt.textContent = cat;
		dom.categorySelector.appendChild(opt);
	});
}

// Duration dropdown
function populateDurationOptions() {
	[5, 10, 15, 20, 30, 45, 60].forEach(min => {
		const opt = document.createElement('option');
		opt.value = min;
		opt.textContent = `${min} minutes`;
		dom.durationSelector.appendChild(opt);
	});
	dom.durationSelector.value = 20;
}

// Timer
function initTimer() {
	const mins = parseInt(dom.durationSelector.value);
	quizState.quizDuration = mins * 60;
	quizState.timeLeft = quizState.quizDuration;
	quizState.startTime = Date.now();
	quizState.endTime = quizState.startTime + quizState.quizDuration * 1000;

	clearInterval(quizState.timerInterval);
	quizState.timerInterval = setInterval(updateTimer, 1000);
	quizState.quizStarted = true;

	dom.startTime.textContent = new Date(quizState.startTime).toLocaleTimeString();
	dom.endTime.textContent = new Date(quizState.endTime).toLocaleTimeString();
	updateTimer();
}

function updateTimer() {
	const now = Date.now();
	quizState.timeLeft = Math.max(0, Math.floor((quizState.endTime - now) / 1000));
	const min = Math.floor(quizState.timeLeft / 60);
	const sec = quizState.timeLeft % 60;
	dom.timerDisplay.textContent = `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
	if (quizState.timeLeft <= 0) endQuiz();
}

// Quiz Controls
function resetQuiz() {
	clearInterval(quizState.timerInterval);
	quizState.currentIndex = 0;
	quizState.userAnswers = {};
	quizState.showAnswer = false;
	quizState.quizStarted = false;
	dom.timerDisplay.textContent = '--:--';
	dom.startTime.textContent = '-';
	dom.endTime.textContent = '-';
	filterQuestions();
	dom.scoreSection.style.display = 'none';
	dom.quizContainer.style.display = 'block';
}

function endQuiz() {
	clearInterval(quizState.timerInterval);

	const total = quizState.filteredQuestions.length;
	const correct = quizState.filteredQuestions.filter(q => quizState.userAnswers[q.id] === q.answer).length;
	const unanswered = quizState.filteredQuestions.filter(q => !quizState.userAnswers.hasOwnProperty(q.id)).length;
	const incorrect = total - correct - unanswered;

	const summaryHTML = `
		<div class="score-summary">
			<h2>Quiz Summary</h2>
			<div class="score">${correct}/${total} Correct (${Math.round((correct / total) * 100)}%)</div>
			<div class="stats">
				<div class="stat correct-stat">✓ Correct: ${correct}</div>
				<div class="stat incorrect-stat">✗ Incorrect: ${incorrect}</div>
				<div class="stat unanswered-stat">? Unanswered: ${unanswered}</div>
			</div>
			<div class="progress-bar">
				<div class="progress-fill" style="width: ${(correct / total) * 100}%"></div>
			</div>
			<button id="restart-btn" class="restart-btn">Restart Quiz</button>
		</div>
		<div class="question-reviews">
			<h3>Question Review</h3>
			${quizState.filteredQuestions.map(q => {
		const ua = quizState.userAnswers[q.id];
		const correct = ua === q.answer;
		const status = !ua ? 'unanswered' : correct ? 'correct' : 'incorrect';
		return `
				<div class="review-question ${status}">
					<h3>Question ${q.id} - ${q.category}</h3>
					<div class="question">${renderMarkdown(q.question)}</div>
					${ua ? `<div class="user-answer"><strong>Your answer:</strong> ${renderMarkdown(ua)} <span>${correct ? '✓' : '✗'}</span></div>` : '<div class="unanswered">Not answered</div>'}
					<div class="correct-answer"><strong>Correct:</strong> ${renderMarkdown(q.answer)}</div>
					<div class="explanation">${renderMarkdown(q.explanation)}</div>
				</div>`;
	}).join('')}
		</div>
	`;

	dom.quizContainer.style.display = 'none';
	dom.scoreSection.innerHTML = summaryHTML;
	dom.scoreSection.style.display = 'block';
	document.getElementById('restart-btn')?.addEventListener('click', resetQuiz);
	window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Question Display
function showQuestion() {
	const question = quizState.filteredQuestions[quizState.currentIndex];
	if (!question) return;

	dom.questionTitle.textContent = `Question ${question.id} - ${question.category}`;
	dom.questionText.innerHTML = renderMarkdown(question.question);
	dom.optionsContainer.innerHTML = '';

	question.options.forEach(option => {
		const div = document.createElement('div');
		div.className = 'option-container';
		div.innerHTML = renderMarkdown(option);
		div.addEventListener('click', () => selectOption(question.id, option));
		if (quizState.userAnswers[question.id] === option) div.classList.add('selected');
		dom.optionsContainer.appendChild(div);
	});

	dom.questionJump.value = question.id;
	updateNavigationButtons();
	updateAnswerDisplay(question);
}

function selectOption(id, option) {
	quizState.userAnswers[id] = option;
	showQuestion();
}

function renderMarkdown(content) {
	return quizState.markedInstance.parse(content);
}

function updateNavigationButtons() {
	dom.previousBtn.disabled = quizState.currentIndex === 0;
	dom.nextBtn.disabled = quizState.currentIndex >= quizState.filteredQuestions.length - 1;
}

function updateAnswerDisplay(question) {
	if (quizState.showAnswer) {
		dom.answerSection.style.display = 'block';
		dom.correctAnswer.innerHTML = renderMarkdown(`**Answer:** ${question.answer}`);
		dom.explanation.innerHTML = renderMarkdown(`**Explanation:** ${question.explanation || ''}`);

		// Load long markdown explanation
		if (question.answer_long_md && question.answer_long_md.length > 0) {
			question.answer_long_md.forEach(mdPath => {
				fetch(mdPath)
					.then(res => res.text())
					.then(md => {
						const html = renderMarkdown(md);
						const div = document.createElement('div');
						div.classList.add('long-answer-md');
						div.innerHTML = html;
						dom.explanation.appendChild(div);
						if (typeof hljs !== 'undefined') hljs.highlightAll();
					})
					.catch(err => console.warn('Markdown file load failed:', mdPath, err));
			});
		}

		// Load long HTML explanation
		if (question.answer_long_html && question.answer_long_html.length > 0) {
			question.answer_long_html.forEach(htmlPath => {
				fetch(htmlPath)
					.then(res => res.text())
					.then(html => {
						const div = document.createElement('div');
						div.classList.add('long-answer-html');
						div.innerHTML = html;
						dom.explanation.appendChild(div);
						if (typeof hljs !== 'undefined') hljs.highlightAll();
					})
					.catch(err => console.warn('HTML file load failed:', htmlPath, err));
			});
		}

	} else {
		dom.answerSection.style.display = 'none';
	}
}


function jumpToQuestion(id) {
	const index = quizState.filteredQuestions.findIndex(q => q.id === id);
	if (index >= 0) {
		quizState.currentIndex = index;
		quizState.showAnswer = false;
		showQuestion();
	}
}

function filterQuestions() {
	const selectedCategory = dom.categorySelector.value;
	quizState.filteredQuestions = quizState.questions.filter(q => q.category === selectedCategory);
	quizState.currentIndex = 0;
	quizState.userAnswers = {};
	showQuestion();
}

// Event Listeners
function setupEventListeners() {
	dom.categorySelector.addEventListener('change', filterQuestions);
	dom.durationSelector.addEventListener('change', () => quizState.timerInterval && initTimer());
	dom.startQuizBtn.addEventListener('click', () => {
		initTimer();
		filterQuestions();
	});
	dom.previousBtn.addEventListener('click', () => {
		if (quizState.currentIndex > 0) {
			quizState.currentIndex--;
			quizState.showAnswer = false;
			showQuestion();
		}
	});
	dom.nextBtn.addEventListener('click', () => {
		if (quizState.currentIndex < quizState.filteredQuestions.length - 1) {
			quizState.currentIndex++;
			quizState.showAnswer = false;
			showQuestion();
		}
	});
	dom.showAnswerBtn.addEventListener('click', () => {
		quizState.showAnswer = !quizState.showAnswer;
		showQuestion();
	});
	dom.restartBtn.addEventListener('click', resetQuiz);
	dom.endBtn.addEventListener('click', endQuiz);
	dom.jumpButton.addEventListener('click', () => jumpToQuestion(Number(dom.questionJump.value)));
	dom.questionJump.addEventListener('keypress', e => {
		if (e.key === 'Enter') jumpToQuestion(Number(dom.questionJump.value));
	});
}
