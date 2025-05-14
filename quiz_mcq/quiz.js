// Quiz State Management
const quizState = {
	questions: [],
	filteredQuestions: [],
	currentIndex: 0,
	userAnswers: {},
	showAnswer: false,
	categories: [],
	timerInterval: null,
	quizDuration: 20 * 60, // Default 20 minutes in seconds
	timeLeft: 20 * 60,
	startTime: null,
	endTime: null,
	markedInstance: marked.marked.setOptions({
		highlight: (code, lang) => hljs.highlightAuto(code).value
	}),
	quizStarted: false
};

// DOM Elements
const dom = {
	categorySelector: document.getElementById('category-selector'),
	durationSelector: document.getElementById('duration-selector'),
	quizContainer: document.getElementById('quiz-container'),
	questionTitle: document.getElementById('question-title'),
	questionText: document.getElementById('question-text'),
	questionImages: document.getElementById('question-images'),
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
	await loadQuestions();
	setupEventListeners();
	populateDurationOptions();
});

// Load Questions from JSON Files
async function loadQuestions() {
	try {
		const indexResponse = await fetch('data/index.json');
		if (!indexResponse.ok) throw new Error('Failed to load question index');

		const indexData = await indexResponse.json();
		const questionFiles = indexData.files;

		const questionPromises = questionFiles.map(file =>
			fetch(`data/${file}`).then(res => res.json())
		);

		const questions = await Promise.all(questionPromises);
		quizState.questions = questions.flat();
		quizState.categories = [...new Set(quizState.questions.map(q => q.category))];

		populateCategories();
	} catch (error) {
		console.error('Error loading questions:', error);
		dom.questionText.innerHTML = `<div class="error">Failed to load questions. Please refresh the page.</div>`;
	}
}

// Timer Functions
function initTimer() {
	const durationMinutes = parseInt(dom.durationSelector.value);
	quizState.quizDuration = durationMinutes * 60;
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

	const minutes = Math.floor(quizState.timeLeft / 60);
	const seconds = quizState.timeLeft % 60;
	dom.timerDisplay.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

	if (quizState.timeLeft <= 0) {
		endQuiz();
	}
}

// Question Display
function showQuestion() {
	const question = quizState.filteredQuestions[quizState.currentIndex];
	if (!question) return;

	dom.questionTitle.textContent = `Question ${question.id} - ${question.category}`;
	dom.questionText.innerHTML = renderMarkdown(question.question);
	dom.optionsContainer.innerHTML = '';

	question.options.forEach((option, index) => {
		const optionContainer = document.createElement('div');
		optionContainer.className = 'option-container';
		optionContainer.innerHTML = renderMarkdown(option);
		optionContainer.addEventListener('click', () => selectOption(question.id, option));

		if (quizState.userAnswers[question.id] === option) {
			optionContainer.classList.add('selected');
		}

		dom.optionsContainer.appendChild(optionContainer);
	});

	dom.questionJump.value = question.id;
	updateNavigationButtons();
	updateAnswerDisplay(question);
}

function updateAnswerDisplay(question) {
	if (quizState.showAnswer) {
		dom.answerSection.style.display = 'block';
		dom.correctAnswer.innerHTML = renderMarkdown(`**Correct Answer:** ${question.answer}`);
		dom.explanation.innerHTML = renderMarkdown(`**Explanation:** ${question.explanation}`);
	} else {
		dom.answerSection.style.display = 'none';
	}
}

// Answer Handling
function selectOption(questionId, option) {
	quizState.userAnswers[questionId] = option;
	showQuestion();
}

// Navigation Controls
function updateNavigationButtons() {
	dom.previousBtn.disabled = quizState.currentIndex === 0;
	dom.nextBtn.disabled = quizState.currentIndex === quizState.filteredQuestions.length - 1;
}

function jumpToQuestion(id) {
	const index = quizState.filteredQuestions.findIndex(q => q.id === id);
	if (index >= 0) {
		quizState.currentIndex = index;
		quizState.showAnswer = false;
		showQuestion();
	} else {
		alert('Question not found in current category!');
	}
}

// Quiz Control Functions
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
	console.log("Ending quiz...");
	clearInterval(quizState.timerInterval);

	// Debug: Verify we have questions and answers
	console.log("Filtered questions:", quizState.filteredQuestions);
	console.log("User answers:", quizState.userAnswers);

	if (!quizState.filteredQuestions || quizState.filteredQuestions.length === 0) {
		console.error("No questions available");
		alert('No questions available! Please select a category and start the quiz first.');
		return;
	}

	// Calculate scores
	const total = quizState.filteredQuestions.length;
	const correct = quizState.filteredQuestions.filter(q =>
		quizState.userAnswers[q.id] === q.answer
	).length;
	const unanswered = quizState.filteredQuestions.filter(q =>
		!quizState.userAnswers.hasOwnProperty(q.id)
	).length;
	const incorrect = total - correct - unanswered;

	// Generate the summary HTML
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
		const userAnswer = quizState.userAnswers[q.id];
		const isCorrect = userAnswer === q.answer;
		const status = !userAnswer ? 'unanswered' : isCorrect ? 'correct' : 'incorrect';

		return `
                    <div class="review-question ${status}">
                        <h3>Question ${q.id} - ${q.category}</h3>
                        <div class="question">${renderMarkdown(q.question)}</div>
                        ${userAnswer ?
				`<div class="user-answer">
                                <span class="status">Your answer:</span> 
                                ${renderMarkdown(userAnswer)}
                                <span class="result-icon">${isCorrect ? '✓' : '✗'}</span>
                            </div>` :
				'<div class="unanswered">Not answered</div>'
			}
                        <div class="correct-answer">
                            <span class="status">Correct answer:</span> 
                            ${renderMarkdown(q.answer)}
                        </div>
                        <div class="explanation">
                            ${renderMarkdown(q.explanation)}
                        </div>
                    </div>
                `;
	}).join('')}
        </div>
    `;

	// Debug: Check DOM elements before updating
	console.log("Quiz container:", dom.quizContainer);
	console.log("Score section:", dom.scoreSection);

	// Update DOM - CRITICAL FIXES:
	// 1. Make sure we're working with the correct elements
	// 2. Force reflow/repaint if needed
	if (dom.quizContainer && dom.scoreSection) {
		// First hide the quiz container
		dom.quizContainer.style.display = 'none';

		// Then update and show the score section
		dom.scoreSection.innerHTML = summaryHTML;
		dom.scoreSection.style.display = 'block';

		// Force a reflow to ensure the display change takes effect
		void dom.scoreSection.offsetHeight;

		// Add restart button listener
		const restartBtn = document.getElementById('restart-btn');
		if (restartBtn) {
			restartBtn.addEventListener('click', resetQuiz);
		}

		// Apply syntax highlighting if available
		if (typeof hljs !== 'undefined') {
			hljs.highlightAll();
		}

		// Scroll to top
		window.scrollTo({ top: 0, behavior: 'smooth' });
		console.log("Quiz ended successfully - DOM updated");
	} else {
		console.error("Critical DOM elements missing!");
		alert("Error: Could not display quiz results. Please refresh the page.");
	}
}

function filterQuestions() {
	const selectedCategory = dom.categorySelector.value;
	quizState.filteredQuestions = selectedCategory === 'All'
		? [...quizState.questions]
		: quizState.questions.filter(q => q.category === selectedCategory);

	quizState.currentIndex = 0;
	quizState.userAnswers = {};
	if (quizState.quizStarted) {
		showQuestion();
	}
}

// Helper Functions
function renderMarkdown(content) {
	return quizState.markedInstance.parse(content);
}

function populateCategories() {
	quizState.categories.forEach(category => {
		const option = document.createElement('option');
		option.value = category;
		option.textContent = category;
		dom.categorySelector.appendChild(option);
	});
}

function populateDurationOptions() {
	const durations = [
		{ value: 5, text: '5 minutes' },
		{ value: 10, text: '10 minutes' },
		{ value: 15, text: '15 minutes' },
		{ value: 20, text: '20 minutes' },
		{ value: 30, text: '30 minutes' },
		{ value: 45, text: '45 minutes' },
		{ value: 60, text: '1 hour' }
	];

	durations.forEach(duration => {
		const option = document.createElement('option');
		option.value = duration.value;
		option.textContent = duration.text;
		dom.durationSelector.appendChild(option);
	});

	dom.durationSelector.value = 20;
}

// Event Listeners
function setupEventListeners() {
	dom.categorySelector.addEventListener('change', filterQuestions);
	dom.durationSelector.addEventListener('change', () => {
		if (quizState.timerInterval) {
			initTimer();
		}
	});
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
	dom.jumpButton.addEventListener('click', () => {
		const targetId = Number(dom.questionJump.value);
		jumpToQuestion(targetId);
	});
	dom.questionJump.addEventListener('keypress', (e) => {
		if (e.key === 'Enter') {
			const targetId = Number(dom.questionJump.value);
			jumpToQuestion(targetId);
		}
	});
}