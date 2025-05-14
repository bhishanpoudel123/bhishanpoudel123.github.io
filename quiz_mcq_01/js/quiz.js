// DOM Elements - Add quizControls here
const quizControls = document.querySelector('.quiz-controls');
const categorySelector = document.getElementById('category-selector');
const quizContainer = document.getElementById('quiz-container');
const questionTitle = document.getElementById('question-title');
const questionText = document.getElementById('question-text');
const questionImages = document.getElementById('question-images');
const optionsContainer = document.getElementById('options-container');
const answerSection = document.getElementById('answer-section');
const correctAnswer = document.getElementById('correct-answer');
const progressSpan = document.getElementById('progress');
const previousBtn = document.getElementById('previous');
const showAnswerBtn = document.getElementById('show-answer');
const nextBtn = document.getElementById('next');
const finishQuizBtn = document.getElementById('finish-quiz');
const resetBtn = document.getElementById('reset');
const scoreSection = document.getElementById('score-section');

// Quiz state variables
const quizState = {
	questions: [],
	filteredQuestions: [],
	currentIndex: 0,
	userAnswers: {},
	showAnswer: false,
	categories: [],
	score: 0
};

// Initialize the quiz
document.addEventListener('DOMContentLoaded', () => {
	loadQuestions();
	setupEventListeners();
});

async function loadQuestions() {
	try {
		const indexResponse = await fetch('data/questions/index.json');
		if (!indexResponse.ok) throw new Error('Failed to load question index');

		const indexData = await indexResponse.json();
		const questionFiles = indexData.files;

		const questionPromises = questionFiles.map(file =>
			fetch(`data/questions/${file}`).then(res => res.json())
		);

		const questions = await Promise.all(questionPromises);
		quizState.questions = questions.flat();

		quizState.categories = [...new Set(quizState.questions.map(q => q.category))];

		quizState.categories.forEach(category => {
			const option = document.createElement('option');
			option.value = category;
			option.textContent = category;
			categorySelector.appendChild(option);
		});

		filterQuestions();
		updateProgress();
	} catch (error) {
		console.error('Error loading questions:', error);
		questionText.textContent = "Error loading questions. Please refresh the page.";
	}
}

function filterQuestions() {
	const selectedCategory = categorySelector.value;

	if (selectedCategory === 'All') {
		quizState.filteredQuestions = [...quizState.questions];
	} else {
		quizState.filteredQuestions = quizState.questions.filter(
			q => q.category === selectedCategory
		);
	}

	quizState.currentIndex = 0;
	showQuestion();
}

function showQuestion() {
	if (quizState.filteredQuestions.length === 0) {
		questionTitle.textContent = "No questions found for selected category";
		questionText.textContent = "";
		optionsContainer.innerHTML = "";
		questionImages.innerHTML = "";
		return;
	}

	const question = quizState.filteredQuestions[quizState.currentIndex];
	questionTitle.textContent = `Question ${question.id} - ${question.category} (${question.metadata.difficulty})`;
	questionText.textContent = question.question;
	optionsContainer.innerHTML = "";
	questionImages.innerHTML = "";

	if (question.metadata.images) {
		question.metadata.images.forEach(img => {
			const imgElement = document.createElement('img');
			imgElement.src = `data/questions/images/${img}`;
			imgElement.className = 'quiz-image';
			imgElement.alt = "Question illustration";
			questionImages.appendChild(imgElement);
		});
	}

	if (question.metadata.type === "MCQ") {
		question.options.forEach((option, index) => {
			const optionId = `option_${question.id}_${index}`;
			const optionContainer = document.createElement('div');
			optionContainer.className = 'option-container';

			const radioInput = document.createElement('input');
			radioInput.type = 'radio';
			radioInput.id = optionId;
			radioInput.name = `question_${question.id}`;
			radioInput.value = option;
			radioInput.className = 'option-radio';

			if (quizState.userAnswers[question.id] === option) {
				radioInput.checked = true;
			}

			radioInput.addEventListener('change', () => {
				quizState.userAnswers[question.id] = option;
				updateProgress();
			});

			const label = document.createElement('label');
			label.htmlFor = optionId;
			label.textContent = option;

			optionContainer.appendChild(radioInput);
			optionContainer.appendChild(label);
			optionsContainer.appendChild(optionContainer);
		});
	}

	answerSection.style.display = 'none';
	quizState.showAnswer = false;
	showAnswerBtn.textContent = "Show Answer";
	previousBtn.disabled = quizState.currentIndex === 0;
	nextBtn.disabled = quizState.currentIndex === quizState.filteredQuestions.length - 1;
}

function showCorrectAnswer() {
	const question = quizState.filteredQuestions[quizState.currentIndex];
	answerSection.style.display = 'block';
	correctAnswer.innerHTML = "";

	question.answers.forEach(answer => {
		if (typeof answer === 'string') {
			const answerElement = document.createElement('div');
			answerElement.className = 'correct-answer';

			if (question.metadata.answer_format === "code") {
				const codeElement = document.createElement('pre');
				codeElement.textContent = answer;
				answerElement.appendChild(codeElement);
			} else {
				answerElement.textContent = answer;
			}

			correctAnswer.appendChild(answerElement);
		}
	});
}

function calculateScore() {
	quizState.score = 0;
	quizState.filteredQuestions.forEach(question => {
		const userAnswer = quizState.userAnswers[question.id];
		if (userAnswer && question.answers.includes(userAnswer)) {
			quizState.score++;
		}
	});
	return quizState.score;
}

function showResults() {
	const score = calculateScore();
	const total = quizState.filteredQuestions.length;
	const percentage = Math.round((score / total) * 100);

	// Hide quiz interface
	quizContainer.style.display = 'none';
	answerSection.style.display = 'none';
	if (quizControls) quizControls.style.display = 'none';

	// Create comprehensive results view
	scoreSection.style.display = 'block';
	scoreSection.innerHTML = `
        <h2>Quiz Results</h2>
        <div class="score-summary">
            <p>You scored <strong>${score}/${total}</strong> (${percentage}%)</p>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${percentage}%"></div>
            </div>
        </div>
        <div id="all-questions-review"></div>
        <div class="result-buttons">
            <button id="restart-quiz">Restart Quiz</button>
        </div>
    `;

	// Add detailed question review
	const reviewContainer = document.getElementById('all-questions-review');
	quizState.filteredQuestions.forEach((question, index) => {
		const userAnswer = quizState.userAnswers[question.id];
		const isCorrect = userAnswer && question.answers.includes(userAnswer);

		const questionElement = document.createElement('div');
		questionElement.className = `review-question ${isCorrect ? 'correct' : 'incorrect'}`;

		questionElement.innerHTML = `
            <h3>Question ${index + 1}: ${question.question}</h3>
            <div class="user-answer">
                <strong>Your answer:</strong> 
                ${userAnswer ? userAnswer : 'Not answered'}
                ${isCorrect ? '✅' : '❌'}
            </div>
            <div class="correct-answers">
                <strong>Correct answer(s):</strong>
                ${question.answers.map(ans =>
			`<div class="correct-answer">${ans}</div>`
		).join('')}
            </div>
            <hr>
        `;

		reviewContainer.appendChild(questionElement);
	});

	document.getElementById('restart-quiz').addEventListener('click', () => {
		location.reload();
	});
}

function reviewAnswers() {
	quizContainer.style.display = 'block';
	quizControls.style.display = 'flex';
	scoreSection.style.display = 'none';
	quizState.currentIndex = 0;
	showQuestion();
	showCorrectAnswer();
}

function updateProgress() {
	const total = quizState.filteredQuestions.length;
	const answered = Object.keys(quizState.userAnswers).length;
	progressSpan.textContent = `Progress: ${answered}/${total} answered`;
}

function setupEventListeners() {
	categorySelector.addEventListener('change', () => {
		filterQuestions();
		updateProgress();
		finishQuizBtn.addEventListener('click', showResults);
	});

	previousBtn.addEventListener('click', () => {
		if (quizState.currentIndex > 0) {
			quizState.currentIndex--;
			showQuestion();
		}
	});

	nextBtn.addEventListener('click', () => {
		if (quizState.currentIndex < quizState.filteredQuestions.length - 1) {
			quizState.currentIndex++;
			showQuestion();
		} else {
			showResults();
		}
	});

	showAnswerBtn.addEventListener('click', () => {
		quizState.showAnswer = !quizState.showAnswer;
		if (quizState.showAnswer) {
			showCorrectAnswer();
			showAnswerBtn.textContent = "Hide Answer";
		} else {
			answerSection.style.display = 'none';
			showAnswerBtn.textContent = "Show Answer";
		}
	});

	resetBtn.addEventListener('click', () => {
		quizState.userAnswers = {};
		updateProgress();
		showQuestion();
	});
}