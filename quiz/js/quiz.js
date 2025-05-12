// Add this at the top of quiz.js
if (typeof marked === 'undefined') {
    console.error('marked.js is not loaded. Please include it before quiz.js');
    // Fallback function if marked is not available
    window.parseMarkdown = function(text) { return text; };
} else {
    marked.setOptions({
        highlight: function(code, lang) {
            if (typeof Prism !== 'undefined' && Prism.languages[lang]) {
                return Prism.highlight(code, Prism.languages[lang], lang);
            }
            return code;
        },
        langPrefix: 'language-'
    });
    window.parseMarkdown = marked.parse;
}

// Quiz state variables
const quizState = {
    currentTag: "",
    currentQuestions: [],
    currentIndex: 0,
    correctCount: 0,
    userResponses: {}
};

// DOM Elements
const tagSelector = document.getElementById('tag-selector');
const startQuizBtn = document.getElementById('start-quiz');
const quizContainer = document.getElementById('quiz-container');
const scoreSection = document.getElementById('score-section');
const summaryElement = document.getElementById('summary');
const quizControls = document.querySelector('.quiz-controls');
const questionNavigation = document.querySelector('.question-navigation');
const questionSelector = document.getElementById('question-selector');
const goToQuestionBtn = document.getElementById('go-to-question');
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
    goToQuestionBtn.addEventListener('click', goToSelectedQuestion);
    questionSelector.addEventListener('change', function() {
        goToQuestionBtn.disabled = this.value === "";
    });
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
        questionNavigation.style.display = 'flex';

        // Reset quiz state
        quizState.currentTag = tag;
        quizState.correctCount = 0;
        quizState.userResponses = {};
        quizState.currentIndex = 0;

        const files = ['questions/linear_regression.json'];
        let questions = [];

        for (let file of files) {
            const res = await fetch(file);
            const qns = await res.json();
            questions.push(...qns.filter(q => q.tags.includes(tag)));
        }

        quizState.currentQuestions = questions;

        if (questions.length === 0) {
            summaryElement.textContent = `No questions found for "${tag}". Please select another topic.`;
            quizControls.style.display = 'none';
            questionNavigation.style.display = 'none';
            return;
        }

        summaryElement.textContent = `Selected topic: "${tag}" | ${questions.length} question${questions.length !== 1 ? 's' : ''}`;
        setupQuestionNavigation();
        showQuestion();
    } catch (error) {
        console.error('Error loading questions:', error);
        summaryElement.textContent = "Error loading questions. Please try again.";
    }
}

function setupQuestionNavigation() {
    questionSelector.innerHTML = '';
    quizState.currentQuestions.forEach((_, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `Q${index + 1}`;
        questionSelector.appendChild(option);
    });
    questionSelector.value = quizState.currentIndex;
}

function goToSelectedQuestion() {
    const selectedIndex = parseInt(questionSelector.value);
    if (!isNaN(selectedIndex)) {
        quizState.currentIndex = selectedIndex;
        showQuestion();
    }
}

function shuffleQuestions() {
    // Fisher-Yates shuffle algorithm
    const questions = quizState.currentQuestions;
    for (let i = questions.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [questions[i], questions[j]] = [questions[j], questions[i]];
    }
    quizState.currentIndex = 0;
    showQuestion();
    summaryElement.textContent = `Selected topic: "${quizState.currentTag}" | ${questions.length} questions (shuffled)`;
    setupQuestionNavigation();
}

function showPreviousQuestion() {
    if (quizState.currentIndex > 0) {
        quizState.currentIndex--;
        showQuestion();
    }
}

function showNextQuestion() {
    if (quizState.currentIndex < quizState.currentQuestions.length - 1) {
        quizState.currentIndex++;
        showQuestion();
    } else {
        showSummary();
    }
}

function toggleZoom(img) {
    img.classList.toggle('zoomed');
    document.body.classList.toggle('zoomed-mode', img.classList.contains('zoomed'));
    
    if (img.classList.contains('zoomed')) {
        img.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

async function showQuestion() {
    const q = quizState.currentQuestions[quizState.currentIndex];
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
        <strong>Q${quizState.currentIndex + 1}/${quizState.currentQuestions.length}: ${q.question_short}</strong><br>
        ${questionLong ? `<details><summary>Show Full Question</summary>
            <div class="markdown">${marked.parse(questionLong)}</div>
            ${q.question_image ? `
            <div class="image-container">
                <img src="${q.question_image}" 
                     class="quiz-image" 
                     alt="Question illustration"
                     onclick="toggleZoom(this)">
                <div class="image-caption" onclick="toggleZoom(this)">Click image to zoom</div>
            </div>` : ''}
        </details>` : ''}
        <details><summary>Show Answer</summary>
            <p><strong>Short Answer:</strong> ${q.answer_short}</p>
            ${answerLong ? `
            <div class="markdown">${marked.parse(answerLong)}</div>
            ${q.answer_image ? `
            <div class="image-container">
                <img src="${q.answer_image}" 
                     class="quiz-image" 
                     alt="Answer illustration"
                     onclick="toggleZoom(this)">
                <div class="image-caption" onclick="toggleZoom(this)">Click image to zoom</div>
            </div>` : ''}
            ` : ''}
        </details>
        <div class="answer-buttons">
            <button onclick="markAnswer(true)">✅ Correct</button>
            <button onclick="markAnswer(false)">❌ Wrong</button>
        </div>
    `;

    quizContainer.appendChild(div);

    // Apply Prism highlighting if available
    if (typeof Prism !== 'undefined') {
        const codeBlocks = div.querySelectorAll('pre code');
        codeBlocks.forEach((block) => {
            Prism.highlightElement(block);
            block.parentElement.classList.add('line-numbers');
        });
    }

    updateNavButtons();
    questionSelector.value = quizState.currentIndex;
}

function updateNavButtons() {
    previousBtn.disabled = quizState.currentIndex === 0;
    nextBtn.disabled = quizState.currentIndex === quizState.currentQuestions.length - 1;
    nextBtn.textContent = quizState.currentIndex === quizState.currentQuestions.length - 1 ? "Finish Quiz ➡️" : "Next ➡️";
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
    const q = quizState.currentQuestions[quizState.currentIndex];
    quizState.userResponses[q.id] = isCorrect;
    if (isCorrect) quizState.correctCount++;

    if (quizState.currentIndex < quizState.currentQuestions.length - 1) {
        quizState.currentIndex++;
        showQuestion();
    } else {
        showSummary();
    }
}

function showSummary() {
    quizContainer.innerHTML = '';
    quizControls.style.display = 'none';
    questionNavigation.style.display = 'none';
    scoreSection.style.display = 'block';

    const total = quizState.currentQuestions.length;
    const percentage = Math.round((quizState.correctCount / total) * 100);

    scoreSection.innerHTML = `
        <h2>Quiz Complete!</h2>
        <p>You answered ${quizState.correctCount} out of ${total} correctly (${percentage}%).</p>
        <button onclick="showReview()">🔍 Review Answers</button>
        <button onclick="location.reload()">🔄 Start New Quiz</button>
    `;
}

function showReview() {
    quizContainer.innerHTML = '<h3>📝 Quiz Review</h3>';

    quizState.currentQuestions.forEach((q, i) => {
        const div = document.createElement('div');
        div.className = 'question-block review';
        const correct = quizState.userResponses[q.id];

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