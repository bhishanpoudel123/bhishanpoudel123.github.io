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
const showAllQuestionsBtn = document.getElementById('show-all-questions');
const previewControls = document.getElementById('preview-controls');

// Create all questions view container
const allQuestionsView = document.createElement('div');
allQuestionsView.id = 'all-questions-view';
document.body.insertBefore(allQuestionsView, quizContainer.nextSibling);

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
    questionSelector.addEventListener('change', function () {
        goToQuestionBtn.disabled = this.value === "";
    });
    showAllQuestionsBtn.addEventListener('click', showAllQuestions);

    // Add mobile view toggle
    const mobileViewToggle = document.getElementById('mobile-view-toggle');
    if (mobileViewToggle) {
        mobileViewToggle.addEventListener('click', toggleMobileView);
    }

    // Check if mobile on load
    checkIfMobile();
}

function toggleMobileView() {
    document.body.classList.toggle('mobile-view');
    const toggleBtn = document.getElementById('mobile-view-toggle');
    if (document.body.classList.contains('mobile-view')) {
        toggleBtn.textContent = '📱 Normal View';
    } else {
        toggleBtn.textContent = '📱 Mobile View';
    }
}

function checkIfMobile() {
    if (window.innerWidth <= 768) {
        // Auto-enable mobile view on small screens
        document.body.classList.add('mobile-view');
        const toggleBtn = document.getElementById('mobile-view-toggle');
        if (toggleBtn) {
            toggleBtn.textContent = '📱 Normal View';
        }
    }
}

async function startQuiz() {
    const selectedTag = tagSelector.value;
    if (selectedTag) {
        await loadQuestions(selectedTag);
    }
}

async function loadQuestions(tag) {
    try {
        // Clear existing UI elements
        quizContainer.innerHTML = '';
        scoreSection.style.display = 'none';
        allQuestionsView.style.display = 'none';

        // Reset quiz state
        quizState.currentTag = tag;
        quizState.correctCount = 0;
        quizState.userResponses = {};
        quizState.currentIndex = 0;

        // Load questions
        const files = ['questions/linear_regression.json', 'questions/interview_drfirst.json'];
        let questions = [];

        for (let file of files) {
            const res = await fetch(file);
            const qns = await res.json();
            questions.push(...qns.filter(q => q.tags.includes(tag)));
        }

        quizState.currentQuestions = questions;

        if (questions.length === 0) {
            summaryElement.textContent = `No questions found for "${tag}". Please select another topic.`;
            previewControls.style.display = 'none';
            return;
        }

        summaryElement.textContent = `Selected topic: "${tag}" | ${questions.length} questions`;
        previewControls.style.display = 'block';
        quizControls.style.display = 'flex';
        questionNavigation.style.display = 'flex';

        // Initialize quiz navigation
        setupQuestionNavigation();
        showQuestion();

    } catch (error) {
        console.error('Error loading questions:', error);
        summaryElement.textContent = "Error loading questions. Please try again.";
        previewControls.style.display = 'none';
    }
}

function showAllQuestions() {
    if (!quizState.currentQuestions.length) return;

    // Hide quiz interface
    quizContainer.style.display = 'none';
    quizControls.style.display = 'none';
    questionNavigation.style.display = 'none';
    previewControls.style.display = 'none';

    // Show preview interface
    allQuestionsView.style.display = 'block';
    allQuestionsView.innerHTML = `
        <h2>All Questions (${quizState.currentQuestions.length})</h2>
        <div class="all-questions-container"></div>
        <button id="back-to-quiz">← Back to Quiz</button>
    `;

    const container = allQuestionsView.querySelector('.all-questions-container');
    const backBtn = allQuestionsView.querySelector('#back-to-quiz');

    backBtn.addEventListener('click', backToQuiz);

    quizState.currentQuestions.forEach((q, index) => {
        const preview = document.createElement('div');
        preview.className = 'question-preview';
        preview.innerHTML = `
            <h3>Q${index + 1}: ${q.question_short}</h3>
            <details>
                <summary>Show Answer</summary>
                <p>${q.answer_short}</p>
            </details>
        `;
        preview.addEventListener('click', () => {
            quizState.currentIndex = index;
            backToQuiz();
        });
        container.appendChild(preview);
    });
}

function backToQuiz() {
    allQuestionsView.style.display = 'none';
    quizContainer.style.display = 'block';
    quizControls.style.display = 'flex';
    questionNavigation.style.display = 'flex';
    previewControls.style.display = 'block';
    showQuestion();
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

        // Determine if we're running on GitHub Pages
        const isGitHubPages = window.location.hostname.includes('github.io');

        // Fetch the markdown content
        const res = await fetch(path.startsWith('content/') ? path : `content/${path}`);
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        let markdown = await res.text();

        // Fix image paths in markdown
        markdown = markdown.replace(
            /!\[(.*?)\]\(\/assets\/images\/(.*?)\)/g,
            `![$1](${isGitHubPages ? '/quiz' : ''}/assets/images/$2)`
        );

        return markdown;
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
    previewControls.style.display = 'none';
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