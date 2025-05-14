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
        if (!res.ok) {
            throw new Error('Failed to load tags');
        }
        const data = await res.json();

        // Clear existing options except the first one
        while (tagSelector.options.length > 1) {
            tagSelector.remove(1);
        }

        // Add options to dropdown with spaces instead of underscores
        data.tags.forEach(tag => {
            const option = document.createElement('option');
            option.value = tag;
            option.textContent = tag.replace(/_/g, ' '); // Display with spaces
            tagSelector.appendChild(option);
        });

        // Debug log
        console.log("Loaded tags:", data.tags);
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

    // Setup image zoom functionality
    window.toggleZoom = function (img) {
        img.classList.toggle('zoomed');
        document.body.classList.toggle('zoomed-mode', img.classList.contains('zoomed'));

        if (img.classList.contains('zoomed')) {
            img.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    };

    // Setup answer marking functionality
    window.markAnswer = function (isCorrect) {
        const q = quizState.currentQuestions[quizState.currentIndex];
        quizState.userResponses[q.id] = isCorrect;
        if (isCorrect) quizState.correctCount++;

        if (quizState.currentIndex < quizState.currentQuestions.length - 1) {
            quizState.currentIndex++;
            showQuestion();
        } else {
            showSummary();
        }
    };
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

async function getDataFolders() {
    try {
        const response = await fetch('data/data_folders.json');
        if (!response.ok) {
            throw new Error('Failed to load data folders');
        }
        const data = await response.json();

        // Handle both array and object with folders property
        if (Array.isArray(data)) {
            return data;
        } else if (data && Array.isArray(data.folders)) {
            return data.folders;
        }

        // Default fallback
        return ['interview_drfirst', 'linear_regression'];
    } catch (error) {
        console.error('Error loading data folders:', error);
        return ['interview_drfirst', 'linear_regression'];
    }
}

async function loadQuestions(tag) {
    try {
        // Clear UI elements
        quizContainer.innerHTML = '';
        scoreSection.style.display = 'none';
        allQuestionsView.style.display = 'none';

        // Reset state
        quizState.currentTag = tag;
        quizState.correctCount = 0;
        quizState.userResponses = {};
        quizState.currentIndex = 0;

        // Get folders
        const dataFolders = await getDataFolders();
        console.log("Searching folders:", dataFolders);

        // Normalize tag
        const normalizedTag = tag.toLowerCase().replace(/_/g, ' ').trim();

        let allQuestions = [];

        // Search each folder
        for (const folder of dataFolders) {
            try {
                console.group(`Searching folder: ${folder}`);

                // Load index
                const indexPath = `data/${folder}/index.json`;
                const indexResponse = await fetch(indexPath);
                if (!indexResponse.ok) {
                    console.warn(`Index not found: ${indexPath}`);
                    continue;
                }

                const indexData = await indexResponse.json();
                const questionFiles = indexData.files || [];
                console.log(`Found ${questionFiles.length} questions in index`);

                // Load each question
                for (const qnFile of questionFiles) {
                    try {
                        const filePath = `data/${folder}/${qnFile}`;
                        const fileResponse = await fetch(filePath);
                        if (!fileResponse.ok) continue;

                        const questions = await fileResponse.json();
                        const questionArray = Array.isArray(questions) ? questions : [questions];

                        // Check tags
                        for (const q of questionArray) {
                            if (q.tags && q.tags.some(t =>
                                t.toLowerCase().replace(/_/g, ' ').trim() === normalizedTag
                            )) {
                                console.log(`Found match: ${q.id}`);
                                allQuestions.push(q);
                            }
                        }
                    } catch (e) {
                        console.error(`Error loading ${qnFile}:`, e);
                    }
                }
                console.groupEnd();
            } catch (e) {
                console.error(`Error with folder ${folder}:`, e);
            }
        }

        // Update state
        quizState.currentQuestions = allQuestions;

        if (allQuestions.length === 0) {
            summaryElement.textContent = `No questions found for "${tag}". Check console for details.`;
            previewControls.style.display = 'none';
            return;
        }

        // Show questions
        summaryElement.textContent = `Topic: "${tag}" | ${allQuestions.length} questions`;
        previewControls.style.display = 'block';
        quizControls.style.display = 'flex';
        questionNavigation.style.display = 'flex';

        setupQuestionNavigation();
        showQuestion();

    } catch (error) {
        console.error('Error:', error);
        summaryElement.textContent = "Error loading questions. See console.";
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

async function showQuestion() {
    const q = quizState.currentQuestions[quizState.currentIndex];
    quizContainer.innerHTML = '';

    let questionLong = '';
    let answerLong = '';
    let answerLongHtml = '';

    // Load long question content if available
    if (q.question_long_path && q.question_long_path !== '') {
        questionLong = await fetchMarkdown(q.question_long_path);
    }

    // Load long answer content from markdown if available
    if (q.answer_long_md && q.answer_long_md.length > 0) {
        answerLong = await fetchMarkdown(q.answer_long_md[0]);
    }

    // Load HTML answer content if available
    if (q.answer_long_html && q.answer_long_html.length > 0) {
        answerLongHtml = await fetchHtml(q.answer_long_html[0]);
    }

    const div = document.createElement('div');
    div.className = 'question-block';

    // Get image paths
    const questionImage = q.question_image || '';
    const answerImage = q.answer_image || '';

    // Create the question HTML
    let html = `
        <strong>Q${quizState.currentIndex + 1}/${quizState.currentQuestions.length}: ${q.question_short}</strong><br>
    `;

    // Add long question if available
    if (questionLong || q.question_long_path) {
        html += `
        <details>
            <summary>Show Full Question</summary>
            ${questionLong ? `<div class="markdown">${marked.parse(questionLong)}</div>` : ''}
            ${questionImage ? `
            <div class="image-container">
                <img src="${questionImage}" 
                     class="quiz-image" 
                     alt="Question illustration"
                     onclick="toggleZoom(this)">
                <div class="image-caption">Click image to zoom</div>
            </div>` : ''}
        </details>
        `;
    }

    // Add answer section
    html += `
        <details>
            <summary>Show Answer</summary>
            <p><strong>Short Answer:</strong> ${q.answer_short}</p>
            ${answerLong ? `
            <div class="markdown">${marked.parse(answerLong)}</div>
            ` : ''}
            ${answerLongHtml ? `
            <div class="markdown html-content">${answerLongHtml}</div>
            ` : ''}
            ${answerImage ? `
            <div class="image-container">
                <img src="${answerImage}" 
                     class="quiz-image" 
                     alt="Answer illustration"
                     onclick="toggleZoom(this)">
                <div class="image-caption">Click image to zoom</div>
            </div>` : ''}
        </details>
        <div class="answer-buttons">
            <button onclick="markAnswer(true)">✅ Correct</button>
            <button onclick="markAnswer(false)">❌ Wrong</button>
        </div>
    `;

    div.innerHTML = html;
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

        // The path is now a full path from the JSON file
        // Remove leading slash if present to make it relative to the current directory
        const cleanPath = path.startsWith('/') ? path.substring(1) : path;

        const res = await fetch(cleanPath);
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return await res.text();
    } catch (error) {
        console.error('Error loading markdown:', error, path);
        return '';
    }
}

async function fetchHtml(path) {
    try {
        if (!path) return '';

        // Remove leading slash if present to make it relative to the current directory
        const cleanPath = path.startsWith('/') ? path.substring(1) : path;

        const res = await fetch(cleanPath);
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return await res.text();
    } catch (error) {
        console.error('Error loading HTML:', error, path);
        return '';
    }
}

function showSummary() {
    quizContainer.innerHTML = '';
    quizControls.style.display = 'none';
    questionNavigation.style.display = 'none';
    previewControls.style.display = 'none';
    scoreSection.style.display = 'block';

    const total = quizState.currentQuestions.length;
    const attempted = Object.keys(quizState.userResponses).length;
    const percentage = Math.round((quizState.correctCount / attempted) * 100) || 0;

    scoreSection.innerHTML = `
        <h2>Quiz Complete!</h2>
        <p>You attempted ${attempted} out of ${total} questions.</p>
        <p>You answered ${quizState.correctCount} out of ${attempted} correctly (${percentage}%).</p>
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

        // Check if this question was answered
        const wasAnswered = q.id in quizState.userResponses;

        div.innerHTML = `
            <strong>Q${i + 1}: ${q.question_short}</strong>
            <p class="answer"><strong>Answer:</strong> ${q.answer_short}</p>
            ${wasAnswered ?
                `<p class="${correct ? 'correct' : 'incorrect'}">
                    You marked this as ${correct ? 'correct ✅' : 'incorrect ❌'}
                </p>` :
                `<p class="unanswered">Not answered</p>`
            }
        `;

        quizContainer.appendChild(div);
    });
}