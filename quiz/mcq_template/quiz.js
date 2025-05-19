const isGitHubPages = window.location.host.includes('github.io');
const basePath = isGitHubPages ? '/quiz/mcq_template' : '';

// Quiz State Management
const quizState = {
    questions: [],
    filteredQuestions: [],
    currentIndex: 0,
    userAnswers: {},
    showAnswer: false,
    tags: [],
    selectedTag: null,
    timerInterval: null,
    quizDuration: 20 * 60,
    timeLeft: 20 * 60,
    startTime: null,
    endTime: null,
    quizStarted: false,
    indexMap: {},
    markedInstance: marked.marked.setOptions({
        highlight: (code, lang) => hljs.highlightAuto(code).value
    }),
    sidebarVisible: true
};

// DOM Elements
const dom = {
    tagSelector: document.getElementById('tag-selector'),
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
    startQuizBtn: document.getElementById('start-quiz'),
    learnTopicBtn: document.getElementById('learn-topic'),
    topicContainer: document.getElementById('topic-container'),
    topicIframe: document.getElementById('topic-iframe'),
    backToQuizBtn: document.getElementById('back-to-quiz'),
    toggleSidebarBtn: document.getElementById('toggle-sidebar'),
    sidebar: document.querySelector('.sidebar')
};

// Initialize Quiz
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadIndexJson();
        populateDurationOptions();
        setupEventListeners();
    } catch (error) {
        console.error('Initialization error:', error);
        dom.questionText.innerHTML = `<div class="error">Initialization failed. Check console for details.</div>`;
    }
});

async function loadIndexJson() {
    try {
        const response = await fetch(`${basePath}/data/index.json`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const indexData = await response.json();
        quizState.indexMap = indexData.categories;
        await loadAllQuestions(indexData.categories);
    } catch (err) {
        console.error('Error loading index:', err);
        throw err;
    }
}

async function loadAllQuestions(categories) {
    const allQuestions = [];
    const allTags = new Set();

    for (const [topic, questionsFile] of Object.entries(categories)) {
        try {
            const questionsResponse = await fetch(`${basePath}/${questionsFile}`);
            if (!questionsResponse.ok) throw new Error(`Failed to load ${questionsFile}`);
            const questionsData = await questionsResponse.json();
            
            for (const questionPath of questionsData.questions) {
                try {
                    const questionResponse = await fetch(`${basePath}/${questionPath}`);
                    if (!questionResponse.ok) throw new Error(`Failed to load ${questionPath}`);
                    const question = await questionResponse.json();
                    
                    if (!question.tags) question.tags = [];
                    if (!question.tags.includes(topic)) question.tags.push(topic);
                    
                    question.tags.forEach(tag => allTags.add(tag));
                    allQuestions.push(question);
                } catch (err) {
                    console.warn(`Skipped ${questionPath}:`, err.message);
                }
            }
        } catch (err) {
            console.warn(`Skipped ${topic} questions:`, err.message);
        }
    }

    quizState.questions = allQuestions;
    quizState.tags = Array.from(allTags).sort();
    populateTagSelector();
}

function populateTagSelector() {
    dom.tagSelector.innerHTML = '<option value="">Select a tag...</option>';
    quizState.tags.forEach(tag => {
        const option = document.createElement('option');
        option.value = tag;
        option.textContent = tag;
        dom.tagSelector.appendChild(option);
    });
}

function filterQuestions() {
    if (!quizState.selectedTag) {
        quizState.filteredQuestions = [];
        showQuestion();
        return;
    }

    quizState.filteredQuestions = quizState.questions.filter(question => 
        question.tags && question.tags.includes(quizState.selectedTag)
    );
    
    quizState.currentIndex = 0;
    quizState.userAnswers = {};
    showQuestion();
}

function populateDurationOptions() {
    [5, 10, 15, 20, 30, 45, 60].forEach(min => {
        const opt = document.createElement('option');
        opt.value = min;
        opt.textContent = `${min} minutes`;
        dom.durationSelector.appendChild(opt);
    });
    dom.durationSelector.value = 20;
}

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

function resetQuiz() {
    clearInterval(quizState.timerInterval);
    quizState.currentIndex = 0;
    quizState.userAnswers = {};
    quizState.showAnswer = false;
    quizState.quizStarted = false;
    quizState.selectedTag = null;
    dom.tagSelector.value = '';
    dom.timerDisplay.textContent = '--:--';
    dom.startTime.textContent = '-';
    dom.endTime.textContent = '-';
    filterQuestions();
    dom.scoreSection.style.display = 'none';
    dom.quizContainer.style.display = 'block';
    dom.topicContainer.style.display = 'none';
}

function endQuiz() {
    clearInterval(quizState.timerInterval);
    showQuizResults();
}

function showQuizResults() {
    const total = quizState.filteredQuestions.length;
    const correct = quizState.filteredQuestions.filter(q => quizState.userAnswers[q.id] === q.answer).length;
    const unanswered = quizState.filteredQuestions.filter(q => !quizState.userAnswers.hasOwnProperty(q.id)).length;
    const incorrect = total - correct - unanswered;

    const percentage = total > 0 ? Math.round((correct / total) * 100) : 0;
    
    const summaryHTML = `
        <div class="score-summary">
            <h2>Quiz Results</h2>
            <div class="score">Score: ${correct}/${total} (${percentage}%)</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="stats">
                <div class="stat correct-stat">✓ Correct: ${correct}</div>
                <div class="stat incorrect-stat">✗ Incorrect: ${incorrect}</div>
                <div class="stat unanswered-stat">? Unanswered: ${unanswered}</div>
            </div>
            <button id="restart-btn" class="restart-btn">Restart Quiz</button>
        </div>
        <div class="question-reviews">
            <h3>Question Review</h3>
            ${quizState.filteredQuestions.map(q => {
                const ua = quizState.userAnswers[q.id];
                const isCorrect = ua === q.answer;
                const status = !ua ? 'unanswered' : isCorrect ? 'correct' : 'incorrect';
                return `
                <div class="review-question ${status}">
                    <h4>Question ${q.id}</h4>
                    <div class="question">${renderMarkdown(q.question)}</div>
                    ${ua ? `<div class="user-answer"><strong>Your answer:</strong> ${renderMarkdown(ua)} <span>${isCorrect ? '✓' : '✗'}</span></div>` : '<div class="unanswered">Not answered</div>'}
                    <div class="correct-answer"><strong>Correct answer:</strong> ${renderMarkdown(q.answer)}</div>
                    <div class="explanation">${renderMarkdown(q.explanation)}</div>
                </div>`;
            }).join('')}
        </div>
    `;

    dom.quizContainer.style.display = 'none';
    dom.topicContainer.style.display = 'none';
    dom.scoreSection.innerHTML = summaryHTML;
    dom.scoreSection.style.display = 'block';
    document.getElementById('restart-btn')?.addEventListener('click', resetQuiz);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showQuestion() {
    if (quizState.filteredQuestions.length === 0) {
        dom.questionTitle.textContent = "No questions available";
        dom.questionText.innerHTML = quizState.selectedTag 
            ? `No questions found for "${quizState.selectedTag}"`
            : "Please select a tag to start the quiz";
        dom.optionsContainer.innerHTML = '';
        return;
    }

    const question = quizState.filteredQuestions[quizState.currentIndex];
    if (!question) return;

    dom.questionTitle.innerHTML = `Question ${question.id}`;
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

    if (Object.keys(quizState.userAnswers).length === quizState.filteredQuestions.length) {
        endQuiz();
    }
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

async function updateAnswerDisplay(question) {
    if (quizState.showAnswer) {
        dom.answerSection.style.display = 'block';
        dom.correctAnswer.innerHTML = renderMarkdown(`**Answer:** ${question.answer}`);
        dom.explanation.innerHTML = renderMarkdown(`**Explanation:** ${question.explanation || ''}`);

        // Clear previous resources
        const existingResources = document.querySelectorAll('.learning-resource');
        existingResources.forEach(el => el.remove());

        // Load and display learning resources if they exist
        if (question.learning_resources && question.learning_resources.length > 0) {
            const resourcesContainer = document.createElement('div');
            resourcesContainer.className = 'learning-resources-container';
            resourcesContainer.innerHTML = '<h3>Additional Resources:</h3>';
            
            for (const resource of question.learning_resources) {
                const resourceDiv = document.createElement('div');
                resourceDiv.className = 'learning-resource';
                
                const details = document.createElement('details');
                const summary = document.createElement('summary');
                summary.textContent = `${resource.title} (${resource.type})`;
                details.appendChild(summary);
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'resource-content';
                
                try {
                    let content = '';
                    switch (resource.type) {
                        case 'markdown':
                            content = await loadMarkdown(resource.path);
                            break;
                        case 'html':
                            content = await loadHTML(resource.path);
                            break;
                        case 'code':
                            content = await loadCode(resource.path);
                            break;
                        default:
                            content = `<div class="error">Unsupported resource type: ${resource.type}</div>`;
                    }
                    contentDiv.innerHTML = content;
                } catch (err) {
                    console.error('Error loading resource:', err);
                    contentDiv.innerHTML = `<div class="error">Failed to load resource: ${err.message}</div>`;
                }
                
                details.appendChild(contentDiv);
                resourceDiv.appendChild(details);
                resourcesContainer.appendChild(resourceDiv);
            }
            
            dom.explanation.appendChild(resourcesContainer);
            hljs.highlightAll();
        }
    } else {
        dom.answerSection.style.display = 'none';
    }
}

async function loadMarkdown(path) {
    try {
        const response = await fetch(`${basePath}/${path}`);
        if (!response.ok) throw new Error('Failed to load markdown');
        const text = await response.text();
        return quizState.markedInstance.parse(text);
    } catch (err) {
        console.error('Error loading markdown:', err);
        return `<div class="error">Failed to load markdown: ${err.message}</div>`;
    }
}

async function loadHTML(path) {
    try {
        const response = await fetch(`${basePath}/${path}`);
        if (!response.ok) throw new Error('Failed to load HTML');
        return await response.text();
    } catch (err) {
        console.error('Error loading HTML:', err);
        return `<div class="error">Failed to load HTML: ${err.message}</div>`;
    }
}

async function loadCode(path) {
    try {
        const response = await fetch(`${basePath}/${path}`);
        if (!response.ok) throw new Error('Failed to load code');
        const code = await response.text();
        const language = path.split('.').pop();
        return `<pre><code class="language-${language}">${escapeHtml(code)}</code></pre>`;
    } catch (err) {
        console.error('Error loading code:', err);
        return `<div class="error">Failed to load code: ${err.message}</div>`;
    }
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function jumpToQuestion(id) {
    const index = quizState.filteredQuestions.findIndex(q => q.id === id);
    if (index >= 0) {
        quizState.currentIndex = index;
        quizState.showAnswer = false;
        showQuestion();
    }
}

async function loadTopicContent() {
    if (!quizState.selectedTag) {
        alert('Please select a tag first');
        return;
    }

    const topicPath = `${basePath}/data/${quizState.selectedTag.replace(/\s+/g, '_')}/${quizState.selectedTag.toLowerCase().replace(/\s+/g, '_')}.html`;

    try {
        const testResponse = await fetch(topicPath);
        if (!testResponse.ok) throw new Error(`File not found: ${topicPath}`);
        
        dom.topicIframe.src = topicPath;
        dom.quizContainer.style.display = 'none';
        dom.scoreSection.style.display = 'none';
        dom.topicContainer.style.display = 'block';
    } catch (err) {
        console.error('Error loading topic:', err);
        alert(`No learning material available for "${quizState.selectedTag}"`);
    }
}

function backToQuiz() {
    dom.topicContainer.style.display = 'none';
    dom.quizContainer.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function toggleSidebar() {
    quizState.sidebarVisible = !quizState.sidebarVisible;
    if (quizState.sidebarVisible) {
        dom.sidebar.classList.remove('hidden');
        document.body.classList.remove('sidebar-hidden');
        dom.toggleSidebarBtn.innerHTML = '☰';
        dom.toggleSidebarBtn.title = 'Hide Sidebar';
    } else {
        dom.sidebar.classList.add('hidden');
        document.body.classList.add('sidebar-hidden');
        dom.toggleSidebarBtn.innerHTML = '☰';
        dom.toggleSidebarBtn.title = 'Show Sidebar';
    }
}

function setupEventListeners() {
    // Tag selection
    dom.tagSelector.addEventListener('change', (e) => {
        quizState.selectedTag = e.target.value || null;
        filterQuestions();
    });
    
    // Quiz controls
    dom.durationSelector.addEventListener('change', () => quizState.timerInterval && initTimer());
    dom.startQuizBtn.addEventListener('click', () => {
        if (!quizState.selectedTag) {
            alert('Please select a tag first');
            return;
        }
        initTimer();
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

    // Learn topic feature
    dom.learnTopicBtn.addEventListener('click', loadTopicContent);
    dom.backToQuizBtn.addEventListener('click', backToQuiz);
    
    // Sidebar toggle
    dom.toggleSidebarBtn.addEventListener('click', toggleSidebar);
}
