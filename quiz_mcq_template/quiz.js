const isGitHubPages = window.location.host.includes('github.io');
const basePath = isGitHubPages ? '/quiz_mcq' : '';

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
    htmlIndex: {},
    markedInstance: marked.marked.setOptions({
        highlight: (code, lang) => hljs.highlightAuto(code).value
    }),
    sidebarVisible: true,
    learningResources: {} // Track learning resources for each question
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
    startQuizBtn: document.getElementById('start-quiz'),
    learnTopicBtn: document.getElementById('learn-topic'),
    topicContainer: document.getElementById('topic-container'),
    topicContent: document.getElementById('topic-content'),
    backToQuizBtn: document.getElementById('back-to-quiz'),
    toggleSidebarBtn: document.getElementById('toggle-sidebar'),
    sidebar: document.querySelector('.sidebar'),
    htmlSelector: document.getElementById('html-selector'),
    topicIframe: document.getElementById('topic-iframe'),
    learningPanel: document.createElement('div'),
    resourceTabs: document.createElement('div'),
    resourceContent: document.createElement('div')
};

// Initialize Quiz
document.addEventListener('DOMContentLoaded', async () => {
    await loadIndexJson();
    populateDurationOptions();
    setupEventListeners();
    initLearningPanel();
});

// Initialize the learning resources panel
function initLearningPanel() {
    dom.learningPanel.className = 'learning-panel hidden';
    dom.learningPanel.innerHTML = `
        <div class="panel-header">
            <h3>Learning Resources</h3>
            <button class="close-panel">×</button>
        </div>
    `;
    
    dom.resourceTabs.className = 'resource-tabs';
    dom.resourceContent.className = 'resource-content';
    
    dom.learningPanel.appendChild(dom.resourceTabs);
    dom.learningPanel.appendChild(dom.resourceContent);
    document.body.appendChild(dom.learningPanel);
    
    // Close panel handler
    dom.learningPanel.querySelector('.close-panel').addEventListener('click', () => {
        dom.learningPanel.classList.add('hidden');
    });
}

// Load index.json and fetch questions
async function loadIndexJson() {
    try {
        const response = await fetch(`${basePath}/data/index.json`);
        if (!response.ok) throw new Error('Failed to load index');
        
        const indexData = await response.json();
        quizState.indexMap = indexData.categories;
        await loadAllQuestions(indexData.categories);
    } catch (err) {
        console.error('Error loading index:', err);
        dom.questionText.innerHTML = `<div class="error">Error loading index file</div>`;
    }
}

async function loadAllQuestions(categories) {
    const allQuestions = [];
    const categoryNames = Object.keys(categories);

    for (const category of categoryNames) {
        const categoryPath = categories[category];
        try {
            const response = await fetch(`${basePath}/${categoryPath}`);
            if (!response.ok) throw new Error(`Failed to load ${categoryPath}`);
            
            const categoryData = await response.json();
            for (const questionPath of categoryData.questions) {
                try {
                    const qResponse = await fetch(`${basePath}/${questionPath}`);
                    if (!qResponse.ok) throw new Error(`Failed to load ${questionPath}`);
                    
                    const question = await qResponse.json();
                    question.category = category;
                    allQuestions.push(question);
                    
                    // Store learning resources
                    if (question.learning_resources) {
                        quizState.learningResources[question.id] = question.learning_resources;
                    }
                } catch (err) {
                    console.warn(`Skipped ${questionPath}:`, err.message);
                }
            }
        } catch (err) {
            console.warn(`Skipped ${categoryPath}:`, err.message);
        }
    }

    quizState.questions = allQuestions;
    quizState.categories = [...new Set(categoryNames)];
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
    dom.topicContainer.style.display = 'none';
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
                    <button class="view-resources" data-question-id="${q.id}">View Learning Resources</button>
                </div>`;
            }).join('')}
        </div>
    `;

    dom.quizContainer.style.display = 'none';
    dom.topicContainer.style.display = 'none';
    dom.scoreSection.innerHTML = summaryHTML;
    dom.scoreSection.style.display = 'block';
    
    document.getElementById('restart-btn')?.addEventListener('click', resetQuiz);
    
    // Add event listeners for learning resources buttons
    document.querySelectorAll('.view-resources').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const questionId = parseInt(e.target.dataset.questionId);
            showLearningResources(questionId);
        });
    });
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show learning resources for a question
function showLearningResources(questionId) {
    const resources = quizState.learningResources[questionId];
    if (!resources || resources.length === 0) {
        dom.resourceContent.innerHTML = '<div class="error">No learning resources available for this question</div>';
        dom.learningPanel.classList.remove('hidden');
        return;
    }

    // Clear previous tabs and content
    dom.resourceTabs.innerHTML = '';
    dom.resourceContent.innerHTML = '';

    // Create tabs for each resource
    resources.forEach((resource, index) => {
        const tab = document.createElement('button');
        tab.className = 'resource-tab' + (index === 0 ? ' active' : '');
        tab.textContent = resource.title;
        tab.dataset.resourceIndex = index;
        
        tab.addEventListener('click', () => {
            // Set active tab
            document.querySelectorAll('.resource-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Load resource content
            loadResourceContent(resource);
        });
        
        dom.resourceTabs.appendChild(tab);
    });

    // Load first resource by default
    if (resources.length > 0) {
        loadResourceContent(resources[0]);
    }

    dom.learningPanel.classList.remove('hidden');
}

// Load content for a specific resource
function loadResourceContent(resource) {
    dom.resourceContent.innerHTML = '<div class="loading">Loading resource...</div>';
    
    const resourcePath = `${basePath}/${resource.path}`;
    
    if (resource.type === 'markdown' || resource.type === 'code') {
        fetch(resourcePath)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                return res.text();
            })
            .then(text => {
                if (resource.type === 'markdown') {
                    dom.resourceContent.innerHTML = quizState.markedInstance.parse(text);
                } else {
                    // For code files, display in a pre block
                    dom.resourceContent.innerHTML = `<pre><code class="language-${resource.path.split('.').pop()}">${escapeHtml(text)}</code></pre>`;
                }
                hljs.highlightAll();
            })
            .catch(err => {
                console.error('Error loading resource:', err);
                dom.resourceContent.innerHTML = `<div class="error">Failed to load resource: ${err.message}</div>`;
            });
    } else if (resource.type === 'html') {
        dom.resourceContent.innerHTML = `<iframe class="resource-iframe" src="${resourcePath}"></iframe>`;
    }
}

// Show all learning resources for a category
async function showCategoryResources(category) {
    const questions = quizState.questions.filter(q => q.category === category);
    if (questions.length === 0) {
        alert('No questions found for this category');
        return;
    }

    // Hide quiz and show topic container
    dom.quizContainer.style.display = 'none';
    dom.scoreSection.style.display = 'none';
    dom.topicContainer.style.display = 'block';
    
    // Create HTML for all resources in this category
    let resourcesHTML = `<h2>${category} Learning Resources</h2>`;
    
    for (const question of questions) {
        if (!question.learning_resources || question.learning_resources.length === 0) continue;
        
        resourcesHTML += `<div class="question-resources">
            <h3>Question ${question.id}: ${question.question}</h3>`;
        
        for (const resource of question.learning_resources) {
            resourcesHTML += `<div class="resource-item">
                <h4>${resource.title} (${resource.type})</h4>`;
            
            if (resource.type === 'markdown') {
                try {
                    const response = await fetch(`${basePath}/${resource.path}`);
                    const text = await response.text();
                    resourcesHTML += quizState.markedInstance.parse(text);
                } catch (err) {
                    resourcesHTML += `<div class="error">Failed to load resource: ${err.message}</div>`;
                }
            } else if (resource.type === 'code') {
                try {
                    const response = await fetch(`${basePath}/${resource.path}`);
                    const code = await response.text();
                    resourcesHTML += `<pre><code class="language-${resource.path.split('.').pop()}">${escapeHtml(code)}</code></pre>`;
                } catch (err) {
                    resourcesHTML += `<div class="error">Failed to load code: ${err.message}</div>`;
                }
            } else if (resource.type === 'html') {
                resourcesHTML += `<iframe src="${basePath}/${resource.path}" class="resource-iframe"></iframe>`;
            }
            
            resourcesHTML += `</div>`;
        }
        
        resourcesHTML += `</div><hr>`;
    }
    
    dom.topicContent.innerHTML = resourcesHTML;
    hljs.highlightAll();
}

// Helper to escape HTML
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
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

// Update the updateAnswerDisplay function to use collapsible sections
async function updateAnswerDisplay(question) {
    if (quizState.showAnswer) {
        dom.answerSection.style.display = 'block';
        dom.correctAnswer.innerHTML = renderMarkdown(`**Answer:** ${question.answer}`);
        dom.explanation.innerHTML = renderMarkdown(`**Explanation:** ${question.explanation || ''}`);
        
        // Clear previous resources
        const existingResources = document.querySelectorAll('.answer-resource');
        existingResources.forEach(el => el.remove());
        
        // Load and display all available resources
        if (question.learning_resources && question.learning_resources.length > 0) {
            const resourcesContainer = document.createElement('div');
            resourcesContainer.className = 'answer-resources-container';
            resourcesContainer.innerHTML = '<h3>Learning Resources:</h3>';
            
            for (const resource of question.learning_resources) {
                const resourceDiv = document.createElement('div');
                resourceDiv.className = 'answer-resource';
                
                // Create collapsible details element
                const details = document.createElement('details');
                const summary = document.createElement('summary');
                summary.textContent = `${resource.title} (${resource.type})`;
                details.appendChild(summary);
                
                let content = '';
                switch (resource.type) {
                    case 'markdown':
                        content = await loadMarkdown(resource.path);
                        break;
                    case 'html':
                        const htmlContent = await loadHTML(resource.path);
                        content = `<div class="html-content">${htmlContent}</div>`;
                        break;
                    case 'code':
                        content = await loadCode(resource.path);
                        break;
                    default:
                        content = `<div class="error">Unknown resource type: ${resource.type}</div>`;
                }
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'resource-content';
                contentDiv.innerHTML = content;
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

async function loadTopicContent() {
    try {
        const selectedCategory = dom.categorySelector.value;
        if (!selectedCategory || selectedCategory === 'All') {
            alert('Please select a specific topic category first');
            return;
        }

        // Construct the path to the category's HTML file
        const categoryFolder = selectedCategory.replace(/\s+/g, '_');
        const topicPath = `${basePath}/data/${categoryFolder}/${categoryFolder.toLowerCase()}.html`;

        // Test if the file exists first
        try {
            const testResponse = await fetch(topicPath);
            if (!testResponse.ok) {
                throw new Error(`File not found: ${topicPath}`);
            }
            
            // Load the topic HTML file into the iframe
            dom.topicIframe.src = topicPath;
            
            // Show/hide sections
            dom.quizContainer.style.display = 'none';
            dom.scoreSection.style.display = 'none';
            dom.topicContainer.style.display = 'block';

            if (quizState.sidebarVisible) toggleSidebar();
        } catch (err) {
            console.error('Error loading topic:', err);
            // If the main HTML file doesn't exist, fall back to showing all resources
            await showCategoryResources(selectedCategory);
        }
    } catch (err) {
        console.error('Error in loadTopicContent:', err);
        alert('An error occurred while loading the topic');
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
    dom.learnTopicBtn.addEventListener('click', loadTopicContent);
    dom.backToQuizBtn.addEventListener('click', backToQuiz);
    dom.toggleSidebarBtn.addEventListener('click', toggleSidebar);
}