














function filterQuestions() {
    const selectedCategory = dom.categorySelector.value;
    
    // First filter by category if not "All"
    let filtered = [...quizState.questions];
    if (selectedCategory !== 'All') {
        filtered = quizState.questions.filter(q => q.tags.includes(selectedCategory));
    }
    
    // Then filter by selected tags if any
    if (quizState.selectedTags.length > 0) {
        filtered = filtered.filter(question => 
            question.tags && question.tags.some(tag => 
                quizState.selectedTags.includes(tag)
            )
        );
    }
    
    quizState.filteredQuestions = filtered;
    quizState.currentIndex = 0;
    quizState.userAnswers = {};
    showQuestion();
}