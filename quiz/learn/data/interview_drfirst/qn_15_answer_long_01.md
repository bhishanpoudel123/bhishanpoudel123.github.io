# Q: medical NLP techniques to categorize unstructured data

## Interview Question: You mentioned applying medical NLP techniques to categorize unstructured medical text - can you elaborate on what you mean by that?  

**Answer:** Absolutely. At AmerisourceBergen, I worked on classifying unstructured patient chatter and clinical notes using NLP, which involved several key steps:  

1. **Text Preprocessing for Medical Data**  
   *"Medical text is messy—it contains abbreviations (e.g., 'HTN' for hypertension), misspellings, and clinician shorthand. I implemented custom cleaning pipelines using regex and spaCy to standardize terms while preserving clinical meaning. For example, mapping 'ASA' to 'aspirin' based on context."*  

2. **Domain-Specific Feature Engineering**  
   *"Beyond standard TF-IDF/Bag-of-Words, I incorporated:  
   - UMLS Metathesaurus codes to link medical concepts  
   - Section headers (e.g., 'HPI:' for History of Present Illness) as metadata  
   - Negation detection (e.g., 'no fever' ≠ 'fever') using modified NegEx patterns"*  

3. **LLM-Augmented Classification**  
   *"For high-stakes categories (e.g., 'urgent' vs. 'routine'), I fine-tuned BioClinicalBERT on labeled patient messages, achieving 92% accuracy. For edge cases, I used GPT-3.5 in a hybrid approach—generating potential labels for human review, which reduced annotation time by 40%."*  

4. **Compliance-Aware Modeling**  
   *"All models were trained with synthetic PHI generation/injection to ensure robust PII masking in production. We also implemented guardrails to flag low-confidence predictions for human review, critical for avoiding misclassification in healthcare workflows."*  

**Impact:** This system automated triage for 15,000+ monthly patient messages, prioritizing urgent cases (e.g., 'chest pain') while routing refill requests to appropriate queues—directly analogous to DrFirst's need to parse free-text form responses.  

*Would you like me to dive deeper into any specific aspect, like handling insurance jargon or the evaluation metrics we used?*  

*(This answer showcases technical depth while linking back to the JD's focus on healthcare NLP and compliance.)*
