# Latency of text classification for medical text

You have worked on medical text classification using LLM.
You might have encountered latency due to llm being slow.
how did you tackle the problem.


**Q: You have worked on medical text classification using LLMs. You might have encountered latency due to LLMs being slow. How did you tackle the problem?**

**Answer:**

Yes, latency was definitely one of the main challenges while using LLMs for real-time or near real-time medical text classification, especially when relying on large models like GPT-3.5 or BERT-based transformers. To address this, I applied a multi-layered approach depending on the use case latency tolerance:



**1. Tiered Model Strategy (Fast + Smart):**
We implemented a two-tiered architecture:

* A lightweight, **fast classifier** (like Logistic Regression or a distilled model such as DistilBERT) was used as the first pass for quick classification.
* Only ambiguous or edge cases were forwarded to the **LLM (e.g., GPT-3.5 or T5)** for deeper contextual understanding.
  This reduced LLM calls by around **60%**, significantly improving average response time.


**2. Caching & Prompt Optimization:**
We applied **aggressive caching** using hash-based keys for repeat messages and **prompt compression** techniques:

* Shorter, more focused prompts with templated context reduced token count and call time.
* Precomputed few-shot examples were cached and reused across sessions.



**3. Asynchronous Processing + Queuing:**
For cases where real-time classification wasn’t critical (e.g., non-urgent patient queries), I used **asynchronous pipelines**:

* Requests were placed in an **AWS SQS** or **Azure Service Bus** queue.
* An async worker handled LLM classification in the background.
* Users got a "response received, processing…" message, and the classified output was updated on the dashboard later.



**4. Distillation + Quantization for On-Device Models:**
We fine-tuned a distilled version of BERT on our clinical dataset using Hugging Face’s `transformers`.
Later, I used **ONNX Runtime** and **8-bit quantization** to deploy it in a production setting. This improved inference latency by over **50%** without sacrificing much accuracy.



**5. Batch Processing and Parallelization:**
When processing bulk text (e.g., overnight logs), I parallelized classification jobs using **Spark UDFs in Databricks** or used **Ray** for distributed LLM inference—keeping memory and token usage under control.



**Outcome:**
With these strategies combined, we brought down average inference time from **2.3 seconds to under 600 milliseconds** for the majority of cases, while still preserving LLM-quality accuracy for edge cases.

