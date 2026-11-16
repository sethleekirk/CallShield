# CallShield — Real-Time AI Phone Scam Detection


**CallShield — Video Demonstration**

https://drive.google.com/file/d/1l8cwhGtPNk4kHR1_KOO7LtR58HIJBiaX/view?usp=sharing

*Note: This video has been edited to remove processing delays and unnecessary pauses, keeping the demonstration concise. The application's results have not been altered.*


## The Problem



Phone scams use impersonation, urgency, and manipulation to trick people into revealing sensitive information or sending money. These scams can be difficult to recognize during a conversation, especially when the caller sounds convincing. CallShield aims to help people identify potentially fraudulent conversations in real time.



## What I Built



CallShield is a locally running AI application that listens to microphone audio and analyzes spoken conversations for potential scam indicators.



The user starts a recording through a web interface. CallShield transcribes the audio, continuously evaluates the conversation, and displays a preliminary scam assessment.



When the recording ends, a second AI model reviews the full conversation and produces a final classification of **YES**, **NO**, or **UNSURE**, along with an explanation.



## How It Works



### Data:



I constructed a semantic database containing approximately **20,000 labeled examples** of scam and legitimate conversations.



Each example is converted into a numerical embedding using a pretrained sentence embedding model.



**Dataset sources:** **Dataset: ICFD-31k**



ICFD-31k contains 31,000 synthetic English and Hinglish phone-call conversations labeled for conversational fraud detection. CallShield uses a subset of these conversations to construct its semantic similarity database.



(see citation in the Datasets section below)



### Model



CallShield combines three pretrained AI models:



1. **Whisper tiny.en** transcribes microphone audio in approximately five-second segments. Qwen2.5 1.5B helps correct transcription errors.

2. **MiniLM (all-MiniLM-L6-v2)** analyzes overlapping conversation sections and compares their embeddings with the 20 most similar scam examples and 20 most similar legitimate examples.

3. **Qwen3.5 2B** reviews the complete transcript and preliminary classification, then provides a final decision and explanation.



MiniLM was selected for fast semantic similarity comparisons, while Qwen provides additional context-based reasoning.



The classification system uses custom similarity-weighting rules and thresholds. The displayed percentages are relative heuristic scores, not calibrated probabilities.



The pretrained neural models were not fine-tuned. The original implementation includes the semantic index, classification logic, real-time transcription pipeline, AI integration, and user interface.



## Results



### Baseline



A baseline that always predicts a legitimate conversation achieves 50.93% accuracy on my evaluation dataset (1,528 legitimate conversations out of 3,000), but detects no scams.



### MiniLM Evaluation



An earlier MiniLM-only classifier was evaluated on 3,000 labeled conversations:



| Metric | Result |
|---|---:|
| Accuracy | 95.43% |
| Scam recall | 98.57% |
| Correctly detected scams | 1,451 |
| Missed scams | 21 |
| False scam warnings | 116 |


These results apply to the MiniLM-only classifier, not the final combined system that uses Qwen to verify, change, and explain the results.



The evaluation was not a fully independent held-out test because classification thresholds were adjusted using evaluation results. Therefore, the reported accuracy is preliminary and may overestimate performance on new conversations.



### Real-World Demonstrations



In small spoken-script tests, CallShield successfully processed microphone audio and produced final classifications.



Two scam-script demonstrations received uncertain preliminary classifications but were identified as potential scams by Qwen's final review.



A legitimate delivery-related conversation was classified as legitimate.



### Limitations



CallShield can produce false positives and false negatives. Its performance depends on transcription quality, dataset coverage, and the language used during a conversation.



The complete hybrid system has not yet been evaluated on a large independent dataset.



CallShield is an experimental prototype and does not provide financial advice or guarantee scam detection.



## What I would Do Next



1. **Independent evaluation:** Test the complete system on a new, unseen dataset to measure accuracy, false positives, and false negatives.

2. **Real-world reliability:** Improve transcription accuracy and measure detection speed across different accents, background noise levels, and conversation types.



## Sources



### Pretrained Models



- [MiniLM — Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

- [Whisper — OpenAI](https://github.com/openai/whisper)

- [Qwen2.5 — Qwen Team](https://ollama.com/library/qwen2.5)

- [Qwen3.5 — Qwen Team](https://ollama.com/library/qwen3.5)



### Software and AI Tools



- [Ollama](https://ollama.com/)

- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)

- [FastAPI](https://fastapi.tiangolo.com/)

- [ChatGPT — OpenAI](https://chatgpt.com/) — coding, debugging, and documentation assistance.



### Datasets



**Dataset: ICFD-31k**



Authors: Rishi Ahuja, Kumar Prateek, and Simranjit Singh (2026).



Source: https://huggingface.co/datasets/rishia2220/icfd-31k



ICFD-31k is released for defensive research and education only. Do not use it to generate scam scripts, automate social engineering, train offensive systems, or support any activity that facilitates fraud.



Citation:

@inproceedings{ahuja2026icfd31k,

&#x20; title     = {{ICFD-31k}: A Large-Scale Dataset and Benchmark for Real-Time Conversational Fraud Detection},

&#x20; author    = {Ahuja, Rishi and Prateek, Kumar and Singh, Simranjit},

&#x20; booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-ECAI)},

&#x20; year      = {2026}

}



## Running CallShield Locally



**Requirements:** Windows, Python, microphone, and Ollama.



Install the Python dependencies:



```bash

pip install -r requirements.txt

```



Download the required Ollama models:



```bash

ollama pull qwen2.5:1.5b

ollama pull qwen3.5:2b

```



Start the application:



```bash

python server.py

```



Open `http://127.0.0.1:8000` in your browser and allow microphone access.



The first run may require downloading the Whisper and MiniLM models.



**Note:** CallShield is a research prototype, not a production-ready financial safety system.

