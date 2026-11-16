with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update Mermaid diagram to include Pertemuan 12
old_mermaid = 'G["Pertemuan 10<br>Seq2Seq NMT + Attention"] --> H["Pertemuan 11<br>Conversational Chatbot"]'
new_mermaid = 'G["Pertemuan 10<br>Seq2Seq NMT + Attention"] --> H["Pertemuan 11<br>Conversational Chatbot"]\n    H --> I["Pertemuan 12<br>Summarization & Generation"]'
if old_mermaid in text:
    text = text.replace(old_mermaid, new_mermaid)

# 2. Update Table of contents
old_toc = '  - [Pertemuan 11: Conversational Chatbot — Retrieval vs Neural Network](#pertemuan-11-conversational-chatbot--retrieval-vs-neural-network)'
new_toc = old_toc + '\n  - [Pertemuan 12: Transformer & Mini Projects (Summarization & Generation)](#pertemuan-12-transformer--mini-projects-summarization--generation)'
if old_toc in text and new_toc not in text:
    text = text.replace(old_toc, new_toc)

# 3. Update Syllabus Table
old_row = '| **11** | Conversational Chatbot | Model 1: Retrieval (TF-IDF + Cosine Sim)<br>Model 2: Neural Classifier (Embedding + GRU/LSTM) | IndoHealth & FAQ Chatbot Dataset | [Buka](<./Pertemuan 11/submitChatbot_Pertemuan11.ipynb>) |'
new_row = old_row + '\n| **12** | Summarization & Generation | Summarization: TF-IDF vs TextRank (ROUGE)<br>Generation: LSTM vs GRU (Perplexity) | IndoSum (Kata.ai) & Wikipedia-ID | [Summarization](<./pertemuan 12/Mini_Project_Text_Summarization.ipynb>)<br>[Generation](<./pertemuan 12/Mini_Project_Text_Generation.ipynb>) |'
if old_row in text and new_row not in text:
    text = text.replace(old_row, new_row)

# 4. Update Detailed Section after Pertemuan 11
old_p11_end = '- Berkas: Folder `Pertemuan 11/` (`Chatbot - Pertemuan11.ipynb`, `submitChatbot_Pertemuan11.ipynb`, dataset Excel/CSV, slide materi PDF).'
new_p12_section = """- Berkas: Folder `Pertemuan 11/` (`Chatbot_Pertemuan11.ipynb`, `submitChatbot_Pertemuan11.ipynb`, `IndoHealth_NLP_NCD_Instruct_QA_Sample.csv`, slide materi PDF).

---

### Pertemuan 12: Transformer & Mini Projects (Summarization & Generation)
- **Topik Teori:** Arsitektur Transformer, Self-Attention ($Q, K, V$), Positional Encoding, Multi-Head Attention, dan varian Encoder-only (BERT/IndoBERT), Decoder-only (GPT), dan Encoder-Decoder (T5/BART).
- **Mini Project 1: Text Summarization Bahasa Indonesia:**
  - **Dataset:** IndoSum (`kata-ai/indosum`) — korpus berita nasional dan ringkasan acuan manusia (*gold summary*).
  - **Model 1 (Asyam):** *Extractive Summarization* berbasis **TF-IDF Sentence Ranking** dengan normalisasi panjang kata dan *chronological reordering*.
  - **Model 2 (Agung):** *Extractive Summarization* berbasis algoritma **TextRank** (graf kemiripan kalimat Cosine + PageRank).
  - **Evaluasi (Daun):** Metrik ROUGE-1, ROUGE-2, ROUGE-L, waktu komputasi, contoh ringkasan terbaik/terburuk, serta analisis kualitatif makna.
  - **Notebook:** [`Mini_Project_Text_Summarization.ipynb`](<./pertemuan 12/Mini_Project_Text_Summarization.ipynb>)
- **Mini Project 2: Text Generation Bahasa Indonesia:**
  - **Dataset:** `indonesian-nlp/wikipedia-id` (HuggingFace) — korpus artikel Wikipedia bahasa Indonesia.
  - **Model 1 (Albi):** LSTM Language Model.
  - **Model 2 (Indah):** GRU Language Model.
  - **Evaluasi (Daun):** Training/Validation Loss, Perplexity, Next-Word Accuracy, waktu training, analisis repetisi, dan pengaruh *Temperature Sampling* (0.5, 0.8, 1.0, 1.2).
  - **Notebook:** [`Mini_Project_Text_Generation.ipynb`](<./pertemuan 12/Mini_Project_Text_Generation.ipynb>)"""

if old_p11_end in text and '### Pertemuan 12' not in text:
    text = text.replace(old_p11_end, new_p12_section)

# 5. Update Directory Tree
old_tree_end = """└── Pertemuan 11/                                        # Praktikum: Conversational Chatbot
    ├── NLP Pertemuan #11.pdf                            # Slide materi Chatbot Retrieval & Neural
    ├── Chatbot - Pertemuan11.ipynb                      # Eksperimen alur chatbot
    ├── submitChatbot_Pertemuan11.ipynb                  # Laporan final evaluasi chatbot
    ├── dataset_chatbot.xlsx                             # Dataset intent chatbot primer
    ├── IndoHealth_NLP_NCD_Instruct_QA_Sample.csv        # Sampel QA kesehatan IndoHealth
    └── Processed Dataset.csv                            # Dataset hasil preprocessing"""

new_tree_end = """├── Pertemuan 11/                                        # Praktikum: Conversational Chatbot
│   ├── NLP Pertemuan #11.pdf                            # Slide materi Chatbot Retrieval & Neural
│   ├── Chatbot_Pertemuan11.ipynb                        # Notebook utama chatbot IndoHealth
│   ├── submitChatbot_Pertemuan11.ipynb                  # Laporan final evaluasi chatbot
│   └── IndoHealth_NLP_NCD_Instruct_QA_Sample.csv        # Sampel QA kesehatan IndoHealth
│
└── pertemuan 12/                                        # Praktikum: Transformer, Summarization & Generation
    ├── NLP Pertemuan #12 Transformer.pdf                # Slide materi Transformer & Self-Attention
    ├── link.txt                                         # Sumber tautan dataset resmi
    ├── Mini_Project_Text_Summarization.ipynb            # Mini Project Summarization (TF-IDF vs TextRank)
    ├── Mini_Project_Text_Generation.ipynb               # Mini Project Generation (LSTM vs GRU)
    └── templatenyaText_Generation.ipynb                 # Berkas template awal generasi teks"""

if old_tree_end in text:
    text = text.replace(old_tree_end, new_tree_end)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("README.md berhasil diperbarui dengan Pertemuan 12!")
