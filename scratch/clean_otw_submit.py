import json

def clean_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Memproses {path} - Total sel awal: {len(nb['cells'])}")
    
    # 1. Cari dan hapus sel lama yang duplikat
    # Cari sel markdown template lama: "## 6. Model 2: Chatbot Neural Network (Tokenisasi → Padding → Embedding → GRU/LSTM)"
    # dan sel kode lama yang mendefinisikan model = tf.keras.Sequential([ tf.keras.layers.Input(shape=(MAXLEN,)), tf.keras.layers.Embedding(VOCAB_SIZE, 64, mask_zero=True), tf.keras.layers.Bidirectional(rnn_layer(64))
    
    cleaned_cells = []
    for c in nb['cells']:
        src = ''.join(c['source']).strip()
        
        # Cek apakah sel template markdown lama
        if c['cell_type'] == 'markdown' and src.startswith("## 6. Model 2: Chatbot Neural Network (Tokenisasi → Padding → Embedding → GRU/LSTM)"):
            print(" -> Menghapus sel markdown template lama")
            continue
            
        # Cek apakah sel kode template lama (yang fit tanpa build_model terpisah)
        if c['cell_type'] == 'code' and 'RNN_TYPE = "GRU"     # ganti "LSTM" untuk membandingkan' in src and 'model = tf.keras.Sequential([' in src and 'tf.keras.layers.Bidirectional(rnn_layer(64))' in src:
            print(" -> Menghapus sel kode lama yang menumpuk / duplikat")
            continue
            
        # Jika sel markdown Bagian 6, perbarui dengan penjelasan teori komprehensif
        if c['cell_type'] == 'markdown' and '## 6. Model 2: Chatbot Neural Network (Bi-LSTM / Bi-GRU)' in src:
            c['source'] = [
                "---\n",
                "## 6. Model 2: Chatbot Neural Network (Intent Classifier Bi-GRU / Bi-LSTM)\n",
                "**Penanggung Jawab:** **ASYAM SAKHI MAULANA** (NIM: `240535600363`)  \n",
                "**Peran & Fokus:** Pipeline Tokenisasi, Embedding Laten, Pemodelan Sekuensial Recurrent Neural Network (Bi-GRU & Bi-LSTM), Regularisasi Dropout, Dense Softmax, & Optimasi Training.\n",
                "\n",
                "---\n",
                "\n",
                "### 🧠 Landasan Teori & Arsitektur Model\n",
                "\n",
                "Berbeda dengan **Model 1 (Retrieval TF-IDF)** yang mengasumsikan kata bersifat independen (*bag-of-words*) dan hanya mengandalkan kemiripan leksikal, **Model 2 (Neural Network)** memodelkan urutan temporal kata (*sequential dependencies*) serta memproyeksikan makna semantik kata ke dalam ruang vektor kontinu (*dense embedding space*).\n",
                "\n",
                "```text\n",
                "Pipeline Arsitektur Model 2:\n",
                "[Teks Pertanyaan Mentah]\n",
                "       │\n",
                "       ▼ Preprocessing (Case Folding, Regex Cleaning, Slang Normalization, Filler Removal)\n",
                "[Teks Bersih]\n",
                "       │\n",
                "       ▼ Tokenisasi & Vocabulary Lookup (<PAD>=0, <OOV>=1)\n",
                "[Sequence of Token IDs]\n",
                "       │\n",
                "       ▼ Zero-Padding (MAXLEN = 30)\n",
                "[Tensor Input: (Batch Size, 30)]\n",
                "       │\n",
                "       ▼ Embedding Layer (dim=64, mask_zero=True)\n",
                "[Dense Vectors: (Batch Size, 30, 64)]\n",
                "       │\n",
                "       ▼ Bidirectional RNN (GRU / LSTM dengan 64 hidden units)\n",
                "       │ ── Forward Pass:  h_fwd  (64 unit) ──\n",
                "       │ ── Backward Pass: h_bwd  (64 unit) ──\n",
                "[Concatenated Context State: (Batch Size, 128)]\n",
                "       │\n",
                "       ▼ Dropout (30%)\n",
                "       │\n",
                "       ▼ Dense Feature Layer (64 units, activation=\"relu\")\n",
                "       │\n",
                "       ▼ Dropout (20%)\n",
                "       │\n",
                "       ▼ Output Layer (3 classes: Diabetes, Hypertension, Comorbidity; Softmax)\n",
                "[Distribusi Probabilitas Intent: P(y|x)]\n",
                "```\n",
                "\n",
                "#### 1. Keunggulan Bidirectional RNN untuk Kalimat Medis\n",
                "* Dalam kalimat tanya medis berbahasa Indonesia, posisi kata kunci sering berpindah-pindah:\n",
                "  - *\"Dok, apakah **kadar gula darah** yang tinggi memicu sering buang air kecil?\"* (Kata kunci di depan).\n",
                "  - *\"Saya sering merasa lemas dan pusing, apakah ini tanda **hipertensi**?\"* (Kata kunci di akhir).\n",
                "* Dengan **Bidirectional RNN**, model membaca urutan kalimat dari dua arah sekaligus:\n",
                "  - $\\overrightarrow{h}_t = \\text{RNN}_{\\text{fwd}}(x_t, \\overrightarrow{h}_{t-1})$ (arah maju: memahami konteks masa lalu).\n",
                "  - $\\overleftarrow{h}_t = \\text{RNN}_{\\text{bwd}}(x_t, \\overleftarrow{h}_{t+1})$ (arah mundur: memahami konteks masa depan).\n",
                "  - Output gabungan merangkum konteks pertanyaan secara utuh.\n",
                "\n",
                "#### 2. Perbandingan Karakteristik: Bi-GRU vs. Bi-LSTM\n",
                "* **GRU (Gated Recurrent Unit):** Menggunakan 2 gerbang (*Reset Gate* dan *Update Gate*). Memiliki parameter ~25% lebih hemat, lebih cepat dilatih, serta sangat cocok untuk dataset dengan ratusan sampel agar tidak overfit.\n",
                "* **LSTM (Long Short-Term Memory):** Menggunakan 3 gerbang (*Input Gate*, *Forget Gate*, *Output Gate*) dan *Cell State* terpisah. Sangat kuat dalam mempertahankan memori jangka panjang.\n",
                "* *Bagian ini mengimplementasikan Bi-GRU sebagai model utama dan menyediakan fungsi komparasi otomatis dengan Bi-LSTM.*"
            ]
            print(" -> Memperbarui teks markdown teori Bagian 6")
            
        cleaned_cells.append(c)
        
    nb['cells'] = cleaned_cells
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Selesai! Total sel akhir di {path}: {len(nb['cells'])}\n")

clean_notebook('Pertemuan 11/otw submit Chatbot_Pertemuan11.ipynb')
clean_notebook('Pertemuan 11/Chatbot_Pertemuan11.ipynb')
