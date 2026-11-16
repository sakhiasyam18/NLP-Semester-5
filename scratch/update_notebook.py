import json
import os

file_path = 'Pertemuan 11/Chatbot_Pertemuan11.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ---------------------------------------------------------
# 1. Update Cell 2 (Setup) with fallback paths
# ---------------------------------------------------------
cell_2_code = '''import os, re, time, random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 42          # seed yang sama untuk SEMUA kelompok agar split data identik
random.seed(SEED)
np.random.seed(SEED)
pd.set_option("display.max_colwidth", 120)

# Sambungkan ke Google Drive (jika dijalankan pada Google Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print("Info (Google Drive):", e)

# Konfigurasi path project & dataset
BASE_PATH = '/content/drive/MyDrive/PROJECT_NLP_HEALTH'
DATASET_PATH = os.path.join(
    BASE_PATH,
    '01_DATASET',
    'IndoHealth_NLP_NCD_Instruct_QA_Sample.csv'
)

# Fallback otomatis jika dataset berada di folder kerja saat ini
if not os.path.exists(DATASET_PATH):
    for candidate in [
        'IndoHealth_NLP_NCD_Instruct_QA_Sample.csv',
        '01_DATASET/IndoHealth_NLP_NCD_Instruct_QA_Sample.csv',
        'Pertemuan 11/IndoHealth_NLP_NCD_Instruct_QA_Sample.csv'
    ]:
        if os.path.exists(candidate):
            DATASET_PATH = candidate
            break

OUTPUT_PATH = os.path.join(BASE_PATH, '03_HASIL_PREPROCESSING') if os.path.exists(BASE_PATH) else '03_HASIL_PREPROCESSING'
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Persiapan notebook berhasil!")
print("SEED         :", SEED)
print("Dataset path :", DATASET_PATH)
'''

nb['cells'][2]['source'] = [l + '\n' for l in cell_2_code.strip().split('\n')]

# ---------------------------------------------------------
# 2. Update Cell 7 (Albi's Preprocessing) - add bridge & RESPONS
# ---------------------------------------------------------
cell_7_src = ''.join(nb['cells'][7]['source'])
bridge_code = '''

# ==============================================================================
# PIPELINE PREPROCESSING & STANDARISASI KOLOM UNTUK MODEL 1 & MODEL 2
# ==============================================================================
def preprocess(text):
    """Pipeline preprocessing terpadu untuk data inferensi/input baru."""
    return remove_filler_words(normalisasi(text_cleaning(text)))

# Standardisasi nama kolom agar sinkron dengan seluruh evaluasi downstream
df['pertanyaan'] = df['instruction']
df['jawaban'] = df['output']
df['intent'] = df['disease_tag']
df['bersih'] = df['instruction_clean']

# Daftar kelas intent unik
CLASSES = sorted(df['intent'].unique().tolist())
print("\\nStandardisasi selesai!")
print("Daftar kelas intent (CLASSES):", CLASSES)

# Mapping respons template default untuk demo & fallback chatbot
RESPONS = {
    "Diabetes": (
        "Informasi Diabetes: Perhatikan kadar glukosa darah berkala, batasi asupan karbohidrat sederhana/gula, "
        "perbanyak serat, rutin berolahraga, dan konsultasikan jadwal serta dosis obat antidiabetes atau insulin ke dokter."
    ),
    "Hypertension": (
        "Informasi Hipertensi: Jaga kestabilan tekanan darah dengan membatasi konsumsi garam (natrium), "
        "mengelola stres, menjaga berat badan ideal, dan mematuhi konsumsi obat penurun tensi yang diresepkan."
    ),
    "Comorbidity": (
        "Informasi Komorbiditas: Pasien dengan kondisi diabetes dan hipertensi bersamaan memerlukan "
        "penanganan terpadu untuk mencegah komplikasi kardiovaskular dan penurunan fungsi ginjal. "
        "Sangat disarankan berkonsultasi secara intensif dengan dokter spesialis."
    )
}
'''
if 'def preprocess(text):' not in cell_7_src:
    nb['cells'][7]['source'] = [l + '\n' for l in (cell_7_src.rstrip() + bridge_code).split('\n')]

# ---------------------------------------------------------
# 3. Update Cell 11 (Indah's Model 1) - add vectorizer alias
# ---------------------------------------------------------
cell_11_code = '''t0 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
X_train_m1 = tfidf.fit_transform(train_df["bersih"])
y_train_labels = train_df["intent"].values
train_time_m1 = time.time() - t0

# Alias vectorizer agar kompatibel dengan kode analisis OOV di Bagian 10 (Agung)
vectorizer = tfidf

def predict_m1(texts):
    # kembalikan (daftar intent, daftar skor kemiripan)
    bersih = [preprocess(t) for t in texts]
    sims = cosine_similarity(tfidf.transform(bersih), X_train_m1)
    idx = sims.argmax(axis=1)
    return list(y_train_labels[idx]), list(sims.max(axis=1))

print(f"Ukuran vocabulary TF-IDF: {len(tfidf.vocabulary_)}")
print(f"Waktu training Model 1  : {train_time_m1:.4f} detik")
'''
nb['cells'][11]['source'] = [l + '\n' for l in cell_11_code.strip().split('\n')]

# ---------------------------------------------------------
# 4. Bagian 6 (Asyam's Part): Replace Cells 12 to 17
# ---------------------------------------------------------
# Let's define the new cells for Bagian 6:
cell_6_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
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
        "       ▼ Output Layer (3 classes: Diabetes, Hypertension, Comorbidity; activation=\"softmax\")\n",
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
        "  - Output gabungan $h = [\\overrightarrow{h}_T \\;;\\; \\overleftarrow{h}_1]$ merangkum esensi pertanyaan secara menyeluruh.\n",
        "\n",
        "#### 2. Perbandingan Karakteristik: Bi-GRU vs. Bi-LSTM\n",
        "* **GRU (Gated Recurrent Unit):** Menggunakan 2 gerbang (*Reset Gate* dan *Update Gate*). Memiliki parameter lebih sedikit (~25–30% lebih hemat), lebih cepat dilatih, serta sangat cocok untuk dataset dengan ratusan sampel agar tidak overfit.\n",
        "* **LSTM (Long Short-Term Memory):** Menggunakan 3 gerbang (*Input Gate*, *Forget Gate*, *Output Gate*) dan *Cell State* terpisah. Sangat kuat dalam mempertahankan memori jangka panjang.\n",
        "* *Bagian ini mengimplementasikan Bi-GRU sebagai model utama dan menyediakan fungsi komparasi otomatis dengan Bi-LSTM.*"
    ]
}

cell_6_1_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# BAGIAN 6.1: TOKENISASI, VOCABULARY & PADDING SEQUENCE\n",
        "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
        "# ==============================================================================\n",
        "\n",
        "import tensorflow as tf\n",
        "tf.random.set_seed(SEED)\n",
        "\n",
        "print(\"Versi TensorFlow:\", tf.__version__)\n",
        "\n",
        "# 1. Membangun kamus kata (Vocabulary) HANYA dari data latih (train_df)\n",
        "# Hal ini mutlak dilakukan untuk mencegah data leakage dari data validasi / data uji.\n",
        "vocab_raw = sorted({tok for s in train_df[\"bersih\"] for tok in str(s).split()})\n",
        "\n",
        "# Token khusus: <PAD> indeks 0 (padding), <OOV> indeks 1 (Out-Of-Vocabulary)\n",
        "word_index = {\"<PAD>\": 0, \"<OOV>\": 1}\n",
        "for w in vocab_raw:\n",
        "    word_index[w] = len(word_index)\n",
        "\n",
        "index_word = {i: w for w, i in word_index.items()}\n",
        "VOCAB_SIZE = len(word_index)\n",
        "\n",
        "# Objek tokenizer kompatibel untuk kebutuhan analisis OOV di Bagian 10 (Agung)\n",
        "class SimpleTokenizer:\n",
        "    def __init__(self, w_index):\n",
        "        self.word_index = w_index\n",
        "tokenizer = SimpleTokenizer(word_index)\n",
        "\n",
        "# 2. Analisis distribusi panjang kalimat latih untuk menetapkan MAXLEN optimal\n",
        "panjang_kalimat = [len(str(s).split()) for s in train_df[\"bersih\"]]\n",
        "p95 = int(np.percentile(panjang_kalimat, 95))\n",
        "p_max = max(panjang_kalimat)\n",
        "p_mean = np.mean(panjang_kalimat)\n",
        "\n",
        "print(f\"Statistik Panjang Kalimat Latih:\")\n",
        "print(f\" • Rata-rata   : {p_mean:.2f} kata\")\n",
        "print(f\" • Maksimum    : {p_max} kata\")\n",
        "print(f\" • Persentil 95: {p95} kata\")\n",
        "\n",
        "# Tetapkan MAXLEN (menampung 95%+ panjang pertanyaan dengan aman tanpa pemborosan komputasi)\n",
        "MAXLEN = max(30, p95)\n",
        "print(f\"Panjang sekuens terpilih (MAXLEN): {MAXLEN}\")\n",
        "\n",
        "# 3. Fungsi konversi teks ke sekuens ID integer\n",
        "def texts_to_sequences(texts):\n",
        "    \"\"\"Mengonversi daftar string kalimat menjadi daftar token IDs (OOV otomatis bernilai 1).\"\"\"\n",
        "    return [[word_index.get(tok, 1) for tok in str(s).split()] for s in texts]\n",
        "\n",
        "# 4. Fungsi Zero-Padding\n",
        "def pad(seqs, maxlen=MAXLEN):\n",
        "    \"\"\"Menyeragamkan panjang input sequence dengan zero-padding di bagian belakang.\"\"\"\n",
        "    arr = np.zeros((len(seqs), maxlen), dtype=\"int32\")\n",
        "    for i, s in enumerate(seqs):\n",
        "        s_cut = s[:maxlen]\n",
        "        arr[i, :len(s_cut)] = s_cut\n",
        "    return arr\n",
        "\n",
        "def encode(texts_bersih):\n",
        "    \"\"\"Pipeline terpadu: teks bersih -> sequences -> padded matrix.\"\"\"\n",
        "    return pad(texts_to_sequences(texts_bersih), maxlen=MAXLEN)\n",
        "\n",
        "# 5. Label Encoding (Pemetaan nama kelas intent ke ID integer)\n",
        "label2id = {c: i for i, c in enumerate(CLASSES)}\n",
        "id2label = {i: c for i, c in enumerate(CLASSES)}\n",
        "\n",
        "# Encode data latih, validasi, dan uji\n",
        "X_train = encode(train_df[\"bersih\"])\n",
        "y_train = train_df[\"intent\"].map(label2id).values\n",
        "\n",
        "X_val = encode(val_df[\"bersih\"])\n",
        "y_val = val_df[\"intent\"].map(label2id).values\n",
        "\n",
        "X_test = encode(test_df[\"bersih\"])\n",
        "y_test = test_df[\"intent\"].map(label2id).values\n",
        "\n",
        "# Inspeksi contoh representasi teks\n",
        "contoh_teks = train_df[\"bersih\"].iloc[0]\n",
        "print(\"\\n--- Verifikasi Pipeline Representasi Teks ---\")\n",
        "print(\"Teks Asli        :\", train_df[\"pertanyaan\"].iloc[0][:80] + \"...\")\n",
        "print(\"Teks Bersih      :\", contoh_teks[:80] + \"...\")\n",
        "print(\"Token IDs        :\", texts_to_sequences([contoh_teks])[0][:10], \"...\")\n",
        "print(\"Bentuk Padded    :\", X_train[0][:15], \"...\")\n",
        "print(\"Dimensi X_train  :\", X_train.shape)\n",
        "print(\"Dimensi X_val    :\", X_val.shape)\n",
        "print(\"Dimensi X_test   :\", X_test.shape)\n",
        "print(\"Ukuran Kosakata  :\", VOCAB_SIZE)\n",
        "print(\"Pemetaan Label   :\", label2id)\n"
    ]
}

cell_6_2_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# BAGIAN 6.2: MEMBANGUN ARSITEKTUR BI-DIRECTIONAL RECURRENT NEURAL NETWORK\n",
        "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
        "# ==============================================================================\n",
        "\n",
        "# Konfigurasi Hyperparameter Model 2\n",
        "RNN_TYPE = \"GRU\"         # Pilihan utama: \"GRU\" (dapat diubah ke \"LSTM\")\n",
        "EMBEDDING_DIM = 64      # Dimensi representasi vektor kontinu kata\n",
        "HIDDEN_UNITS = 64       # Jumlah neuron sel RNN (output Bi-RNN = 64 * 2 = 128 dimensi)\n",
        "DROPOUT_RATE = 0.3      # Tingkat dropout regularisasi untuk menghindari overfitting\n",
        "LEARNING_RATE = 1e-3    # Learning rate optimizer Adam\n",
        "\n",
        "def build_model(rnn_type=RNN_TYPE, vocab_size=VOCAB_SIZE, maxlen=MAXLEN, num_classes=len(CLASSES)):\n",
        "    \"\"\"Membangun arsitektur Sequential Bidirectional RNN (GRU / LSTM) untuk klasifikasi intent.\"\"\"\n",
        "    recurrent_layer = (\n",
        "        tf.keras.layers.GRU(HIDDEN_UNITS, return_sequences=False, dropout=0.2)\n",
        "        if rnn_type == \"GRU\" else\n",
        "        tf.keras.layers.LSTM(HIDDEN_UNITS, return_sequences=False, dropout=0.2)\n",
        "    )\n",
        "    \n",
        "    m = tf.keras.Sequential([\n",
        "        # 1. Input Layer: Menerima matriks ID token integer berukuran (MAXLEN,)\n",
        "        tf.keras.layers.Input(shape=(maxlen,), name=\"input_sequence_ids\"),\n",
        "        \n",
        "        # 2. Embedding Layer: mask_zero=True agar komputasi RNN mengabaikan token padding (ID 0)\n",
        "        tf.keras.layers.Embedding(\n",
        "            input_dim=vocab_size,\n",
        "            output_dim=EMBEDDING_DIM,\n",
        "            mask_zero=True,\n",
        "            name=\"dense_word_embedding\"\n",
        "        ),\n",
        "        \n",
        "        # 3. Bidirectional Recurrent Layer: Membaca teks secara sekuensial maju dan mundur\n",
        "        tf.keras.layers.Bidirectional(\n",
        "            recurrent_layer,\n",
        "            name=f\"bidirectional_{rnn_type.lower()}\"\n",
        "        ),\n",
        "        \n",
        "        # 4. Regularisasi Dropout utama\n",
        "        tf.keras.layers.Dropout(DROPOUT_RATE, name=\"dropout_recurrent\"),\n",
        "        \n",
        "        # 5. Fully Connected Hidden Layer dengan Non-Linearitas ReLU\n",
        "        tf.keras.layers.Dense(64, activation=\"relu\", name=\"dense_feature_extractor\"),\n",
        "        tf.keras.layers.Dropout(0.2, name=\"dropout_dense\"),\n",
        "        \n",
        "        # 6. Output Layer: Distribusi probabilitas Softmax untuk seluruh kelas intent\n",
        "        tf.keras.layers.Dense(num_classes, activation=\"softmax\", name=\"output_intent_probabilities\")\n",
        "    ], name=f\"Chatbot_IntentClassifier_Bi_{rnn_type}\")\n",
        "    \n",
        "    m.compile(\n",
        "        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),\n",
        "        loss=\"sparse_categorical_crossentropy\",\n",
        "        metrics=[\"accuracy\"]\n",
        "    )\n",
        "    return m\n",
        "\n",
        "# Inisialisasi model utama\n",
        "model = build_model(rnn_type=RNN_TYPE)\n",
        "model.summary()\n"
    ]
}

cell_6_3_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# BAGIAN 6.3: PELATIHAN MODEL DENGAN EARLY STOPPING & PENCATATAN WAKTU TRAINING\n",
        "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
        "# ==============================================================================\n",
        "\n",
        "# Callbacks:\n",
        "# 1. EarlyStopping: Menghentikan pelatihan jika val_loss tidak membaik selama 20 epoch,\n",
        "#    serta merestorasi bobot model pada epoch terbaik (restore_best_weights=True).\n",
        "# 2. ReduceLROnPlateau: Memperkecil learning rate jika model mulai stagnan.\n",
        "callbacks_m2 = [\n",
        "    tf.keras.callbacks.EarlyStopping(\n",
        "        monitor=\"val_loss\",\n",
        "        patience=20,\n",
        "        restore_best_weights=True,\n",
        "        verbose=1\n",
        "    ),\n",
        "    tf.keras.callbacks.ReduceLROnPlateau(\n",
        "        monitor=\"val_loss\",\n",
        "        factor=0.5,\n",
        "        patience=8,\n",
        "        min_lr=1e-5,\n",
        "        verbose=0\n",
        "    )\n",
        "]\n",
        "\n",
        "print(f\"Memulai pelatihan Model 2: Bi-{RNN_TYPE} pada dataset latih...\")\n",
        "t0 = time.time()\n",
        "\n",
        "history_m2 = model.fit(\n",
        "    X_train,\n",
        "    y_train,\n",
        "    validation_data=(X_val, y_val),\n",
        "    epochs=200,\n",
        "    batch_size=16,\n",
        "    verbose=0,  # Ubah ke 1 jika ingin melihat log setiap epoch\n",
        "    callbacks=callbacks_m2\n",
        ")\n",
        "\n",
        "train_time_m2 = time.time() - t0\n",
        "\n",
        "best_epoch = int(np.argmin(history_m2.history[\"val_loss\"])) + 1\n",
        "best_val_loss = min(history_m2.history[\"val_loss\"])\n",
        "best_val_acc = history_m2.history[\"val_accuracy\"][best_epoch - 1]\n",
        "\n",
        "print(\"\\n\" + \"=\"*55)\n",
        "print(f\"Hasil Pelatihan Model 2 (Bi-{RNN_TYPE}):\")\n",
        "print(f\" • Total Epoch Dijalankan : {len(history_m2.history['loss'])}\")\n",
        "print(f\" • Best Epoch Terpilih     : Epoch ke-{best_epoch}\")\n",
        "print(f\" • Best Validation Loss   : {best_val_loss:.4f}\")\n",
        "print(f\" • Best Validation Acc    : {best_val_acc * 100:.2f}%\")\n",
        "print(f\" • Total Waktu Training   : {train_time_m2:.2f} detik\")\n",
        "print(\"=\"*55)\n"
    ]
}

cell_6_4_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# BAGIAN 6.4: VISUALISASI KURVA PEMBELAJARAN (LOSS & AKURASI)\n",
        "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
        "# ==============================================================================\n",
        "\n",
        "h = history_m2.history\n",
        "epochs_range = range(1, len(h[\"loss\"]) + 1)\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n",
        "\n",
        "# Subplot 1: Kurva Cross-Entropy Loss\n",
        "axes[0].plot(epochs_range, h[\"loss\"], label=\"Train Loss\", color=\"#1f77b4\", linewidth=2)\n",
        "axes[0].plot(epochs_range, h[\"val_loss\"], label=\"Validation Loss\", color=\"#d62728\", linewidth=2, linestyle=\"--\")\n",
        "axes[0].axvline(best_epoch, color=\"green\", linestyle=\":\", label=f\"Best Epoch ({best_epoch})\")\n",
        "axes[0].set_title(f\"Kurva Loss Model 2 (Bi-{RNN_TYPE})\", fontsize=12, fontweight=\"bold\")\n",
        "axes[0].set_xlabel(\"Epoch\", fontsize=10)\n",
        "axes[0].set_ylabel(\"Sparse Categorical Crossentropy\", fontsize=10)\n",
        "axes[0].grid(True, linestyle=\"--\", alpha=0.6)\n",
        "axes[0].legend(loc=\"upper right\", frameon=True)\n",
        "\n",
        "# Subplot 2: Kurva Akurasi\n",
        "axes[1].plot(epochs_range, [acc * 100 for acc in h[\"accuracy\"]], label=\"Train Accuracy\", color=\"#1f77b4\", linewidth=2)\n",
        "axes[1].plot(epochs_range, [acc * 100 for acc in h[\"val_accuracy\"]], label=\"Validation Accuracy\", color=\"#2ca02c\", linewidth=2, linestyle=\"--\")\n",
        "axes[1].axvline(best_epoch, color=\"green\", linestyle=\":\", label=f\"Best Epoch ({best_epoch})\")\n",
        "axes[1].set_title(f\"Kurva Akurasi Model 2 (Bi-{RNN_TYPE})\", fontsize=12, fontweight=\"bold\")\n",
        "axes[1].set_xlabel(\"Epoch\", fontsize=10)\n",
        "axes[1].set_ylabel(\"Akurasi (%)\", fontsize=10)\n",
        "axes[1].grid(True, linestyle=\"--\", alpha=0.6)\n",
        "axes[1].legend(loc=\"lower right\", frameon=True)\n",
        "\n",
        "plt.suptitle(f\"Evaluasi Pembelajaran Model 2: Bi-{RNN_TYPE} pada Dataset IndoHealth\", fontsize=13, fontweight=\"bold\", y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
}

cell_6_5_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# BAGIAN 6.5: FUNGSI INFERENSI MODEL 2 (PREDIKSI INTENT & CONFIDENCE SCORE)\n",
        "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
        "# ==============================================================================\n",
        "\n",
        "def predict_m2(texts):\n",
        "    \"\"\"\n",
        "    Memprediksi intent dan confidence score untuk daftar kalimat pertanyaan mentah.\n",
        "    \n",
        "    Parameters:\n",
        "        texts (list of str): Daftar pertanyaan mentah dari pengguna.\n",
        "        \n",
        "    Returns:\n",
        "        tuple (list of str, list of float):\n",
        "            - Daftar label intent hasil prediksi (e.g. ['Diabetes', 'Hypertension', ...])\n",
        "            - Daftar nilai confidence score tertinggi [0.0 - 1.0]\n",
        "    \"\"\"\n",
        "    # 1. Terapkan pipeline preprocessing terpadu pada input\n",
        "    teks_bersih = [preprocess(t) for t in texts]\n",
        "    \n",
        "    # 2. Encode teks bersih menjadi sequence integer ber-padding\n",
        "    tensor_input = encode(teks_bersih)\n",
        "    \n",
        "    # 3. Prediksi probabilitas melalui forward pass model\n",
        "    probs = model.predict(tensor_input, verbose=0)\n",
        "    \n",
        "    # 4. Ambil label intent probabilitas tertinggi beserta confidence score-nya\n",
        "    idx_pred = probs.argmax(axis=1)\n",
        "    predicted_labels = [CLASSES[i] for i in idx_pred]\n",
        "    confidence_scores = list(probs.max(axis=1))\n",
        "    \n",
        "    return predicted_labels, confidence_scores\n",
        "\n",
        "# Verifikasi fungsi predict_m2 pada beberapa pertanyaan sampel\n",
        "sampel_pertanyaan = [\n",
        "    \"Bagaimana cara mengontrol gula darah agar tidak semakin tinggi?\",\n",
        "    \"Tekanan darah saya mencapai 150/95, apa obat yang harus saya konsumsi?\",\n",
        "    \"Apakah komplikasi ginjal sering terjadi pada pasien diabetes yang juga menderita hipertensi?\"\n",
        "]\n",
        "\n",
        "pred_label, pred_skor = predict_m2(sampel_pertanyaan)\n",
        "print(\"--- Pengujian Inferensi predict_m2 ---\")\n",
        "for t, l, s in zip(sampel_pertanyaan, pred_label, pred_skor):\n",
        "    print(f\"Kalimat  : \\\"{t}\\\"\")\n",
        "    print(f\"Prediksi : [{l}] | Keyakinan: {s*100:.2f}%\\n\")\n"
    ]
}

cell_6_6_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# BAGIAN 6.6: EKSPERIMEN KOMPARASI ARSITEKTUR BI-GRU VS BI-LSTM (NILAI TAMBAH)\n",
        "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
        "# ==============================================================================\n",
        "\n",
        "print(\"Menjalankan eksperimen komparasi arsitektur: Bi-GRU vs. Bi-LSTM...\")\n",
        "\n",
        "hasil_komparasi = []\n",
        "\n",
        "for jenis_rnn in [\"GRU\", \"LSTM\"]:\n",
        "    tf.keras.backend.clear_session()\n",
        "    tf.random.set_seed(SEED)\n",
        "    \n",
        "    m_exp = build_model(rnn_type=jenis_rnn)\n",
        "    total_params = m_exp.count_params()\n",
        "    \n",
        "    t_start = time.time()\n",
        "    hist_exp = m_exp.fit(\n",
        "        X_train, y_train,\n",
        "        validation_data=(X_val, y_val),\n",
        "        epochs=150,\n",
        "        batch_size=16,\n",
        "        verbose=0,\n",
        "        callbacks=[\n",
        "            tf.keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=20, restore_best_weights=True)\n",
        "        ]\n",
        "    )\n",
        "    durasi = time.time() - t_start\n",
        "    \n",
        "    # Evaluasi pada data validasi\n",
        "    val_preds = m_exp.predict(X_val, verbose=0).argmax(axis=1)\n",
        "    val_acc = accuracy_score(y_val, val_preds)\n",
        "    val_f1 = f1_score(y_val, val_preds, average=\"macro\", zero_division=0)\n",
        "    \n",
        "    hasil_komparasi.append({\n",
        "        \"Arsitektur Model\": f\"Bidirectional {jenis_rnn}\",\n",
        "        \"Total Parameter\": f\"{total_params:,}\",\n",
        "        \"Epoch Konvergen\": len(hist_exp.history[\"loss\"]),\n",
        "        \"Waktu Training (detik)\": round(durasi, 2),\n",
        "        \"Validation Accuracy\": f\"{val_acc * 100:.2f} %\",\n",
        "        \"Validation F1-Macro\": f\"{val_f1 * 100:.2f} %\"\n",
        "    })\n",
        "\n",
        "df_komparasi_rnn = pd.DataFrame(hasil_komparasi)\n",
        "print(\"\\n=== Ringkasan Komparasi Arsitektur Model 2 (Bi-GRU vs Bi-LSTM) ===\")\n",
        "display(df_komparasi_rnn)\n",
        "\n",
        "# Catatan: Variabel `model` dan `history_m2` tetap mempertahankan model utama (Bi-GRU)\n",
        "# untuk evaluasi downstream pada Bagian 7, 8, 9, 10, 11, dan 12.\n"
    ]
}

# Replace cells 12..17 with the new list of cells
new_b6_cells = [
    cell_6_markdown,
    cell_6_1_code,
    cell_6_2_code,
    cell_6_3_code,
    cell_6_4_code,
    cell_6_5_code,
    cell_6_6_code
]

nb['cells'] = nb['cells'][:12] + new_b6_cells + nb['cells'][18:]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Berhasil memperbarui {file_path}! Total cells sekarang: {len(nb['cells'])}")
