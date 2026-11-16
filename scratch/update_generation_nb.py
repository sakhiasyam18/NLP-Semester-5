import json

file_path = 'pertemuan 12/templatenyaText_Generation.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0: Header with team information
cell_0_text = """# 📝 MINI PROJECT: TEXT GENERATION BAHASA INDONESIA
## Komparasi Language Model: LSTM vs. GRU (Next-Word Prediction)
**Tugas Lanjutan Pertemuan 11 & 12 — Pemrosesan Bahasa Alami (NLP)**  
**Program Studi:** S1 Teknik Informatika — Universitas Negeri Malang (UM)  
**Dosen Pengampu:** Prof. Ir. Aji Prasetya Wibawa, S.T., M.M.T., Ph.D. & M. Zainal Arifin, S.Si., M.Kom., Ph.D.  
**Mentor Asisten:** Agung Bella Putra Utama & Fadia Irsania Putri  

---

### 👥 Pembagian Peran & Job Desk Tim (5 Anggota)

| No | NIM | Nama Mahasiswa | Peran & Tanggung Jawab Utama |
| :-: | :---: | :--- | :--- |
| **1** | `240535601034` | **ALBI ADWITIYA CHANDRA** | **Pengembang Model 1 (LSTM Language Model):**<br>• Membangun layer arsitektur LSTM (`layers.LSTM(RNN_UNITS)`).<br>• Pelatihan model LSTM dengan Early Stopping & pencatatan durasi training.<br>• Pengujian generasi teks LSTM pada berbagai tingkat temperature (0.5 - 1.2). |
| **2** | `240535610197` | **INDAH HARIYANTI** | **Pengembang Model 2 (GRU Language Model):**<br>• Membangun layer arsitektur GRU (`layers.GRU(RNN_UNITS)`).<br>• Pelatihan model GRU dengan Early Stopping & pencatatan durasi training.<br>• Analisis efisiensi komputasi & parameter GRU dibandingkan LSTM. |
| **3** | `240535604624` | **CHRISTIAN LIONEL DAUN DARMA** | **Data Preparation, Evaluator & Kesimpulan:**<br>• Pemuatan & kurasi dataset `indonesian-nlp/wikipedia-id` dari HuggingFace.<br>• Evaluasi kuantitatif: Training/Validation Loss, Perplexity, dan Next-Word Accuracy.<br>• Evaluasi kualitatif 10 prompt (relevansi, koherensi, repetisi) & penulisan kesimpulan. |
| **4** | `240535600363` | **ASYAM SAKHI MAULANA** | **Mini Project Text Summarization:** Model 1 (TF-IDF Sentence Ranking). |
| **5** | `240535607403` | **AGUSTINUS MARIA AGUNG HERIYANTO** | **Mini Project Text Summarization:** Model 2 (TextRank Algorithm). |

---

### 🎯 Tujuan & Aspek Komparasi Sesuai Slide 21 PDF
Masing-masing kelompok wajib membandingkan kedua model generasi teks dari sisi:
1. **Training Loss dan Validation Loss**
2. **Perplexity** (Tingkat ketidakpastian model dalam menebak kata berikutnya)
3. **Next-Word Accuracy** (Akurasi prediksi kata berikutnya pada test set)
4. **Waktu Training** (Efisiensi komputasi detik per epoch)
5. **Kualitas Teks Hasil Generasi** (Gramatikal, keluwesan bahasa Indonesia)
6. **Koherensi dan Relevansi terhadap Prompt Awal**
7. **Tingkat Repetisi Kata** (Kecenderungan looping / pengulangan frasa)
8. **Pengaruh Hyperparameter Temperature** (0.5, 0.8, 1.0, 1.2)
9. **Contoh Output Terbaik dan Terburuk**
10. **Analisis Error & Jawaban Pertanyaan:** *Apakah model dengan perplexity lebih rendah selalu menghasilkan teks yang lebih natural?*

*Catatan: Sangat disarankan mengaktifkan akselerator GPU di Google Colab: `Runtime > Change runtime type > T4 GPU`.*
"""

nb['cells'][0]['source'] = [l + '\n' for l in cell_0_text.strip().split('\n')]

# Add !pip install datasets at the top of cell 2 if not present
cell_2_code = ''.join(nb['cells'][2]['source'])
if 'pip install' not in cell_2_code:
    cell_2_code = "# Instalasi datasets Hugging Face jika belum terpasang\n!pip install --quiet datasets\n\nfrom datasets import load_dataset\n" + cell_2_code
nb['cells'][2]['source'] = [l + '\n' for l in cell_2_code.strip().split('\n')]

# Update Cell 11: provide both LSTM and GRU branches cleanly
cell_11_code = """def build_model(kind):
    inp = layers.Input(shape=(max_len-1,))
    x = layers.Embedding(vocab_size, EMB_DIM)(inp)
    if kind == 'LSTM':
        # Model 1 (Albi): Long Short-Term Memory
        x = layers.LSTM(RNN_UNITS)(x)
    elif kind == 'GRU':
        # Model 2 (Indah): Gated Recurrent Unit
        x = layers.GRU(RNN_UNITS)(x)
    else:
        raise ValueError(f"Model {kind} tidak dikenali.")
    out = layers.Dense(vocab_size, activation='softmax')(x)
    m = models.Model(inp, out, name=f"LanguageModel_{kind}")
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m

# Inisialisasi Model 1 (LSTM - Albi)
lstm_model = build_model('LSTM')
lstm_model.summary()

# Inisialisasi Model 2 (GRU - Indah)
gru_model = build_model('GRU')
gru_model.summary()
"""
nb['cells'][11]['source'] = [l + '\n' for l in cell_11_code.strip().split('\n')]

# Update Cell 16: activate GRU training
cell_16_code = """# ==============================================================================
# Model 2 — GRU (Penanggung Jawab: Indah Hariyanti - 240535610197)
# ==============================================================================
cb_gru = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
t0 = time.time()
h2 = gru_model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb_gru)
time_gru = time.time() - t0
print(f"Waktu training GRU: {time_gru:.1f} detik")
"""
nb['cells'][16]['source'] = [l + '\n' for l in cell_16_code.strip().split('\n')]

output_gen_path = 'pertemuan 12/Mini_Project_Text_Generation.ipynb'
with open(output_gen_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Berhasil membuat {output_gen_path} dengan {len(nb['cells'])} cells!")
