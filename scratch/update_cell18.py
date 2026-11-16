import json

file_path = 'Pertemuan 11/Chatbot_Pertemuan11.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 18 (Index 18)
cell_18_source = [
    "# ==============================================================================\n",
    "# BAGIAN 6.6: EKSPERIMEN KOMPARASI ARSITEKTUR BI-GRU VS BI-LSTM (NILAI TAMBAH)\n",
    "# Pengembang: Asyam Sakhi Maulana (240535600363)\n",
    "# ==============================================================================\n",
    "\n",
    "print(\"Menjalankan eksperimen komparasi arsitektur: Bi-GRU vs. Bi-LSTM...\")\n",
    "\n",
    "# 1. Evaluasi model utama yang telah dilatih (Bi-GRU) pada data validasi\n",
    "val_preds_primary = model.predict(X_val, verbose=0).argmax(axis=1)\n",
    "val_acc_primary = accuracy_score(y_val, val_preds_primary)\n",
    "val_f1_primary = f1_score(y_val, val_preds_primary, average=\"macro\", zero_division=0)\n",
    "\n",
    "hasil_komparasi = [{\n",
    "    \"Arsitektur Model\": f\"Bidirectional {RNN_TYPE} (Model Utama)\",\n",
    "    \"Total Parameter\": f\"{model.count_params():,}\",\n",
    "    \"Epoch Konvergen\": len(history_m2.history[\"loss\"]),\n",
    "    \"Waktu Training (detik)\": round(train_time_m2, 2),\n",
    "    \"Validation Accuracy\": f\"{val_acc_primary * 100:.2f} %\",\n",
    "    \"Validation F1-Macro\": f\"{val_f1_primary * 100:.2f} %\"\n",
    "}]\n",
    "\n",
    "# 2. Melatih arsitektur alternatif pembanding (Bi-LSTM) dengan konfigurasi identik\n",
    "alt_rnn = \"LSTM\" if RNN_TYPE == \"GRU\" else \"GRU\"\n",
    "print(f\"Melatih model alternatif pembanding: Bi-{alt_rnn}...\")\n",
    "\n",
    "model_alt = build_model(rnn_type=alt_rnn)\n",
    "t_start = time.time()\n",
    "hist_alt = model_alt.fit(\n",
    "    X_train, y_train,\n",
    "    validation_data=(X_val, y_val),\n",
    "    epochs=200,\n",
    "    batch_size=16,\n",
    "    verbose=0,\n",
    "    callbacks=[\n",
    "        tf.keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=20, restore_best_weights=True)\n",
    "    ]\n",
    ")\n",
    "durasi_alt = time.time() - t_start\n",
    "\n",
    "val_preds_alt = model_alt.predict(X_val, verbose=0).argmax(axis=1)\n",
    "val_acc_alt = accuracy_score(y_val, val_preds_alt)\n",
    "val_f1_alt = f1_score(y_val, val_preds_alt, average=\"macro\", zero_division=0)\n",
    "\n",
    "hasil_komparasi.append({\n",
    "    \"Arsitektur Model\": f\"Bidirectional {alt_rnn} (Pembanding)\",\n",
    "    \"Total Parameter\": f\"{model_alt.count_params():,}\",\n",
    "    \"Epoch Konvergen\": len(hist_alt.history[\"loss\"]),\n",
    "    \"Waktu Training (detik)\": round(durasi_alt, 2),\n",
    "    \"Validation Accuracy\": f\"{val_acc_alt * 100:.2f} %\",\n",
    "    \"Validation F1-Macro\": f\"{val_f1_alt * 100:.2f} %\"\n",
    "})\n",
    "\n",
    "df_komparasi_rnn = pd.DataFrame(hasil_komparasi)\n",
    "print(\"\\n=== Ringkasan Komparasi Arsitektur Model 2 (Bi-GRU vs Bi-LSTM) ===\")\n",
    "display(df_komparasi_rnn)\n"
]

nb['cells'][18]['source'] = cell_18_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Berhasil memperbarui Cell 18 (Bagian 6.6)!")
