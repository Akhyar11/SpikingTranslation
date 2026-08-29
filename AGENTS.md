# Spiking Seq2Seq Project Rules

Ini adalah aturan wajib bagi AI Agent yang berkontribusi dalam repository ini:

1. **Step-by-Step Execution**: Pengerjaan proyek harus patuh pada urutan fase yang ada di file `task.md`. Dilarang melompat ke komponen lain sebelum fase sebelumnya dinyatakan selesai dan diuji.
2. **Task Tracking Mandatory**: Sebelum memulai sebuah tugas, kamu WAJIB membaca file `task.md` untuk mengetahui status terakhir. Setiap kali kamu menyelesaikan sub-task atau fase, kamu WAJIB meng-update (memperbarui) file `task.md` dengan memberikan tanda centang `[x]` pada task yang bersangkutan.
3. **Test Correctness Mandatory**: Setiap kali ada penambahan fungsi atau perubahan logika, wajib membuat unit test (`#[cfg(test)]`) di dalam file yang bersangkutan dan menjalankan `cargo test` untuk memvalidasinya. Dilarang menganggap kode "berhasil" tanpa bukti output test yang lulus.
4. **Rust Best Practices**: Gunakan fitur Rust dengan optimal (ownership, type safety). Karena SNN membutuhkan presisi matematika yang tinggi, pastikan tidak ada *silent overflow* atau *shape mismatch* pada matrix/tensor (ndarray).
5. **No Framework Cheating**: Jangan mengimpor `Softmax`, `CrossEntropyLoss` bawaan dari framework luar. Semua *forward* dan *backward pass* harus mengikuti dokumen matematika kustom kita, terutama pada Delta-BPTT.
