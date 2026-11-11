# Dự Án Phân Tích & Đối Sánh Chất Lượng Rượu Vang (Wine Quality Analysis)

Dự án nghiên cứu, làm sạch, trực quan hóa và đối sánh thống kê chuyên sâu trên tập dữ liệu chuẩn **Wine Quality** từ UCI Machine Learning Repository (bao gồm 6,497 mẫu rượu vang đỏ và vang trắng).

---

## 📁 Cấu Trúc Dự Án

```
bom_ruou/
├── data/
│   ├── chatluong_ruougoc.csv           # Dữ liệu gốc gộp từ UCI (Được bảo tồn nguyên vẹn 100%)
│   ├── duleu_chuan.feather             # Dữ liệu đã làm sạch & chuẩn hóa (Apache Arrow)
│   ├── bao_cao_doi_sanh_thong_ke.csv   # Báo cáo ANOVA & Kruskal-Wallis
│   ├── duleu_lech_moi_truong.feather   # Dữ liệu mô phỏng lệch môi trường & điểm tiềm năng
│   ├── duleu_lech_moi_truong.csv       # File CSV dữ liệu lệch môi trường
│   ├── bao_cao_lech_moi_truong.csv     # Báo cáo kiểm định Data Drift (KS & Wasserstein)
│   └── ma_tran_chuyen_doi_chat_luong.csv # Ma trận chuyển dịch bậc chất lượng
├── bieu_do/                            # Bộ sưu tập biểu đồ độ phân giải cao (300 DPI)
│   ├── 01_phan_phoi_chat_luong.png     # Phân bố điểm số (3-9) và tỷ lệ 3 bậc
│   ├── 02_so_sanh_vang_do_trang.png    # Đối sánh đặc tính hóa lý Vang đỏ vs Vang trắng
│   ├── 03_ma_tran_tuong_quan.png       # Ma trận tương quan Pearson đa biến
│   ├── 04_yeu_to_quyet_dinh_chat_luong.png # Boxplot/Violinplot các yếu tố then chốt
│   ├── 05_phan_phoi_11_chi_so_hoa_ly.png   # Phân phối tần suất 11 chỉ số sau Capping
│   ├── 06_phan_phoi_cot_hang.png       # [Kaggle Style] Phân phối theo lưới cột và hàng (plotPerColumnDistribution)
│   ├── 07_ma_tran_tuong_quan_matshow.png # [Kaggle Style] Ma trận tương quan matshow kèm thước màu (plotCorrelationMatrix)
│   ├── 08_phan_tan_va_mat_do.png       # [Kaggle Style] Ma trận phân tán & mật độ KDE kèm r (plotScatterMatrix)
│   └── 09_mo_phong_lech_moi_truong.png # Đối sánh đa chiều độ lệch hóa lý & điểm tiềm năng tái xác lập
├── src/                                # Mã nguồn module hóa
│   ├── __init__.py
│   ├── data_loader.py                  # Tải dữ liệu UCI, nạp/lưu CSV và Feather
│   ├── cleaner.py                      # Làm sạch, Winsorization Capping, phân 3 bậc
│   ├── visualizer.py                   # Vẽ biểu đồ chuẩn tiếng Việt bằng Seaborn/Matplotlib
│   └── environment_simulator.py       # Công cụ mô phỏng độ lệch môi trường & tái thiết lập điểm tiềm năng
├── notebooks/                          # Bộ Jupyter Notebooks tương tác
│   ├── data_cleaning_preprocessing.ipynb # Quy trình làm sạch dữ liệu chi tiết
│   ├── exploratory_data_analysis.ipynb   # Khám phá EDA đa chiều & giải đáp câu hỏi
│   ├── data_export_and_benchmark.ipynb   # Đối sánh thống kê ANOVA & Hồ sơ vàng
│   └── environmental_drift_simulation.ipynb # Khám phá tương tác mô phỏng môi trường & điểm tiềm năng
├── run_pipeline.py                     # Script chạy tự động toàn bộ 5 bước quy trình chỉ với 1 dòng lệnh
└── README.md
```

---

## 🚀 Hướng Dẫn Vận Hành

### Cách 1: Chạy tự động toàn bộ quy trình chỉ với 1 dòng lệnh
Chỉ cần mở terminal tại thư mục dự án và chạy:
```bash
python run_pipeline.py
```
Hệ thống sẽ tự động:
1. Nạp và kiểm tra dữ liệu gốc từ kho UCI.
2. Thực hiện làm sạch, xử lý ngoại lai (Capping 1% - 99%) và phân loại 3 bậc chất lượng.
3. Xuất toàn bộ 8 biểu đồ chất lượng cao 300 DPI vào thư mục `bieu_do/`.
4. Tính toán kiểm định giả thuyết thống kê (ANOVA F-test, Kruskal-Wallis H-test) và in bảng đối sánh trực tiếp trên terminal.
5. Mô phỏng độ lệch môi trường (Sốc nhiệt, Oxy hóa, Lệch cảm biến, Lão hóa), tái xác lập điểm chất lượng tiềm năng, xuất biểu đồ 09 và lưu dữ liệu an toàn vào `data/duleu_lech_moi_truong.*` (tuyệt đối không đè file thô `chatluong_ruougoc.csv`).

### Cách 2: Khám phá trực quan qua Jupyter Notebooks
Khởi động Jupyter Notebook:
```bash
jupyter notebook notebooks/
```
Bạn có thể mở tuần tự 4 notebook:
1. `data_cleaning_preprocessing.ipynb`: Xem chi tiết từng bước làm sạch và chuyển đổi.
2. `exploratory_data_analysis.ipynb`: Khám phá các mối liên hệ hóa học và trực quan hóa tương quan.
3. `data_export_and_benchmark.ipynb`: Khảo sát kiểm định giả thuyết thống kê và trích xuất hồ sơ vàng của rượu thượng hạng.
4. `environmental_drift_simulation.ipynb`: Tương tác trực quan mô phỏng độ lệch môi trường và phân tích điểm tiềm năng.

---

## 🧪 Phân Loại 3 Bậc Chất Lượng Rượu Vang

- **Kém (Low/Poor):** Điểm chất lượng $3 - 4$ (Chiếm ~3.79%)
- **Trung bình (Medium/Normal):** Điểm chất lượng $5 - 6$ (Chiếm ~76.56%)
- **Thượng hạng (High/Premium):** Điểm chất lượng $7 - 9$ (Chiếm ~19.66%)

---

## 📊 Những Phát Hiện Then Chốt (Key Scientific Insights)

1. **Độ cồn (`do_con`) là yếu tố phân hóa mạnh nhất:**
   - Rượu Thượng hạng có độ cồn trung bình đạt **11.42%**, cao hơn đáng kể so với mức ~10.2% của rượu Kém và Trung bình ($F = 585.45, p < 10^{-200}$).
2. **Axit bay hơi (`axit_bay_hoi`) là chỉ số kiểm soát âm tính quan trọng nhất:**
   - Hàm lượng axit bay hơi (axit axetic gây mùi giấm chua) ở rượu Thượng hạng rất thấp và ổn định (trung bình **0.289 g/dm³**), trong khi ở rượu Kém cao gấp 1.5 lần (**0.453 g/dm³**).
3. **Sự khác biệt bản chất giữa Vang Đỏ và Vang Trắng:**
   - Vang trắng có hàm lượng **SO₂ tổng số** cao hơn gấp đôi vang đỏ để chống oxy hóa (do không có tannin từ vỏ nho).
   - Vang đỏ có hàm lượng **axit cố định** và **sunfat** cao hơn để bảo toàn cấu trúc vị chát.
