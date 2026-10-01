"""
Script: run_pipeline.py
Mô tả: Kịch bản điều phối toàn bộ quy trình tự động chỉ với 1 dòng lệnh:
    python run_pipeline.py

Các công đoạn tự động:
1. Tải và kiểm tra dữ liệu thô (UCI Wine Quality) -> data/chatluong_ruougoc.csv
2. Làm sạch, khử ngoại lai Capping, phân 3 bậc chất lượng -> data/duleu_chuan.feather
3. Tạo và lưu bộ sưu tập 5 biểu đồ trực quan EDA 300 DPI -> bieu_do/
4. Thực hiện kiểm định thống kê ANOVA / Kruskal-Wallis và xuất báo cáo -> data/bao_cao_doi_sanh_thong_ke.csv
5. In bảng tổng kết đối sánh trực quan ra màn hình console.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Thêm đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import tai_du_lieu_uci, doc_du_lieu_chuan
from src.cleaner import lam_sach_toan_dien, CAC_COT_HOA_LY
from src.visualizer import ve_tat_ca_bieu_do


def in_dong_ke(ky_tu="=", do_dai=78):
    print(ky_tu * do_dai)


def in_tieu_de(tieu_de: str):
    in_dong_ke("=")
    print(f" {tieu_de.upper()}")
    in_dong_ke("=")


def chay_doi_sanh_thong_ke(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thực hiện kiểm định giả thuyết thống kê (ANOVA F-test và Kruskal-Wallis)
    giữa 3 bậc chất lượng rượu vang và xuất báo cáo.
    """
    nhom_kem = df[df["bac_chat_luong"] == "Kém"]
    nhom_tb = df[df["bac_chat_luong"] == "Trung bình"]
    nhom_thuong_hang = df[df["bac_chat_luong"] == "Thượng hạng"]

    ket_qua = []
    for cot in CAC_COT_HOA_LY:
        # Giá trị trung bình của từng nhóm
        mean_kem = nhom_kem[cot].mean()
        mean_tb = nhom_tb[cot].mean()
        mean_th = nhom_thuong_hang[cot].mean()

        # Kiểm định ANOVA
        f_stat, p_anova = stats.f_oneway(nhom_kem[cot], nhom_tb[cot], nhom_thuong_hang[cot])
        # Kiểm định Kruskal-Wallis
        h_stat, p_kw = stats.kruskal(nhom_kem[cot], nhom_tb[cot], nhom_thuong_hang[cot])

        ket_qua.append({
            "Chỉ số hóa lý": cot,
            "Kém (Mean)": round(mean_kem, 3),
            "Trung bình (Mean)": round(mean_tb, 3),
            "Thượng hạng (Mean)": round(mean_th, 3),
            "ANOVA F-stat": round(f_stat, 2),
            "ANOVA p-value": f"{p_anova:.2e}",
            "Kruskal H-stat": round(h_stat, 2),
            "Ý nghĩa (p<0.05)": "Có" if p_kw < 0.05 else "Không"
        })

    df_kq = pd.DataFrame(ket_qua).sort_values(by="ANOVA F-stat", ascending=False).reset_index(drop=True)
    return df_kq


def main():
    thoi_gian_bat_dau = time.time()
    in_tieu_de("QUY TRÌNH TỰ ĐỘNG PHÂN TÍCH & ĐỐI SÁNH CHẤT LƯỢNG RƯỢU VANG")

    # ----------------------------------------------------
    # BƯỚC 1: TẢI & NẠP DỮ LIỆU GỐC
    # ----------------------------------------------------
    print("\n[BƯỚC 1/4] KIỂM TRA VÀ NẠP DỮ LIỆU THÔ TỪ UCI...")
    df_goc = tai_du_lieu_uci(bat_buoc_tai_lai=False)
    print(f" -> Nạp thành công {len(df_goc)} mẫu dữ liệu thô.")

    # ----------------------------------------------------
    # BƯỚC 2: TIỀN XỬ LÝ & LÀM SẠCH DỮ LIỆU
    # ----------------------------------------------------
    print("\n[BƯỚC 2/4] LÀM SẠCH, CAPPING NGOẠI LAI VÀ PHÂN BẬC CHẤT LƯỢNG...")
    df_chuan = lam_sach_toan_dien(df_goc, loai_bo_trung_lap=False, luu_file=True)
    print(f" -> Dữ liệu chuẩn hóa đã được lưu vào data/duleu_chuan.feather.")

    # ----------------------------------------------------
    # BƯỚC 3: TRỰC QUAN HÓA & XUẤT BIỂU ĐỒ VÀO bieu_do/
    # ----------------------------------------------------
    print("\n[BƯỚC 3/4] XUẤT 8 BIỂU ĐỒ PHÂN TÍCH CHUYÊN NGHIỆP VÀO THƯ MỤC bieu_do/...")
    danh_sach_anh = ve_tat_ca_bieu_do(df_chuan)
    print(f" -> Đã xuất thành công {len(danh_sach_anh)} biểu đồ 300 DPI:")
    for anh in danh_sach_anh:
        print(f"    * {anh.name}")

    # ----------------------------------------------------
    # BƯỚC 4: ĐỐI SÁNH THỐNG KÊ (STATISTICAL BENCHMARKING)
    # ----------------------------------------------------
    print("\n[BƯỚC 4/4] ĐỐI SÁNH THỐNG KÊ VÀ KIỂM ĐỊNH GIẢ THUYẾT (ANOVA & KRUSKAL)...")
    df_doi_sanh = chay_doi_sanh_thong_ke(df_chuan)

    duong_dan_bao_cao = PROJECT_ROOT / "data" / "bao_cao_doi_sanh_thong_ke.csv"
    df_doi_sanh.to_csv(duong_dan_bao_cao, index=False, encoding="utf-8-sig")
    print(f" -> Đã lưu báo cáo thống kê đối sánh vào: {duong_dan_bao_cao}\n")

    in_dong_ke("-")
    print(" BẢNG ĐỐI SÁNH THỐNG KÊ THEO 3 BẬC CHẤT LƯỢNG (SẮP XẾP THEO MỨC ĐỘ KHÁC BIỆT)")
    in_dong_ke("-")
    print(df_doi_sanh.to_string(index=False))
    in_dong_ke("-")

    # ----------------------------------------------------
    # TỔNG KẾT
    # ----------------------------------------------------
    thoi_gian_chay = time.time() - thoi_gian_bat_dau
    in_tieu_de(f"HOÀN TẤT TOÀN BỘ QUY TRÌNH THÀNH CÔNG (THỜI GIAN: {thoi_gian_chay:.2f} GIÂY)")
    print("\nCác sản phẩm đã tạo:")
    print(" 1. Dữ liệu chuẩn hóa:     data/duleu_chuan.feather")
    print(" 2. Báo cáo đối sánh CSV:  data/bao_cao_doi_sanh_thong_ke.csv")
    print(" 3. Bộ sưu tập biểu đồ:    bieu_do/ (01 đến 08)")
    print(" 4. Bộ 3 Notebooks:        notebooks/*.ipynb")
    print("\nBạn có thể khởi động Jupyter Notebook để khám phá trực quan từng bước:")
    print("    jupyter notebook notebooks/")
    in_dong_ke("=")


if __name__ == "__main__":
    main()
