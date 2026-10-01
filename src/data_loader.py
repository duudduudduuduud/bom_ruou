"""
Module: data_loader.py
Mô tả: Quản lý nạp, tải và lưu trữ dữ liệu rượu vang (Wine Quality).
Hỗ trợ tự động tải từ kho lưu trữ UCI, đọc/ghi file CSV và định dạng nén Feather.
"""

import os
from pathlib import Path
from typing import Optional
import urllib.request
import pandas as pd

# Đường dẫn mặc định của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "chatluong_ruougoc.csv"
CLEANED_DATA_PATH = DATA_DIR / "duleu_chuan.feather"

# URL kho dữ liệu UCI
URL_VANG_DO = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
URL_VANG_TRANG = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def tai_du_lieu_uci(duong_dan_luu: Optional[Path] = None, bat_buoc_tai_lai: bool = False) -> pd.DataFrame:
    """
    Tải bộ dữ liệu Vang đỏ và Vang trắng từ kho dữ liệu UCI Machine Learning Repository.
    Gộp 2 tập dữ liệu, bổ sung cột phân loại 'loai_ruou' và lưu ra file CSV.

    Tham số:
        duong_dan_luu: Đường dẫn lưu file CSV thô (mặc định data/chatluong_ruougoc.csv).
        bat_buoc_tai_lai: Nếu True, tải lại dù file đã tồn tại.

    Trả về:
        DataFrame chứa dữ liệu gốc đã gộp của cả vang đỏ và vang trắng.
    """
    if duong_dan_luu is None:
        duong_dan_luu = RAW_DATA_PATH

    duong_dan_luu = Path(duong_dan_luu)
    duong_dan_luu.parent.mkdir(parents=True, exist_ok=True)

    # Kiểm tra nếu file đã có dữ liệu hợp lệ (> 0 bytes) và không bắt buộc tải lại
    if duong_dan_luu.exists() and duong_dan_luu.stat().st_size > 0 and not bat_buoc_tai_lai:
        print(f"[INFO] Dữ liệu gốc đã tồn tại tại: {duong_dan_luu}. Đang đọc từ ổ đĩa...")
        return doc_du_lieu_goc(duong_dan_luu)

    print("[INFO] Đang tải dữ liệu vang đỏ và vang trắng từ UCI Machine Learning Repository...")

    # Đọc dữ liệu từ URL (UCI dùng dấu chấm phẩy ';' làm dấu phân cách)
    df_do = pd.read_csv(URL_VANG_DO, sep=";")
    df_do["loai_ruou"] = "Vang đỏ"

    df_trang = pd.read_csv(URL_VANG_TRANG, sep=";")
    df_trang["loai_ruou"] = "Vang trắng"

    # Gộp 2 tập dữ liệu
    df_gop = pd.concat([df_do, df_trang], ignore_index=True)
    print(f"[INFO] Tải thành công! Vang đỏ: {len(df_do)} mẫu, Vang trắng: {len(df_trang)} mẫu. Tổng: {len(df_gop)} mẫu.")

    # Lưu ra file CSV gốc với encoding utf-8-sig để hỗ trợ tiếng Việt trong Excel
    df_gop.to_csv(duong_dan_luu, index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã lưu dữ liệu gốc vào: {duong_dan_luu}")

    return df_gop


def doc_du_lieu_goc(duong_dan: Optional[Path] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu thô từ file CSV. Nếu file chưa có hoặc rỗng thì tự động tải.

    Tham số:
        duong_dan: Đường dẫn file CSV (mặc định data/chatluong_ruougoc.csv).

    Trả về:
        DataFrame dữ liệu thô.
    """
    if duong_dan is None:
        duong_dan = RAW_DATA_PATH

    duong_dan = Path(duong_dan)
    if not duong_dan.exists() or duong_dan.stat().st_size == 0:
        print("[CẢNH BÁO] File dữ liệu gốc không tồn tại hoặc rỗng. Tiến hành tự động tải từ UCI...")
        return tai_du_lieu_uci(duong_dan)

    # Đọc với dự phòng delimiter
    try:
        df = pd.read_csv(duong_dan, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(duong_dan, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(duong_dan, encoding="utf-8")

    return df


def luu_du_lieu_chuan(df: pd.DataFrame, duong_dan: Optional[Path] = None) -> Path:
    """
    Lưu DataFrame đã làm sạch vào định dạng nén nhị phân Feather (pyarrow).

    Tham số:
        df: DataFrame đã làm sạch và chuẩn hóa.
        duong_dan: Đường dẫn lưu file .feather (mặc định data/duleu_chuan.feather).

    Trả về:
        Đường dẫn file đã lưu.
    """
    if duong_dan is None:
        duong_dan = CLEANED_DATA_PATH

    duong_dan = Path(duong_dan)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)

    # Đảm bảo reset_index trước khi lưu feather
    df_luu = df.reset_index(drop=True)
    df_luu.to_feather(duong_dan)
    print(f"[INFO] Đã lưu dữ liệu chuẩn hóa ({len(df_luu)} dòng, {df_luu.shape[1]} cột) vào: {duong_dan}")
    return duong_dan


def doc_du_lieu_chuan(duong_dan: Optional[Path] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu chuẩn hóa từ file Feather.

    Tham số:
        duong_dan: Đường dẫn file .feather (mặc định data/duleu_chuan.feather).

    Trả về:
        DataFrame dữ liệu chuẩn hóa.
    """
    if duong_dan is None:
        duong_dan = CLEANED_DATA_PATH

    duong_dan = Path(duong_dan)
    if not duong_dan.exists() or duong_dan.stat().st_size == 0:
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu chuẩn hóa tại {duong_dan}. "
            "Vui lòng chạy quy trình làm sạch (cleaner.py) trước!"
        )

    return pd.read_feather(duong_dan)


if __name__ == "__main__":
    df = tai_du_lieu_uci(bat_buoc_tai_lai=True)
    print("Mẫu dữ liệu nạp được:")
    print(df.head())
    print("\nThông tin kích thước:", df.shape)
