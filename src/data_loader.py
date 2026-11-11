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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "chatluong_ruougoc.csv"
CLEANED_DATA_PATH = DATA_DIR / "duleu_chuan.feather"

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

    if duong_dan_luu.exists() and duong_dan_luu.stat().st_size > 0 and not bat_buoc_tai_lai:
        print(f"[INFO] Dữ liệu gốc đã tồn tại tại: {duong_dan_luu}. Đang đọc từ ổ đĩa...")
        return doc_du_lieu_goc(duong_dan_luu)

    print("[INFO] Đang tải dữ liệu vang đỏ và vang trắng từ UCI Machine Learning Repository...")

    df_do = pd.read_csv(URL_VANG_DO, sep=";")
    df_do["loai_ruou"] = "Vang đỏ"

    df_trang = pd.read_csv(URL_VANG_TRANG, sep=";")
    df_trang["loai_ruou"] = "Vang trắng"

    df_gop = pd.concat([df_do, df_trang], ignore_index=True)
    print(f"[INFO] Tải thành công! Vang đỏ: {len(df_do)} mẫu, Vang trắng: {len(df_trang)} mẫu. Tổng: {len(df_gop)} mẫu.")

    df_gop.to_csv(duong_dan_luu, index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã lưu dữ liệu gốc vào: {duong_dan_luu}")

    return df_gop


def doc_du_lieu_goc(duong_dan: Optional[Path] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu thô từ file CSV.
    Hỗ trợ tìm kiếm thông minh tại data/chatluong_ruougoc.csv hoặc data/raw/.
    Nếu file chưa có hoặc rỗng thì tự động tải từ UCI.

    Tham số:
        duong_dan: Đường dẫn file CSV (mặc định tìm tại data/chatluong_ruougoc.csv).

    Trả về:
        DataFrame dữ liệu thô.
    """
    if duong_dan is None:
        if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0:
            duong_dan = RAW_DATA_PATH
        elif (DATA_DIR / "raw" / "chatluong_ruougoc.csv").exists():
            duong_dan = DATA_DIR / "raw" / "chatluong_ruougoc.csv"
        elif (DATA_DIR / "raw" / "winequality-red.csv").exists():
            duong_dan = DATA_DIR / "raw" / "winequality-red.csv"
        else:
            duong_dan = RAW_DATA_PATH

    duong_dan = Path(duong_dan)
    if not duong_dan.exists() or duong_dan.stat().st_size == 0:
        print("[CẢNH BÁO] File dữ liệu gốc không tồn tại hoặc rỗng. Tiến hành tự động tải từ UCI...")
        return tai_du_lieu_uci(duong_dan)

    try:
        df = pd.read_csv(duong_dan, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(duong_dan, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(duong_dan, encoding="utf-8")

    return df


def doc_du_lieu_vang_do() -> pd.DataFrame:
    """
    Đọc riêng tập dữ liệu Vang đỏ (Red Wine Quality) của Nhóm 6 từ data/raw/winequality-red.csv.
    """
    duong_dan_red = DATA_DIR / "raw" / "winequality-red.csv"
    if not duong_dan_red.exists():
        raise FileNotFoundError(f"Không tìm thấy file vang đỏ tại: {duong_dan_red}")
    try:
        df = pd.read_csv(duong_dan_red)
        if df.shape[1] == 1:
            # Nếu chỉ có 1 cột, thử đọc lại bằng dấu chấm phẩy (định dạng UCI gốc)
            df = pd.read_csv(duong_dan_red, sep=";")
    except Exception:
        df = pd.read_csv(duong_dan_red, sep=";")
    return df


def luu_du_lieu_chuan(df: pd.DataFrame, duong_dan: Optional[Path] = None) -> Path:
    """
    Lưu DataFrame đã làm sạch vào định dạng nén nhị phân Feather (pyarrow).
    Tự động đồng bộ ra cả data/duleu_chuan.feather và data/processed/duleu_chuan.feather.

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

    df_luu = df.reset_index(drop=True)
    df_luu.to_feather(duong_dan)
    print(f"[INFO] Đã lưu dữ liệu chuẩn hóa ({len(df_luu)} dòng, {df_luu.shape[1]} cột) vào: {duong_dan}")

    processed_path = DATA_DIR / "processed" / "duleu_chuan.feather"
    dulieu_path = DATA_DIR / "dulieu_chuan.feather"
    if duong_dan.resolve() != processed_path.resolve():
        try:
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df_luu.to_feather(processed_path)
            print(f"[INFO] Đã đồng bộ bản sao sang: {processed_path}")
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể đồng bộ bản sao processed: {e}")

    if duong_dan.resolve() != dulieu_path.resolve():
        try:
            df_luu.to_feather(dulieu_path)
            print(f"[INFO] Đã đồng bộ bản sao sang: {dulieu_path}")
        except Exception as e:
            pass

    return duong_dan


def doc_du_lieu_chuan(duong_dan: Optional[Path] = None) -> pd.DataFrame:
    """
    Đọc dữ liệu chuẩn hóa từ file Feather.
    Hỗ trợ tìm tại data/duleu_chuan.feather hoặc data/processed/duleu_chuan.feather.

    Tham số:
        duong_dan: Đường dẫn file .feather (mặc định data/duleu_chuan.feather).

    Trả về:
        DataFrame dữ liệu chuẩn hóa.
    """
    if duong_dan is None:
        if CLEANED_DATA_PATH.exists() and CLEANED_DATA_PATH.stat().st_size > 0:
            duong_dan = CLEANED_DATA_PATH
        elif (DATA_DIR / "dulieu_chuan.feather").exists() and (DATA_DIR / "dulieu_chuan.feather").stat().st_size > 0:
            duong_dan = DATA_DIR / "dulieu_chuan.feather"
        elif (DATA_DIR / "processed" / "duleu_chuan.feather").exists():
            duong_dan = DATA_DIR / "processed" / "duleu_chuan.feather"
        elif (DATA_DIR / "processed" / "dulieu_chuan.feather").exists():
            duong_dan = DATA_DIR / "processed" / "dulieu_chuan.feather"
        else:
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
