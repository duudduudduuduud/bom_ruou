"""
Module: cleaner.py
Mô tả: Module thực hiện tiền xử lý và làm sạch dữ liệu rượu vang.
Bao gồm:
- Chuẩn hóa tên cột sang tiếng Việt.
- Kiểm tra missing values và dữ liệu trùng lặp.
- Xử lý giá trị ngoại lai (Outliers) bằng kỹ thuật Winsorization / Capping (1% - 99%).
- Phân nhóm chất lượng thành 3 bậc: Kém (<= 4), Trung bình (4 < điểm < 6.5), Thượng hạng (>= 6.5).
- Xuất dữ liệu chuẩn sang định dạng Feather.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


try:
    from src.data_loader import doc_du_lieu_goc, luu_du_lieu_chuan
except ImportError:
    from data_loader import doc_du_lieu_goc, luu_du_lieu_chuan


BAN_DO_TEN_COT = {
    "fixed acidity": "axit_co_dinh",
    "volatile acidity": "axit_bay_hoi",
    "citric acid": "axit_citric",
    "residual sugar": "duong_du",
    "chlorides": "muoi_clorua",
    "free sulfur dioxide": "so2_tu_do",
    "total sulfur dioxide": "so2_tong",
    "density": "ti_trong",
    "pH": "do_ph",
    "sulphates": "sunfat",
    "alcohol": "do_con",
    "quality": "diem_chat_luong",
    "loai_ruou": "loai_ruou"
}

CAC_COT_HOA_LY = [
    "axit_co_dinh",
    "axit_bay_hoi",
    "axit_citric",
    "duong_du",
    "muoi_clorua",
    "so2_tu_do",
    "so2_tong",
    "ti_trong",
    "do_ph",
    "sunfat",
    "do_con"
]


def chuan_hoa_ten_cot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột của DataFrame sang tiếng Việt theo chuẩn BAN_DO_TEN_COT.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=BAN_DO_TEN_COT)


def kiem_tra_chat_luong_du_lieu(df: pd.DataFrame) -> dict:
    """
    Kiểm tra sơ bộ chất lượng dữ liệu: số dòng thiếu, số dòng trùng lặp.
    """
    tong_so_dong = len(df)
    so_dong_thieu = df.isnull().sum().to_dict()
    so_dong_trung_lap = int(df.duplicated().sum())

    return {
        "tong_so_dong": tong_so_dong,
        "so_dong_trung_lap": so_dong_trung_lap,
        "ti_le_trung_lap": round((so_dong_trung_lap / tong_so_dong) * 100, 2),
        "cot_thieu_du_lieu": {k: v for k, v in so_dong_thieu.items() if v > 0}
    }


def xu_ly_ngoai_lai_capping(
    df: pd.DataFrame,
    danh_sach_cot: List[str],
    phan_vi_duoi: float = 0.01,
    phan_vi_tren: float = 0.99
) -> Tuple[pd.DataFrame, dict]:
    """
    Xử lý các giá trị ngoại lai bằng phương pháp Winsorization / Capping.
    """
    df_clean = df.copy()
    thong_tin_nguong = {}

    for cot in danh_sach_cot:
        if cot in df_clean.columns and np.issubdtype(df_clean[cot].dtype, np.number):
            q_duoi = df_clean[cot].quantile(phan_vi_duoi)
            q_tren = df_clean[cot].quantile(phan_vi_tren)

            so_luong_duoi = (df_clean[cot] < q_duoi).sum()
            so_luong_tren = (df_clean[cot] > q_tren).sum()

            df_clean[cot] = df_clean[cot].clip(lower=q_duoi, upper=q_tren)

            thong_tin_nguong[cot] = {
                "nguong_duoi": round(float(q_duoi), 4),
                "nguong_tren": round(float(q_tren), 4),
                "so_mau_bi_cat_duoi": int(so_luong_duoi),
                "so_mau_bi_cat_tren": int(so_luong_tren)
            }

    return df_clean, thong_tin_nguong


def detect_outliers_iqr(data: pd.DataFrame, column: str) -> Tuple[int, float, float]:
    """
    Hàm phát hiện và đếm số lượng ngoại lai theo phương pháp IQR (Interquartile Range) của Nhóm 6.
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return len(outliers), float(lower_bound), float(upper_bound)


def chuan_hoa_thang_do_standard_scaler(
    df: pd.DataFrame,
    danh_sach_cot: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Chuẩn hóa thang đo các đặc trưng số về phân phối chuẩn (Mean ~ 0, Std ~ 1) bằng StandardScaler.
    """
    df_copy = df.copy()
    if danh_sach_cot is None:
        danh_sach_cot = [c for c in CAC_COT_HOA_LY if c in df_copy.columns]
        if not danh_sach_cot:
            danh_sach_cot = list(df_copy.select_dtypes(include=[np.number]).columns)

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df_copy[danh_sach_cot])
    df_scaled = pd.DataFrame(scaled_matrix, columns=danh_sach_cot, index=df_copy.index)
    return df_scaled, scaler


def tao_bien_phan_bac_chat_luong(
    df: pd.DataFrame,
    nguong_te: float = 4.0,
    nguong_tot: float = 6.5
) -> pd.DataFrame:
    """
    Tạo biến phân loại chất lượng rượu vang theo ngưỡng mới:
    - Kém (Rượu tệ): điểm <= 4.0
    - Trung bình: 4.0 < điểm < 6.5
    - Thượng hạng (Rượu tốt): điểm >= 6.5

    Bổ sung các cột (giữ nguyên tên cột để tương thích toàn bộ workflow):
    - bac_chat_luong: Nhãn phân loại ('Kém', 'Trung bình', 'Thượng hạng')
    - ma_bac_chat_luong: Mã số thứ tự (0: Kém, 1: Trung bình, 2: Thượng hạng)
    - chat_luong_cao: Nhãn nhị phân (1 nếu điểm >= 6.5, 0 nếu < 6.5)
    """
    df = df.copy()
    diem = df["diem_chat_luong"]

    thu_tu_bac = ["Kém", "Trung bình", "Thượng hạng"]
    anh_xa_ma = {"Kém": 0, "Trung bình": 1, "Thượng hạng": 2}

    nhan_phan_loai = np.select(
        condlist=[diem <= nguong_te, diem >= nguong_tot],
        choicelist=["Kém", "Thượng hạng"],
        default="Trung bình"
    )

    df["bac_chat_luong"] = pd.Categorical(nhan_phan_loai, categories=thu_tu_bac, ordered=True)
    df["ma_bac_chat_luong"] = df["bac_chat_luong"].map(anh_xa_ma).astype(int)
    df["chat_luong_cao"] = (diem >= nguong_tot).astype(int)

    return df


def lam_sach_toan_dien(
    df_goc: Optional[pd.DataFrame] = None,
    loai_bo_trung_lap: bool = False,
    luu_file: bool = True
) -> pd.DataFrame:
    """
    Quy trình làm sạch và chuẩn hóa toàn diện từ dữ liệu thô.
    """
    if df_goc is None:
        df_goc = doc_du_lieu_goc()

    print("[INFO] Bắt đầu quy trình làm sạch dữ liệu...")

    df = chuan_hoa_ten_cot(df_goc)
    print(f"[INFO] Đã chuẩn hóa {len(df.columns)} cột sang tiếng Việt.")

    chat_luong = kiem_tra_chat_luong_du_lieu(df)
    print(f"[INFO] Tổng số mẫu: {chat_luong['tong_so_dong']}")
    print(f"[INFO] Số dòng trùng lặp: {chat_luong['so_dong_trung_lap']} ({chat_luong['ti_le_trung_lap']}%)")
    if chat_luong["cot_thieu_du_lieu"]:
        print(f"[CẢNH BÁO] Phát hiện cột thiếu dữ liệu: {chat_luong['cot_thieu_du_lieu']}")
        for col in chat_luong["cot_thieu_du_lieu"].keys():
            df[col] = df.groupby("loai_ruou")[col].transform(lambda x: x.fillna(x.median()))
    else:
        print("[INFO] Dữ liệu hoàn hảo: Không có giá trị thiếu (No missing values).")

    if loai_bo_trung_lap:
        so_dong_truoc = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Đã loại bỏ {so_dong_truoc - len(df)} dòng trùng lặp. Còn lại {len(df)} dòng.")

    df_capped, _ = xu_ly_ngoai_lai_capping(
        df,
        danh_sach_cot=CAC_COT_HOA_LY,
        phan_vi_duoi=0.01,
        phan_vi_tren=0.99
    )
    print("[INFO] Đã hoàn tất Winsorization/Capping cho 11 chỉ số hóa lý (bảo tồn nguyên vẹn kích thước mẫu).")

    df_chuan = tao_bien_phan_bac_chat_luong(df_capped, nguong_te=4.0, nguong_tot=6.5)
    print("[INFO] Phân bố 3 bậc chất lượng (Kém <= 4.0 | Trung bình | Thượng hạng >= 6.5):")
    for bac, count in df_chuan["bac_chat_luong"].value_counts().items():
        ty_le = (count / len(df_chuan)) * 100
        print(f"       - Bậc {bac}: {count} mẫu ({ty_le:.2f}%)")

    if luu_file:
        luu_du_lieu_chuan(df_chuan)

    print("[THÀNH CÔNG] Hoàn tất làm sạch và chuẩn hóa dữ liệu!")
    return df_chuan


if __name__ == "__main__":
    df_chuan = lam_sach_toan_dien(luu_file=True)
    print("\n5 dòng đầu của dữ liệu chuẩn hóa:")
    print(df_chuan.head())