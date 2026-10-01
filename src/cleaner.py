"""
Module: cleaner.py
Mô tả: Module thực hiện tiền xử lý và làm sạch dữ liệu rượu vang.
Bao gồm:
- Chuẩn hóa tên cột sang tiếng Việt.
- Kiểm tra missing values và dữ liệu trùng lặp.
- Xử lý giá trị ngoại lai (Outliers) bằng kỹ thuật Winsorization / Capping (1% - 99%).
- Phân nhóm chất lượng thành 3 bậc: Kém (3-4), Trung bình (5-6), Thượng hạng (7-9).
- Xuất dữ liệu chuẩn sang định dạng Feather.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Import từ module nội bộ
try:
    from src.data_loader import doc_du_lieu_goc, luu_du_lieu_chuan
except ImportError:
    from data_loader import doc_du_lieu_goc, luu_du_lieu_chuan

# Bản đồ ánh xạ tên cột từ tiếng Anh sang tiếng Việt chuẩn
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

# Danh sách 11 chỉ số hóa lý cần phân tích
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
    # Loại bỏ khoảng trắng thừa ở tên cột nếu có
    df.columns = [c.strip() for c in df.columns]
    # Đổi tên cột
    df = df.rename(columns=BAN_DO_TEN_COT)
    return df


def kiem_tra_chat_luong_du_lieu(df: pd.DataFrame) -> dict:
    """
    Kiểm tra sơ bộ chất lượng dữ liệu: số dòng thiếu, số dòng trùng lặp.
    """
    tong_so_dong = len(df)
    so_dong_thieu = df.isnull().sum().to_dict()
    so_dong_trung_lap = int(df.duplicated().sum())

    thong_tin = {
        "tong_so_dong": tong_so_dong,
        "so_dong_trung_lap": so_dong_trung_lap,
        "ti_le_trung_lap": round((so_dong_trung_lap / tong_so_dong) * 100, 2),
        "cot_thieu_du_lieu": {k: v for k, v in so_dong_thieu.items() if v > 0}
    }
    return thong_tin


def xu_ly_ngoai_lai_capping(
    df: pd.DataFrame,
    danh_sach_cot: List[str],
    phan_vi_duoi: float = 0.01,
    phan_vi_tren: float = 0.99
) -> Tuple[pd.DataFrame, dict]:
    """
    Xử lý các giá trị ngoại lai bằng phương pháp Winsorization / Capping.
    Các giá trị nhỏ hơn phân vị dưới sẽ được gán bằng giá trị phân vị dưới.
    Các giá trị lớn hơn phân vị trên sẽ được gán bằng giá trị phân vị trên.
    Kỹ thuật này giúp bảo tồn 100% số lượng quan sát của tập dữ liệu.

    Tham số:
        df: DataFrame đầu vào.
        danh_sach_cot: Danh sách các cột số cần xử lý.
        phan_vi_duoi: Ngưỡng phân vị dưới (mặc định 0.01 tức 1%).
        phan_vi_tren: Ngưỡng phân vị trên (mặc định 0.99 tức 99%).

    Trả về:
        Tuple gồm (DataFrame sau xử lý, dict thông tin ngưỡng của từng cột).
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


def tao_bien_phan_bac_chat_luong(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo biến phân loại 3 bậc chất lượng rượu vang:
    - Kém (Low/Poor): điểm 3 - 4
    - Trung bình (Medium/Normal): điểm 5 - 6
    - Thượng hạng (High/Premium): điểm 7 - 9

    Bổ sung thêm:
    - bac_chat_luong: Nhãn phân loại dạng chuỗi ('Kém', 'Trung bình', 'Thượng hạng')
    - ma_bac_chat_luong: Mã số thứ tự (0, 1, 2)
    - chat_luong_cao: Nhãn nhị phân (1 nếu Thượng hạng, 0 nếu Kém hoặc Trung bình)
    """
    df = df.copy()

    def gan_nhan(diem):
        if diem <= 4:
            return "Kém"
        elif diem <= 6:
            return "Trung bình"
        else:
            return "Thượng hạng"

    df["bac_chat_luong"] = df["diem_chat_luong"].apply(gan_nhan)

    # Đặt thứ tự cho Categorical để hỗ trợ sắp xếp và biểu đồ
    thu_tu_bac = ["Kém", "Trung bình", "Thượng hạng"]
    df["bac_chat_luong"] = pd.Categorical(df["bac_chat_luong"], categories=thu_tu_bac, ordered=True)

    # Mã số tương ứng
    anh_xa_ma = {"Kém": 0, "Trung bình": 1, "Thượng hạng": 2}
    df["ma_bac_chat_luong"] = df["bac_chat_luong"].map(anh_xa_ma).astype(int)

    # Nhãn nhị phân
    df["chat_luong_cao"] = (df["diem_chat_luong"] >= 7).astype(int)

    return df


def lam_sach_toan_dien(
    df_goc: Optional[pd.DataFrame] = None,
    loai_bo_trung_lap: bool = False,
    luu_file: bool = True
) -> pd.DataFrame:
    """
    Quy trình làm sạch và chuẩn hóa toàn diện từ dữ liệu thô.

    Tham số:
        df_goc: DataFrame thô đầu vào. Nếu None, sẽ tự đọc từ data/chatluong_ruougoc.csv.
        loai_bo_trung_lap: True nếu muốn xóa dòng trùng lặp hoàn toàn (mặc định False theo thiết kế).
        luu_file: True nếu muốn lưu ra file data/duleu_chuan.feather.

    Trả về:
        DataFrame đã làm sạch và phân bậc chuẩn mực.
    """
    if df_goc is None:
        df_goc = doc_du_lieu_goc()

    print("[INFO] Bắt đầu quy trình làm sạch dữ liệu...")

    # 1. Chuẩn hóa tên cột
    df = chuan_hoa_ten_cot(df_goc)
    print(f"[INFO] Đã chuẩn hóa {len(df.columns)} cột sang tiếng Việt.")

    # 2. Báo cáo chất lượng dữ liệu
    chat_luong = kiem_tra_chat_luong_du_lieu(df)
    print(f"[INFO] Tổng số mẫu: {chat_luong['tong_so_dong']}")
    print(f"[INFO] Số dòng trùng lặp: {chat_luong['so_dong_trung_lap']} ({chat_luong['ti_le_trung_lap']}%)")
    if chat_luong["cot_thieu_du_lieu"]:
        print(f"[CẢNH BÁO] Phát hiện cột thiếu dữ liệu: {chat_luong['cot_thieu_du_lieu']}")
        # Điền giá trị thiếu bằng trung vị (median) theo từng loại rượu
        for col in chat_luong["cot_thieu_du_lieu"].keys():
            df[col] = df.groupby("loai_ruou")[col].transform(lambda x: x.fillna(x.median()))
    else:
        print("[INFO] Dữ liệu hoàn hảo: Không có giá trị thiếu (No missing values).")

    # Xử lý trùng lặp nếu được yêu cầu
    if loai_bo_trung_lap:
        so_dong_truoc = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Đã loại bỏ {so_dong_truoc - len(df)} dòng trùng lặp. Còn lại {len(df)} dòng.")

    # 3. Xử lý ngoại lai bằng Winsorization / Capping (1% - 99%)
    df_capped, thong_tin_capping = xu_ly_ngoai_lai_capping(
        df,
        danh_sach_cot=CAC_COT_HOA_LY,
        phan_vi_duoi=0.01,
        phan_vi_tren=0.99
    )
    print("[INFO] Đã hoàn tất Winsorization/Capping cho 11 chỉ số hóa lý (bảo tồn nguyên vẹn kích thước mẫu).")

    # 4. Tạo biến phân bậc chất lượng
    df_chuan = tao_bien_phan_bac_chat_luong(df_capped)
    print("[INFO] Phân bố 3 bậc chất lượng:")
    for bac, count in df_chuan["bac_chat_luong"].value_counts().items():
        ty_le = (count / len(df_chuan)) * 100
        print(f"       - Bậc {bac}: {count} mẫu ({ty_le:.2f}%)")

    # 5. Lưu ra định dạng Feather
    if luu_file:
        luu_du_lieu_chuan(df_chuan)

    print("[THÀNH CÔNG] Hoàn tất làm sạch và chuẩn hóa dữ liệu!")
    return df_chuan


if __name__ == "__main__":
    df_chuan = lam_sach_toan_dien(luu_file=True)
    print("\n5 dòng đầu của dữ liệu chuẩn hóa:")
    print(df_chuan.head())
