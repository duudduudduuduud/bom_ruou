"""
Module: environment_simulator.py
Mô tả: Công cụ mô phỏng độ lệch dữ liệu hóa lý rượu vang trong các điều kiện môi trường
khác nhau (nhiệt độ bảo quản, hở nút bần/oxy hóa, lão hóa hầm rượu, lệch chuẩn cảm biến)
và tái thiết lập điểm chất lượng tiềm năng (Potential Quality Points) sau biến động.

Đặc tính quan trọng:
- Tuyệt đối KHÔNG thay đổi hay ghi đè lên file dữ liệu gốc ban đầu (chatluong_ruougoc.csv).
- Toàn bộ dữ liệu mô phỏng và điểm tiềm năng được xuất ra các file chuyên biệt riêng:
    + data/duleu_lech_moi_truong.feather
    + data/duleu_lech_moi_truong.csv
    + data/bao_cao_lech_moi_truong.csv
    + data/ma_tran_chuyen_doi_chat_luong.csv
- Thiết lập điểm chất lượng tiềm năng (continuous & discrete) cùng ma trận chuyển dịch chất lượng.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from src.data_loader import doc_du_lieu_chuan
    from src.cleaner import CAC_COT_HOA_LY, tao_bien_phan_bac_chat_luong
except ImportError:
    from data_loader import doc_du_lieu_chuan
    from cleaner import CAC_COT_HOA_LY, tao_bien_phan_bac_chat_luong

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Định nghĩa các kịch bản môi trường chuẩn
KICH_BAN_MOI_TRUONG = {
    "heat_shock": "Sốc nhiệt & Nhiệt độ cao (>28°C - 35°C)",
    "oxidation_cork_failure": "Oxy hóa & Hở nút bần (Cork Failure)",
    "sensor_drift": "Lệch chuẩn thiết bị đo & Nhiễu cảm biến",
    "cellar_aging": "Lưu kho dài hạn & Lão hóa hầm rượu (Cellar Aging)",
}


class QualityPointsEvaluator:
    """
    Bộ ước lượng điểm chất lượng rượu vang tiềm năng dựa trên mô hình Gradient Boosting
    được huấn luyện trên dữ liệu chuẩn hóa nguyên bản, đảm bảo tính khách quan và khoa học.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            min_samples_leaf=20,
            random_state=self.random_state
        )
        self.da_huan_luyen = False
        self.cac_cot_dac_trung = CAC_COT_HOA_LY

    def huan_luyen(self, df_chuan: pd.DataFrame) -> "QualityPointsEvaluator":
        """Huấn luyện mô hình chuẩn trên dữ liệu sạch nguyên bản."""
        X = df_chuan[self.cac_cot_dac_trung]
        y = df_chuan["diem_chat_luong"]
        self.model.fit(X, y)
        self.da_huan_luyen = True
        return self

    def du_doan_diem_tiem_nang(self, df_bien_dong: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự đoán điểm chất lượng tiềm năng sau khi dữ liệu hóa lý bị lệch bởi môi trường.
        
        Trả về:
            - diem_lien_tuc: Điểm thực liên tục (continuous) bị chặn trong khoảng [3.0, 9.0]
            - diem_lam_tron: Điểm số nguyên làm tròn theo thang điểm UCI (3-9)
            - bac_tiem_nang: Phân bậc chất lượng ('Kém', 'Trung bình', 'Thượng hạng')
        """
        if not self.da_huan_luyen:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi .huan_luyen(df_chuan) trước.")

        X = df_bien_dong[self.cac_cot_dac_trung]
        diem_raw = self.model.predict(X)
        diem_lien_tuc = np.clip(np.round(diem_raw, 2), 3.0, 9.0)
        diem_lam_tron = np.clip(np.round(diem_lien_tuc).astype(int), 3, 9)

        # Phân bậc chuẩn theo ngưỡng dự án (<= 4: Kém, 4 < điểm < 6.5: Trung bình, >= 6.5: Thượng hạng)
        nhan_bac = np.select(
            condlist=[diem_lien_tuc <= 4.0, diem_lien_tuc >= 6.5],
            choicelist=["Kém", "Thượng hạng"],
            default="Trung bình"
        )
        return diem_lien_tuc, diem_lam_tron, nhan_bac


class WineEnvironmentSimulator:
    """
    Công cụ mô phỏng độ lệch của các chỉ số hóa lý rượu vang trong các môi trường khác nhau.
    """

    def __init__(self, df_chuan: Optional[pd.DataFrame] = None, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)
        
        if df_chuan is None:
            self.df_goc = doc_du_lieu_chuan().copy()
        else:
            self.df_goc = df_chuan.copy()

        # Khởi tạo và huấn luyện bộ đánh giá điểm tiềm năng
        self.evaluator = QualityPointsEvaluator(random_state=self.random_state)
        self.evaluator.huan_luyen(self.df_goc)

    def _gioi_han_vat_ly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bảo toàn các giới hạn vật lý và hóa học thực tế của rượu vang.
        """
        df_clip = df.copy()

        # Các chỉ số nồng độ không thể âm
        for col in ["axit_co_dinh", "axit_bay_hoi", "axit_citric", "duong_du",
                    "muoi_clorua", "so2_tu_do", "so2_tong", "sunfat"]:
            if col in df_clip.columns:
                df_clip[col] = df_clip[col].clip(lower=0.0)

        # Giới hạn pH rượu vang thông thường: 2.7 - 4.2
        if "do_ph" in df_clip.columns:
            df_clip["do_ph"] = df_clip["do_ph"].clip(lower=2.7, upper=4.2)

        # Giới hạn nồng độ cồn: 7.0% - 16.0%
        if "do_con" in df_clip.columns:
            df_clip["do_con"] = df_clip["do_con"].clip(lower=7.0, upper=16.0)

        # Giới hạn tỉ trọng rượu vang: 0.980 - 1.040
        if "ti_trong" in df_clip.columns:
            df_clip["ti_trong"] = df_clip["ti_trong"].clip(lower=0.980, upper=1.040)

        # SO2 tự do luôn nhỏ hơn hoặc bằng SO2 tổng số
        if "so2_tu_do" in df_clip.columns and "so2_tong" in df_clip.columns:
            df_clip["so2_tu_do"] = np.minimum(df_clip["so2_tu_do"], df_clip["so2_tong"])

        return df_clip

    def mo_phong_soc_nhiet(self, df: pd.DataFrame, intensity: float = 0.5) -> pd.DataFrame:
        """
        Kịch bản 1: Sốc nhiệt & Bảo quản nơi nhiệt độ cao (>28°C - 35°C).
        - Đẩy nhanh quá trình phân hủy/oxy hóa SO2 bảo quản.
        - Gia tăng axit bay hơi (axit axetic) do phản ứng oxy hóa rượu.
        - Giảm nhẹ nồng độ cồn do bay hơi qua vi lỗ nút bần.
        - Tăng nhẹ pH do suy thoái axit hữu cơ.
        """
        alpha = float(np.clip(intensity, 0.0, 1.0))
        sim = df.copy()

        # Tiêu hao SO2 bảo vệ
        sim["so2_tu_do"] = sim["so2_tu_do"] * (1.0 - 0.40 * alpha)
        sim["so2_tong"] = sim["so2_tong"] * (1.0 - 0.20 * alpha)

        # Tăng axit bay hơi (mùi chua giấm)
        sim["axit_bay_hoi"] = sim["axit_bay_hoi"] * (1.0 + 0.50 * alpha)

        # Bay hơi nhẹ độ cồn
        sim["do_con"] = sim["do_con"] * (1.0 - 0.03 * alpha)

        # pH tăng nhẹ
        sim["do_ph"] = sim["do_ph"] + 0.06 * alpha

        return self._gioi_han_vat_ly(sim)

    def mo_phong_oxy_hoa_nut_ban(self, df: pd.DataFrame, intensity: float = 0.5) -> pd.DataFrame:
        """
        Kịch bản 2: Oxy hóa & Hở nút bần (Cork Failure).
        - Không khí xâm nhập làm cạn kiệt SO2 tự do cực nhanh.
        - Axit bay hơi tăng vọt do vi khuẩn axetic hiếu khí phát triển.
        - Axit citric phân hủy tạo các hợp chất phụ.
        """
        alpha = float(np.clip(intensity, 0.0, 1.0))
        sim = df.copy()

        # Cạn kiệt SO2 tự do
        sim["so2_tu_do"] = sim["so2_tu_do"] * (1.0 - 0.70 * alpha)
        sim["so2_tong"] = sim["so2_tong"] * (1.0 - 0.35 * alpha)

        # Axit bay hơi tăng rất mạnh
        sim["axit_bay_hoi"] = sim["axit_bay_hoi"] * (1.0 + 0.90 * alpha)

        # Suy giảm axit citric tự nhiên
        sim["axit_citric"] = sim["axit_citric"] * (1.0 - 0.25 * alpha)

        # Độ cồn giảm nhẹ do oxy hóa acetaldehyde
        sim["do_con"] = sim["do_con"] * (1.0 - 0.04 * alpha)

        return self._gioi_han_vat_ly(sim)

    def mo_phong_lech_cam_bien(self, df: pd.DataFrame, intensity: float = 0.5) -> pd.DataFrame:
        """
        Kịch bản 3: Lệch chuẩn thiết bị đo phòng thí nghiệm & Nhiễu cảm biến.
        - Lệch điểm 0 của đầu dò pH (pH electrode drift).
        - Sai số nhiệt độ khi đo tỉ trọng bằng phù kế (hydrometer) hoặc khúc xạ kế đo cồn.
        - Nhiễu Gaussian ngẫu nhiên phản ánh độ bất định đo lường.
        """
        alpha = float(np.clip(intensity, 0.0, 1.0))
        sim = df.copy()
        n_samples = len(sim)

        # Lệch hệ thống pH + nhiễu ngẫu nhiên
        noise_ph = self.rng.normal(0, 0.03 * alpha, size=n_samples)
        sim["do_ph"] = sim["do_ph"] + (0.15 * alpha) + noise_ph

        # Sai số nhiệt độ đo tỉ trọng
        noise_density = self.rng.normal(0, 0.0008 * alpha, size=n_samples)
        sim["ti_trong"] = sim["ti_trong"] + (0.002 * alpha) + noise_density

        # Sai số đo độ cồn
        noise_alcohol = self.rng.normal(0, 0.15 * alpha, size=n_samples)
        sim["do_con"] = sim["do_con"] - (0.30 * alpha) + noise_alcohol

        # Nhiễu nhẹ các chỉ số khác
        for col in ["so2_tu_do", "axit_bay_hoi", "sunfat"]:
            std_col = sim[col].std()
            sim[col] = sim[col] + self.rng.normal(0, 0.03 * alpha * std_col, size=n_samples)

        return self._gioi_han_vat_ly(sim)

    def mo_phong_lao_hoa_ham_ruou(self, df: pd.DataFrame, intensity: float = 0.5) -> pd.DataFrame:
        """
        Kịch bản 4: Lưu kho dài hạn & Lão hóa hầm rượu (Cellar Maturation).
        - Quá trình kết tủa muối tartrate làm giảm nhẹ axit cố định.
        - SO2 bảo quản giảm dần đều theo thời gian tự nhiên.
        - Tỉ trọng cân bằng lại khi các phức chất lắng đọng.
        - Axit bay hơi tăng rất nhẹ theo tuổi rượu.
        """
        alpha = float(np.clip(intensity, 0.0, 1.0))
        sim = df.copy()

        # Kết tủa tartrate làm giảm nhẹ axit cố định
        sim["axit_co_dinh"] = sim["axit_co_dinh"] * (1.0 - 0.08 * alpha)

        # Giảm dần SO2 tự do và tổng số
        sim["so2_tu_do"] = sim["so2_tu_do"] * (1.0 - 0.30 * alpha)
        sim["so2_tong"] = sim["so2_tong"] * (1.0 - 0.15 * alpha)

        # Tăng nhẹ axit bay hoi
        sim["axit_bay_hoi"] = sim["axit_bay_hoi"] * (1.0 + 0.15 * alpha)

        # Cân bằng tỉ trọng
        sim["ti_trong"] = sim["ti_trong"] * (1.0 - 0.001 * alpha)

        return self._gioi_han_vat_ly(sim)

    def tao_du_lieu_lech_moi_truong(
        self,
        kich_ban: str = "heat_shock",
        intensity: float = 0.5,
        df_nguon: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Tạo tập dữ liệu bị lệch theo kịch bản môi trường xác định và tái thiết lập
        toàn bộ điểm chất lượng tiềm năng.

        Tham số:
            kich_ban: Tên kịch bản ('heat_shock', 'oxidation_cork_failure', 'sensor_drift', 'cellar_aging')
            intensity: Mức độ tác động từ môi trường (0.0 đến 1.0)
            df_nguon: DataFrame đầu vào (mặc định lấy bản sao dữ liệu sạch)

        Trả về:
            DataFrame hoàn toàn mới chứa dữ liệu bị lệch và các chỉ số điểm tiềm năng.
        """
        if df_nguon is None:
            df_base = self.df_goc.copy()
        else:
            df_base = df_nguon.copy()

        if kich_ban == "heat_shock":
            df_sim = self.mo_phong_soc_nhiet(df_base, intensity)
        elif kich_ban == "oxidation_cork_failure":
            df_sim = self.mo_phong_oxy_hoa_nut_ban(df_base, intensity)
        elif kich_ban == "sensor_drift":
            df_sim = self.mo_phong_lech_cam_bien(df_base, intensity)
        elif kich_ban == "cellar_aging":
            df_sim = self.mo_phong_lao_hoa_ham_ruou(df_base, intensity)
        else:
            raise ValueError(f"Kịch bản '{kich_ban}' không hợp lệ. Chọn: {list(KICH_BAN_MOI_TRUONG.keys())}")

        # Tái xác lập điểm chất lượng tiềm năng
        diem_tiem_nang, diem_lam_tron, bac_tiem_nang = self.evaluator.du_doan_diem_tiem_nang(df_sim)

        # Lưu trữ song song điểm gốc và điểm tiềm năng mới
        df_sim["kich_ban_moi_truong"] = kich_ban
        df_sim["mo_ta_kich_ban"] = KICH_BAN_MOI_TRUONG.get(kich_ban, kich_ban)
        df_sim["cuong_do_lech"] = round(float(intensity), 2)

        # Điểm gốc (bảo toàn 100% nhãn ban đầu)
        df_sim["diem_chat_luong_goc"] = df_base["diem_chat_luong"]
        df_sim["bac_chat_luong_goc"] = df_base["bac_chat_luong"]

        # Điểm tiềm năng tái xác lập
        df_sim["diem_chat_luong_tiem_nang"] = diem_tiem_nang
        df_sim["diem_chat_luong_lam_tron"] = diem_lam_tron
        df_sim["bac_chat_luong_tiem_nang"] = pd.Categorical(
            bac_tiem_nang,
            categories=["Kém", "Trung bình", "Thượng hạng"],
            ordered=True
        )

        # Chênh lệch điểm chất lượng: Delta = Điểm tiềm năng - Điểm gốc
        df_sim["delta_diem_chat_luong"] = np.round(df_sim["diem_chat_luong_tiem_nang"] - df_sim["diem_chat_luong_goc"], 2)

        # Trạng thái chuyển dịch thứ hạng
        mapping_rank = {"Kém": 0, "Trung bình": 1, "Thượng hạng": 2}
        rank_goc = df_sim["bac_chat_luong_goc"].astype(str).map(mapping_rank)
        rank_moi = df_sim["bac_chat_luong_tiem_nang"].astype(str).map(mapping_rank)

        conditions = [
            rank_moi < rank_goc,
            rank_moi == rank_goc,
            rank_moi > rank_goc
        ]
        choices = ["Giảm bậc (Tụt hạng)", "Bảo toàn bậc", "Tăng bậc (Cải thiện)"]
        df_sim["trang_thai_chuyen_doi"] = np.select(conditions, choices, default="Bảo toàn bậc")

        return df_sim

    def tinh_toan_chi_so_drift(
        self,
        df_sim: pd.DataFrame,
        df_goc: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Tính toán các chỉ số thống kê đối sánh độ lệch phân phối (Data Drift):
        - Mean gốc vs Mean sau lệch
        - Biến thiên tương đối (%)
        - Kiểm định Kolmogorov-Smirnov (KS-statistic & p-value)
        - Khoảng cách Wasserstein (Earth Mover's Distance)
        """
        if df_goc is None:
            df_goc = self.df_goc

        ket_qua = []
        for cot in CAC_COT_HOA_LY:
            if cot in df_goc.columns and cot in df_sim.columns:
                val_goc = df_goc[cot].dropna().values
                val_sim = df_sim[cot].dropna().values

                mean_goc = np.mean(val_goc)
                mean_sim = np.mean(val_sim)
                delta_pct = ((mean_sim - mean_goc) / mean_goc) * 100 if mean_goc != 0 else 0.0

                # Kiểm định KS 2 mẫu
                ks_res = stats.ks_2samp(val_goc, val_sim)
                # Khoảng cách Wasserstein
                wass_dist = stats.wasserstein_distance(val_goc, val_sim)

                ket_qua.append({
                    "Chỉ số hóa lý": cot,
                    "Mean Gốc": round(float(mean_goc), 3),
                    "Mean Sau Lệch": round(float(mean_sim), 3),
                    "Biến thiên (%)": round(float(delta_pct), 2),
                    "KS-Statistic": round(float(ks_res.statistic), 4),
                    "KS p-value": f"{ks_res.pvalue:.2e}",
                    "Wasserstein Dist": round(float(wass_dist), 4),
                    "Có Drift (p<0.05)": "Có" if ks_res.pvalue < 0.05 else "Không"
                })

        df_kq = pd.DataFrame(ket_qua).sort_values(by="KS-Statistic", ascending=False).reset_index(drop=True)
        return df_kq

    def tinh_ma_tran_chuyen_doi_chat_luong(self, df_sim: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Thiết lập bảng ma trận đối sánh chuyển dịch giữa các bậc chất lượng
        từ nguyên bản sang tiềm năng sau biến động môi trường.
        """
        tiers = ["Kém", "Trung bình", "Thượng hạng"]
        ma_tran = pd.crosstab(
            df_sim["bac_chat_luong_goc"],
            df_sim["bac_chat_luong_tiem_nang"],
            rownames=["Bậc Gốc"],
            colnames=["Bậc Tiềm Năng (Sau Lệch)"],
            dropna=False
        ).reindex(index=tiers, columns=tiers, fill_value=0)

        tong_so_mau = len(df_sim)
        so_mau_tut_hang = int((df_sim["trang_thai_chuyen_doi"] == "Giảm bậc (Tụt hạng)").sum())
        so_mau_giu_hang = int((df_sim["trang_thai_chuyen_doi"] == "Bảo toàn bậc").sum())
        so_mau_tang_hang = int((df_sim["trang_thai_chuyen_doi"] == "Tăng bậc (Cải thiện)").sum())

        thong_ke_chuyen_doi = {
            "tong_so_mau": tong_so_mau,
            "so_mau_tut_hang": so_mau_tut_hang,
            "ti_le_tut_hang_pct": round((so_mau_tut_hang / tong_so_mau) * 100, 2),
            "so_mau_giu_hang": so_mau_giu_hang,
            "ti_le_giu_hang_pct": round((so_mau_giu_hang / tong_so_mau) * 100, 2),
            "so_mau_tang_hang": so_mau_tang_hang,
            "ti_le_tang_hang_pct": round((so_mau_tang_hang / tong_so_mau) * 100, 2),
            "delta_diem_trung_binh": round(float(df_sim["delta_diem_chat_luong"].mean()), 3),
            "diem_goc_mean": round(float(df_sim["diem_chat_luong_goc"].mean()), 3),
            "diem_tiem_nang_mean": round(float(df_sim["diem_chat_luong_tiem_nang"].mean()), 3)
        }

        return ma_tran, thong_ke_chuyen_doi

    def xuat_bao_cao_va_du_lieu_an_toan(
        self,
        df_sim: pd.DataFrame,
        df_drift: pd.DataFrame,
        ma_tran: pd.DataFrame,
        thu_muc_data: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Xuất toàn bộ kết quả mô phỏng lệch môi trường ra các file độc lập riêng biệt,
        CAM KẾT: Hoàn toàn không ghi đè hay làm thay đổi file thô data/chatluong_ruougoc.csv.
        """
        if thu_muc_data is None:
            thu_muc_data = DATA_DIR

        thu_muc_data = Path(thu_muc_data)
        thu_muc_data.mkdir(parents=True, exist_ok=True)

        duong_dan_feather = thu_muc_data / "duleu_lech_moi_truong.feather"
        duong_dan_csv = thu_muc_data / "duleu_lech_moi_truong.csv"
        duong_dan_bao_cao = thu_muc_data / "bao_cao_lech_moi_truong.csv"
        duong_dan_ma_tran = thu_muc_data / "ma_tran_chuyen_doi_chat_luong.csv"

        # 1. Lưu DataFrame mô phỏng (Feather + CSV)
        # Chuyển kiểu categorical sang chuỗi trước khi lưu feather để tránh xung đột phiên bản
        df_export = df_sim.copy()
        for c in df_export.select_dtypes(include=["category"]).columns:
            df_export[c] = df_export[c].astype(str)

        df_export.to_feather(duong_dan_feather)
        df_export.to_csv(duong_dan_csv, index=False, encoding="utf-8-sig")

        # 2. Lưu báo cáo kiểm định Drift
        df_drift.to_csv(duong_dan_bao_cao, index=False, encoding="utf-8-sig")

        # 3. Lưu ma trận chuyển đổi chất lượng
        ma_tran.to_csv(duong_dan_ma_tran, encoding="utf-8-sig")

        return {
            "feather": duong_dan_feather,
            "csv": duong_dan_csv,
            "bao_cao_drift": duong_dan_bao_cao,
            "ma_tran_chuyen_doi": duong_dan_ma_tran
        }


def tao_bo_du_lieu_da_moi_truong(
    df_chuan: Optional[pd.DataFrame] = None,
    intensity: float = 0.5
) -> pd.DataFrame:
    """
    Hàm tiện ích: Tạo bộ dữ liệu tổng hợp kết hợp cả 4 kịch bản môi trường,
    giúp dễ dàng đối sánh tác động giữa các điều kiện bảo quản.
    """
    sim_tool = WineEnvironmentSimulator(df_chuan)
    danh_sach_kq = []

    for kb in KICH_BAN_MOI_TRUONG.keys():
        df_kb = sim_tool.tao_du_lieu_lech_moi_truong(kich_ban=kb, intensity=intensity)
        danh_sach_kq.append(df_kb)

    df_tong_hop = pd.concat(danh_sach_kq, ignore_index=True)
    return df_tong_hop
