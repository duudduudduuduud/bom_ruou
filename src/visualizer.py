"""
Module: visualizer.py
Mô tả: Module trực quan hóa dữ liệu rượu vang chuyên nghiệp bằng Seaborn và Matplotlib.
Hỗ trợ:
- Tự động thiết lập thẩm mỹ hiện đại, chuẩn font tiếng Việt.
- Tự động xuất biểu đồ độ phân giải cao (300 DPI) vào thư mục bieu_do/.
- Cung cấp hàm vẽ từng biểu đồ và hàm tổng hợp vẽ toàn bộ bộ sưu tập biểu đồ.
"""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

try:
    from src.data_loader import doc_du_lieu_chuan
    from src.cleaner import CAC_COT_HOA_LY
except ImportError:
    from data_loader import doc_du_lieu_chuan
    from cleaner import CAC_COT_HOA_LY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIEU_DO_DIR = PROJECT_ROOT / "bieu_do"


def thiet_lap_phong_cach_do_hoa():
    """
    Cấu hình phong cách đồ họa chung cho toàn bộ biểu đồ trong dự án:
    - Giao diện sạch sẽ, lưới mờ nhẹ (whitegrid).
    - Font chữ hỗ trợ tốt Unicode / tiếng Việt.
    - Bảng màu hài hòa, chuyên nghiệp.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.dpi"] = 120


def dam_bao_thu_muc_bieu_do(thu_muc: Optional[Path] = None) -> Path:
    """Đảm bảo thư mục lưu biểu đồ tồn tại."""
    if thu_muc is None:
        thu_muc = BIEU_DO_DIR
    thu_muc = Path(thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    return thu_muc


def ve_phan_phoi_chat_luong(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 01: Phân bố điểm chất lượng (3-9) và tỷ lệ 3 bậc chất lượng.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "01_phan_phoi_chat_luong.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette_ruou = {"Vang đỏ": "#8B0000", "Vang trắng": "#DAA520"}
    sns.countplot(
        data=df,
        x="diem_chat_luong",
        hue="loai_ruou",
        palette=palette_ruou,
        ax=axes[0],
        edgecolor="black",
        linewidth=0.8
    )
    axes[0].set_title("Phân Bố Điểm Chất Lượng (Quality Score 3 - 9)")
    axes[0].set_xlabel("Điểm chất lượng")
    axes[0].set_ylabel("Số lượng mẫu rượu")
    axes[0].legend(title="Loại rượu")

    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points"
            )

    dem_bac = df["bac_chat_luong"].value_counts()[["Kém", "Trung bình", "Thượng hạng"]]
    mau_sac_bac = ["#E74C3C", "#3498DB", "#2ECC71"]
    explode = (0.05, 0, 0.05)

    wedges, texts, autotexts = axes[1].pie(
        dem_bac,
        labels=dem_bac.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=mau_sac_bac,
        explode=explode,
        shadow=True,
        textprops={"fontsize": 11, "weight": "bold"}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(11)
    axes[1].set_title("Tỷ Lệ 3 Bậc Chất Lượng Rượu Vang")

    plt.suptitle("PHÂN TÍCH PHÂN BỐ CHẤT LƯỢNG RƯỢU VANG", fontsize=16, fontweight="bold", y=1.02)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_so_sanh_loai_ruou(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 02: So sánh đặc tính hóa lý nổi bật giữa Vang Đỏ và Vang Trắng.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "02_so_sanh_vang_do_trang.png"

    cac_chi_so = [
        ("axit_bay_hoi", "Axit bay hoi (g/dm3)", "Axit bay hoi cao hon o Vang do"),
        ("so2_tong", "SO2 tong so (mg/dm3)", "Vang trang co SO2 tong so cao hon"),
        ("duong_du", "Duong du (g/dm3)", "Vang trang co duong du cao hon"),
        ("do_ph", "Do pH", "Do pH cua vang do thuong cao hon"),
        ("sunfat", "Sunfat (g/dm3)", "Sunfat o vang do cao hon"),
        ("axit_co_dinh", "Axit co dinh (g/dm3)", "Axit co dinh o vang do cao hon")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    palette_ruou = {"Vang đỏ": "#8B0000", "Vang trắng": "#E1AD01"}

    for idx, (col, label, _) in enumerate(cac_chi_so):
        sns.boxplot(
            data=df,
            x="loai_ruou",
            y=col,
            hue="loai_ruou",
            palette=palette_ruou,
            legend=False,
            ax=axes[idx],
            width=0.45,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "7"}
        )
        axes[idx].set_title(label, fontsize=12)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(label)

    plt.suptitle("SO SÁNH ĐẶC TRƯNG HÓA LÝ GIỮA VANG ĐỎ VÀ VANG TRẮNG", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_ma_tran_tuong_quan(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 03: Ma trận hệ số tương quan Pearson giữa các chỉ số hóa lý và chất lượng.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "03_ma_tran_tuong_quan.png"

    cot_tinh_toan = CAC_COT_HOA_LY + ["diem_chat_luong", "ma_bac_chat_luong"]
    ma_tran_corr = df[cot_tinh_toan].corr()

    # Nhãn hiển thị tiếng Việt đẹp mắt
    nhan_cot = [
        "Axit cố định", "Axit bay hơi", "Axit citric", "Đường dư",
        "Muối clorua", "SO2 tự do", "SO2 tổng", "Tỉ trọng",
        "Độ pH", "Sunfat", "Độ cồn", "Điểm CL", "Bậc CL"
    ]

    plt.figure(figsize=(13, 10))
    mask = np.triu(np.ones_like(ma_tran_corr, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        ma_tran_corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan Pearson (r)"},
        xticklabels=nhan_cot,
        yticklabels=nhan_cot
    )
    plt.title("MA TRẬN TƯƠNG QUAN GIỮA CÁC ĐẶC TÍNH HÓA LÝ VÀ CHẤT LƯỢNG RƯỢU", fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_yeu_to_quyet_dinh_chat_luong(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 04: Đi sâu vào các yếu tố có tương quan mạnh nhất đến 3 bậc chất lượng.
    - Độ cồn (tương quan dương mạnh nhất)
    - Axit bay hơi (tương quan âm mạnh, gây chua/hỏng vị rượu)
    - Tỉ trọng (liên quan đến độ cồn và đường)
    - Sunfat (hợp chất hương vị và ổn định)
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "04_yeu_to_quyet_dinh_chat_luong.png"

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    palette_bac = {"Kém": "#E74C3C", "Trung bình": "#3498DB", "Thượng hạng": "#2ECC71"}

    sns.violinplot(
        data=df,
        x="bac_chat_luong",
        y="do_con",
        hue="bac_chat_luong",
        palette=palette_bac,
        legend=False,
        inner="quartile",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("1. Độ Cồn (% thể tích) theo Bậc Chất Lượng", fontsize=13)
    axes[0, 0].set_xlabel("Bậc chất lượng")
    axes[0, 0].set_ylabel("Độ cồn (%)")

    sns.boxplot(
        data=df,
        x="bac_chat_luong",
        y="axit_bay_hoi",
        hue="bac_chat_luong",
        palette=palette_bac,
        legend=False,
        width=0.45,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("2. Axit Bay Hơi (g/dm³) theo Bậc Chất Lượng", fontsize=13)
    axes[0, 1].set_xlabel("Bậc chất lượng")
    axes[0, 1].set_ylabel("Axit bay hơi (g/dm³)")

    sns.boxplot(
        data=df,
        x="bac_chat_luong",
        y="ti_trong",
        hue="bac_chat_luong",
        palette=palette_bac,
        legend=False,
        width=0.45,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("3. Tỉ Trọng (g/cm³) theo Bậc Chất Lượng", fontsize=13)
    axes[1, 0].set_xlabel("Bậc chất lượng")
    axes[1, 0].set_ylabel("Tỉ trọng (g/cm³)")

    sns.violinplot(
        data=df,
        x="bac_chat_luong",
        y="sunfat",
        hue="bac_chat_luong",
        palette=palette_bac,
        legend=False,
        inner="quartile",
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("4. Hàm Lượng Sunfat (g/dm³) theo Bậc Chất Lượng", fontsize=13)
    axes[1, 1].set_xlabel("Bậc chất lượng")
    axes[1, 1].set_ylabel("Sunfat (g/dm³)")

    plt.suptitle("CÁC CHỈ SỐ HÓA LÝ THEN CHỐT QUYẾT ĐỊNH BẬC CHẤT LƯỢNG RƯỢU VANG", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_phan_phoi_11_chi_so_hoa_ly(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 05: Phân phối (Histogram + Đường KDE) của toàn bộ 11 chỉ số hóa lý.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "05_phan_phoi_11_chi_so_hoa_ly.png"

    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    axes = axes.flatten()

    for idx, cot in enumerate(CAC_COT_HOA_LY):
        sns.histplot(
            data=df,
            x=cot,
            kde=True,
            ax=axes[idx],
            color="#2980B9",
            bins=30,
            edgecolor="white"
        )
        axes[idx].set_title(f"Phân phối: {cot}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(cot)
        axes[idx].set_ylabel("Tần suất")

    fig.delaxes(axes[11])

    plt.suptitle("PHÂN BỐ TẦN SUẤT 11 CHỈ SỐ HÓA LÝ RƯỢU VANG (SAU WINSORIZATION)", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file



def ve_bieu_do_phan_phoi_cot_hang(
    df: pd.DataFrame,
    n_graph_shown: int = 12,
    n_graph_per_row: int = 4,
    chi_loc_gia_tri_it: bool = False,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 06: Phân phối theo cột hàng (Distribution graphs: histogram / bar graph of column data).
    Mô phỏng cấu trúc hàm plotPerColumnDistribution trong starter notebook:
    - Hiển thị các đồ thị phân phối dạng lưới theo số hàng và số cột cấu hình.
    - Cột phân loại/ít giá trị hiển thị dạng Bar plot, cột số liên tục hiển thị dạng Histogram.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "06_phan_phoi_cot_hang.png"

    df_plot = df.copy()
    nunique = df_plot.nunique()
    if chi_loc_gia_tri_it:
        df_plot = df_plot[[col for col in df_plot if 1 < nunique[col] < 50]]
    else:
        df_plot = df_plot[[col for col in df_plot if nunique[col] > 1]]

    n_row, n_col = df_plot.shape
    column_names = list(df_plot.columns)
    so_cot_ve = min(n_col, n_graph_shown)
    n_graph_row = int((so_cot_ve + n_graph_per_row - 1) / n_graph_per_row)

    plt.figure(figsize=(5 * n_graph_per_row, 3.8 * n_graph_row), dpi=100, facecolor="w", edgecolor="k")

    for i in range(so_cot_ve):
        plt.subplot(n_graph_row, n_graph_per_row, i + 1)
        column_df = df_plot.iloc[:, i]
        if not np.issubdtype(type(column_df.iloc[0]), np.number) or nunique[column_names[i]] < 10:
            value_counts = column_df.value_counts()
            value_counts.plot.bar(color="#3498DB", edgecolor="black", linewidth=0.7)
        else:
            column_df.hist(bins=25, color="#2ECC71", edgecolor="black", linewidth=0.7)
        plt.ylabel("Số lượng mẫu (counts)", fontsize=9)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)
        plt.title(f"{column_names[i]} (Cột {i+1})", fontsize=11, fontweight="bold")

    plt.suptitle("BIỂU ĐỒ PHÂN PHỐI DỮ LIỆU THEO HÀNG VÀ CỘT (DISTRIBUTION GRAPHS)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout(pad=1.2, w_pad=1.0, h_pad=1.2)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_ma_tran_tuong_quan_matshow(
    df: pd.DataFrame,
    graph_width: int = 10,
    ten_file: str = "chatluong_ruougoc.csv",
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 07: Ma trận tương quan dạng lưới ma trận (Correlation matrix with matshow).
    Mô phỏng cấu trúc hàm plotCorrelationMatrix trong starter notebook:
    - Dùng plt.matshow và plt.colorbar.
    - Nhãn trục nằm phía dưới với góc xoay 90 độ.
    - Hiển thị tiêu đề Correlation Matrix for {ten_file}.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "07_ma_tran_tuong_quan_matshow.png"

    ten_bang = getattr(df, "dataframeName", ten_file)
    df_num = df.select_dtypes(include=[np.number]).dropna(axis="columns")
    df_num = df_num[[col for col in df_num if df_num[col].nunique() > 1]]

    if df_num.shape[1] < 2:
        print("[CẢNH BÁO] Không đủ số cột số để vẽ ma trận tương quan!")
        return duong_dan_file

    corr = df_num.corr()
    fig = plt.figure(figsize=(graph_width, graph_width), dpi=100, facecolor="w", edgecolor="k")
    fig.set_layout_engine("none")  # Tắt auto layout để tránh xung đột với matshow
    corr_mat = plt.matshow(corr, fignum=fig.number, cmap="coolwarm")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)
    plt.gca().xaxis.tick_bottom()
    cbar = plt.colorbar(corr_mat, shrink=0.8)
    cbar.ax.set_ylabel("Hệ số tương quan", fontsize=11, fontweight="bold")
    plt.title(f"Correlation Matrix for {ten_bang}", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_bieu_do_phan_tan_va_mat_do(
    df: pd.DataFrame,
    plot_size: int = 18,
    text_size: int = 9,
    so_cot_toi_da: int = 10,
    so_mau_lay: Optional[int] = 1000,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 08: Ma trận phân tán và mật độ (Scatter and Density plots with KDE).
    Mô phỏng cấu trúc hàm plotScatterMatrix trong starter notebook:
    - Sử dụng pd.plotting.scatter_matrix với đường chéo là ước lượng mật độ nhân (KDE).
    - Tính toán và chú thích hệ số tương quan 'Corr. coef = %.3f' ở nửa trên tam giác ma trận.
    - Giới hạn tối đa 10 cột để ma trận hiển thị rõ nét và tính toán nhanh chóng.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "08_phan_tan_va_mat_do.png"

    df_num = df.select_dtypes(include=[np.number]).dropna(axis="columns")
    df_num = df_num[[col for col in df_num if df_num[col].nunique() > 1]]

    column_names = list(df_num.columns)
    if len(column_names) > so_cot_toi_da:
        column_names = column_names[:so_cot_toi_da]
    df_plot = df_num[column_names]

    if so_mau_lay is not None and len(df_plot) > so_mau_lay:
        df_plot = df_plot.sample(n=so_mau_lay, random_state=42)

    ax = pd.plotting.scatter_matrix(
        df_plot,
        alpha=0.7,
        figsize=[plot_size, plot_size],
        diagonal="kde",
        color="#2C3E50",
        hist_kwds={"edgecolor": "black"}
    )
    corrs = df_plot.corr().values
    for i, j in zip(*np.triu_indices_from(ax, k=1)):
        ax[i, j].annotate(
            "Corr. coef = %.3f" % corrs[i, j],
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=text_size,
            weight="bold",
            color="#C0392B"
        )

    plt.suptitle("SCATTER AND DENSITY PLOT (MA TRẬN PHÂN TÁN VÀ ĐƯỜNG MẬT ĐỘ KDE)", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_tuong_quan_voi_quality_barplot(
    df: pd.DataFrame,
    cot_muc_tieu: str = "quality",
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ cột thể hiện mức độ tương quan của từng chỉ số với biến chất lượng (Quality).
    Các chỉ số tương quan dương tô màu xanh lá (g), tương quan âm tô màu đỏ (r).
    Kế thừa từ quy trình trực quan hóa của Nhóm 6.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "04_correlation_with_quality.png"

    df_num = df.select_dtypes(include=[np.number]).dropna(axis="columns")
    if cot_muc_tieu not in df_num.columns:
        if "diem_chat_luong" in df_num.columns:
            cot_muc_tieu = "diem_chat_luong"
        else:
            return duong_dan_file

    corr_series = df_num.corr()[cot_muc_tieu].drop(cot_muc_tieu).sort_values(ascending=False)
    plt.figure(figsize=(10, 5.5), dpi=100)
    colors = ["#2ECC71" if x > 0 else "#E74C3C" for x in corr_series]
    corr_series.plot(kind="bar", color=colors, edgecolor="black", linewidth=0.7)
    plt.title(f"Mức Độ Tương Quan Của Các Chỉ Số Với {cot_muc_tieu}", fontsize=14, fontweight="bold")
    plt.ylabel("Hệ số tương quan Pearson")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_cac_dac_trung_chinh_vs_quality_boxplots(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Lưới 2x2 Boxplots của 4 chỉ số có tương quan mạnh nhất đến chất lượng:
    - Alcohol vs Quality
    - Volatile Acidity vs Quality
    - Sulphates vs Quality
    - Citric Acid vs Quality
    Kế thừa từ thiết kế của Nhóm 6.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "05_key_chemical_features_vs_quality.png"

    col_q = "quality" if "quality" in df.columns else "diem_chat_luong"
    col_alc = "alcohol" if "alcohol" in df.columns else "do_con"
    col_va = "volatile acidity" if "volatile acidity" in df.columns else "axit_bay_hoi"
    col_sul = "sulphates" if "sulphates" in df.columns else "sunfat"
    col_cit = "citric acid" if "citric acid" in df.columns else "axit_citric"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    sns.boxplot(x=col_q, y=col_alc, data=df, hue=col_q, ax=axes[0, 0], palette="Blues", legend=False)
    axes[0, 0].set_title(f"{col_alc} vs {col_q}", fontsize=12, fontweight="bold")

    sns.boxplot(x=col_q, y=col_va, data=df, hue=col_q, ax=axes[0, 1], palette="Reds", legend=False)
    axes[0, 1].set_title(f"{col_va} vs {col_q}", fontsize=12, fontweight="bold")

    sns.boxplot(x=col_q, y=col_sul, data=df, hue=col_q, ax=axes[1, 0], palette="Greens", legend=False)
    axes[1, 0].set_title(f"{col_sul} vs {col_q}", fontsize=12, fontweight="bold")

    sns.boxplot(x=col_q, y=col_cit, data=df, hue=col_q, ax=axes[1, 1], palette="Oranges", legend=False)
    axes[1, 1].set_title(f"{col_cit} vs {col_q}", fontsize=12, fontweight="bold")

    plt.suptitle("CÁC HÓA CHẤT CHÍNH VS CHẤT LƯỢNG (KEY CHEMICALS VS QUALITY)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file


def ve_tat_ca_boxplots(
    df: pd.DataFrame,
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Lưới 4x3 Boxplots khảo sát ngoại lai cho toàn bộ các đặc trưng hóa lý.
    Kế thừa từ thiết kế của Nhóm 6.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "01_feature_boxplots.png"

    df_num = df.select_dtypes(include=[np.number]).copy()
    if "quality" in df_num.columns:
        features = [c for c in df_num.columns if c != "quality"]
    elif "diem_chat_luong" in df_num.columns:
        features = [c for c in df_num.columns if c not in ["diem_chat_luong", "ma_bac_chat_luong", "chat_luong_cao"]]
    else:
        features = list(df_num.columns)

    n_feats = min(len(features), 12)
    fig = plt.figure(figsize=(15, 12), dpi=100)
    for i in range(n_feats):
        plt.subplot(4, 3, i + 1)
        sns.boxplot(y=df[features[i]], color="#5DADE2")
        plt.title(f"Boxplot: {features[i]}", fontsize=11, fontweight="bold")
    plt.suptitle("BOXPLOTS CÁC ĐẶC TRƯNG HÓA LÝ (FEATURE BOXPLOTS)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[XUẤT BIỂU ĐỒ] Đã lưu: {duong_dan_file}")
    return duong_dan_file

plotPerColumnDistribution = ve_bieu_do_phan_phoi_cot_hang
plotCorrelationMatrix = ve_ma_tran_tuong_quan_matshow
plotScatterMatrix = ve_bieu_do_phan_tan_va_mat_do


def ve_so_sanh_lech_moi_truong(
    df_goc: pd.DataFrame,
    df_sim: pd.DataFrame,
    ten_kich_ban: str = "Sốc nhiệt & Nhiệt độ cao",
    thu_muc_luu: Optional[Path] = None
) -> Path:
    """
    Biểu đồ 09: Đối sánh đa chiều độ lệch hóa lý và tái thiết lập điểm tiềm năng sau biến động môi trường.
    Gồm 4 khung hình chuyên sâu:
    1. So sánh phân phối KDE các chỉ số hóa lý then chốt (Axit bay hơi, SO2 tự do, Độ cồn, pH).
    2. Phân bố điểm chất lượng tiềm năng (Gốc vs Sau lệch).
    3. Ma trận chuyển dịch bậc chất lượng (Heatmap Transition Matrix).
    4. Phân phối mức độ suy giảm điểm số (Delta Points) theo từng bậc ban đầu.
    """
    thiet_lap_phong_cach_do_hoa()
    thu_muc = dam_bao_thu_muc_bieu_do(thu_muc_luu)
    duong_dan_file = thu_muc / "09_mo_phong_lech_moi_truong.png"

    fig = plt.figure(figsize=(16, 12), dpi=120)

    # 1. Khung 1: Phân phối KDE 4 chỉ số nhạy cảm nhất
    ax1 = plt.subplot(2, 2, 1)
    chi_so_khao_sat = ["axit_bay_hoi", "so2_tu_do", "do_con", "do_ph"]
    nhan_chi_so = ["Axit bay hơi", "SO₂ tự do", "Độ cồn", "Độ pH"]

    colors = ["#2E86C1", "#E74C3C"]
    # Vẽ mini-grid 2x2 trong subplot 1 bằng cách lồng inset hoặc vẽ normalized plot
    # Đơn giản và rõ ràng nhất: vẽ KDE của axit bay hơi và so2 tự do
    sns.kdeplot(df_goc["axit_bay_hoi"], ax=ax1, color="#2E86C1", label="Axit bay hơi (Gốc)", linewidth=2)
    sns.kdeplot(df_sim["axit_bay_hoi"], ax=ax1, color="#E74C3C", linestyle="--", label="Axit bay hơi (Sau lệch)", linewidth=2)
    ax1.set_title("Biến động Axit bay hơi (Volatile Acidity)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Nồng độ (g/dm³)")
    ax1.set_ylabel("Mật độ KDE")
    ax1.legend(loc="upper right")

    # 2. Khung 2: So sánh phân phối điểm chất lượng gốc vs tiềm năng
    ax2 = plt.subplot(2, 2, 2)
    bins = np.arange(2.5, 9.6, 0.5)
    ax2.hist(df_goc["diem_chat_luong"], bins=bins, alpha=0.55, color="#2E86C1", label=f"Điểm gốc (Mean: {df_goc['diem_chat_luong'].mean():.2f})", edgecolor="black")
    ax2.hist(df_sim["diem_chat_luong_tiem_nang"], bins=bins, alpha=0.55, color="#E67E22", label=f"Điểm tiềm năng (Mean: {df_sim['diem_chat_luong_tiem_nang'].mean():.2f})", edgecolor="black")
    ax2.axvline(df_goc["diem_chat_luong"].mean(), color="#1B4F72", linestyle=":", linewidth=2)
    ax2.axvline(df_sim["diem_chat_luong_tiem_nang"].mean(), color="#A04000", linestyle=":", linewidth=2)
    ax2.set_title("Phân bố Điểm chất lượng: Gốc vs Tiềm năng Tái xác lập", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Điểm chất lượng (Scale 3 - 9)")
    ax2.set_ylabel("Số lượng mẫu")
    ax2.legend(loc="upper right")

    # 3. Khung 3: Heatmap ma trận chuyển đổi bậc chất lượng
    ax3 = plt.subplot(2, 2, 3)
    tiers = ["Kém", "Trung bình", "Thượng hạng"]
    ma_tran = pd.crosstab(
        df_sim["bac_chat_luong_goc"],
        df_sim["bac_chat_luong_tiem_nang"],
        dropna=False
    ).reindex(index=tiers, columns=tiers, fill_value=0)

    sns.heatmap(ma_tran, annot=True, fmt="d", cmap="YlOrRd", cbar=True, ax=ax3, linewidths=1, linecolor="gray")
    ax3.set_title("Ma trận Chuyển dịch Bậc Chất lượng (Transition Matrix)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Bậc Tiềm năng (Sau biến động)", fontweight="bold")
    ax3.set_ylabel("Bậc Nguyên bản (Gốc)", fontweight="bold")

    # 4. Khung 4: Phân phối Delta Điểm số theo bậc gốc
    ax4 = plt.subplot(2, 2, 4)
    sns.boxplot(
        data=df_sim,
        x="bac_chat_luong_goc",
        y="delta_diem_chat_luong",
        palette=["#E74C3C", "#F39C12", "#27AE60"],
        ax=ax4,
        order=tiers
    )
    ax4.axhline(0, color="gray", linestyle="--", linewidth=1.5)
    ax4.set_title("Biên độ Thay đổi Điểm (Delta Points) theo Bậc ban đầu", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Bậc chất lượng ban đầu")
    ax4.set_ylabel("Δ Điểm (Tiềm năng - Gốc)")

    fig.suptitle(f"MÔ PHỎNG LỆCH HÓA LÝ & TÁI THIẾT LẬP ĐIỂM TIỀM NĂNG\nKịch bản: {ten_kich_ban}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(duong_dan_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[XUẤT BIỂU ĐỒ] Đã lưu biểu đồ mô phỏng độ lệch tại: {duong_dan_file}")
    return duong_dan_file


def ve_tat_ca_bieu_do(
    df: Optional[pd.DataFrame] = None,
    thu_muc_luu: Optional[Path] = None
) -> List[Path]:
    """
    Hàm tổng hợp: Tự động vẽ và xuất toàn bộ 8 biểu đồ phân tích EDA vào bieu_do/.
    Bao gồm cả các biểu đồ chuyên đề rượu vang và 3 biểu đồ phong cách Kaggle Starter Notebook.

    Tham số:
        df: DataFrame dữ liệu chuẩn hóa. Nếu None, sẽ tự đọc từ data/duleu_chuan.feather.
        thu_muc_luu: Thư mục lưu file ảnh.

    Trả về:
        Danh sách đường dẫn các file ảnh đã lưu.
    """
    if df is None:
        df = doc_du_lieu_chuan()

    print("\n[INFO] Đang tiến hành tạo toàn bộ các biểu đồ phân tích EDA...")
    danh_sach_file = []
    danh_sach_file.append(ve_phan_phoi_chat_luong(df, thu_muc_luu))
    danh_sach_file.append(ve_so_sanh_loai_ruou(df, thu_muc_luu))
    danh_sach_file.append(ve_ma_tran_tuong_quan(df, thu_muc_luu))
    danh_sach_file.append(ve_yeu_to_quyet_dinh_chat_luong(df, thu_muc_luu))
    danh_sach_file.append(ve_phan_phoi_11_chi_so_hoa_ly(df, thu_muc_luu))
    danh_sach_file.append(ve_bieu_do_phan_phoi_cot_hang(df, n_graph_shown=12, n_graph_per_row=4, thu_muc_luu=thu_muc_luu))
    danh_sach_file.append(ve_ma_tran_tuong_quan_matshow(df, graph_width=10, ten_file="duleu_chuan", thu_muc_luu=thu_muc_luu))
    danh_sach_file.append(ve_bieu_do_phan_tan_va_mat_do(df, plot_size=18, text_size=9, so_cot_toi_da=10, so_mau_lay=1000, thu_muc_luu=thu_muc_luu))

    print(f"[HOÀN TẤT] Đã tạo thành công {len(danh_sach_file)} biểu đồ tại thư mục bieu_do/\n")
    return danh_sach_file


if __name__ == "__main__":
    ve_tat_ca_bieu_do()
