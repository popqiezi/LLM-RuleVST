# -*- coding: utf-8 -*-
"""
ARPA-CPA 海事碰撞风险基线（单文件版）

每个 CSV 对应一个固定船舶对，字段至少包括：

date,lon,lat,target_lon,target_lat,target_sog,target_cog,
distance_m,D_OT,v_OT,TCPA,DCPA,U_D,time_diff_sec,CRI

默认实验协议：
历史 40 步输入 -> 未来 20 步 CRI 预测

输出指标：
MAE、MSE、Precision、Recall、F1-score

说明：
1. 输入数据没有本船 SOG 和 COG，因此根据历史窗口末端的
   经纬度序列估计本船 SOG 和 COG。
2. 目标船优先使用 target_sog 和 target_cog。
3. ARPA-CPA 仅使用历史窗口内的信息，不使用未来真实位置。
4. 未来真实 CRI 仅用于最终指标计算。
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 只需要修改这里
# =========================================================

# 存放测试集 CSV 的文件夹
# 每个 CSV 文件对应一个固定船舶对
INPUT_DIR = Path(r"/home/la/CODE/风险预测/data/TSD")

# 输出文件夹
OUTPUT_DIR = Path(r"/home/la/CODE/风险预测/ARPA_CPA/arpaocalc/rusult")

# 填写数据集名称：AIS 或 UAV-TSD
DATASET_NAME = "AIS"

# CSV 是否位于多层子文件夹中
# 如果所有 CSV 都直接位于 INPUT_DIR 中，保持 False
RECURSIVE = False

# 与论文保持一致：40 步输入，20 步预测
HISTORY_STEPS = 40
FUTURE_STEPS = 20

# 滑动窗口步长
# 必须与你现有模型测试集的窗口生成方式一致
WINDOW_STRIDE = 20

# 使用历史窗口末端最近多少个位置间隔估计本船 SOG 和 COG
MOTION_LOOKBACK_STEPS = 5

# CRI 参数
D1_M = 100.0
D2_M = 900.0

W_TIME = 0.1
W_DISTANCE = 0.9

# 高风险阈值
HIGH_RISK_THRESHOLD = 0.87

# 论文表格输出保留的小数位数
ROUND_DIGITS = 5


# =========================================================
# 2. 常量
# =========================================================

EARTH_RADIUS_M = 6378137.0

# 1 knot = 1852 m/h
KNOT_TO_MPS = 1852.0 / 3600.0
MPS_TO_KNOT = 1.0 / KNOT_TO_MPS

REQUIRED_COLUMNS = {
    "date",
    "lon",
    "lat",
    "target_lon",
    "target_lat",
    "target_sog",
    "target_cog",
    "CRI",
}


# =========================================================
# 3. 基础函数
# =========================================================

def finite_number(value):
    """判断一个值是否为有效有限数值。"""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def haversine_m(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的大圆距离。

    返回
    ----
    distance : float
        距离，单位 m。
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2.0) ** 2
    )

    a = max(0.0, min(1.0, a))

    return (
        2.0
        * EARTH_RADIUS_M
        * math.asin(math.sqrt(a))
    )


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    计算点1指向点2的初始方位角。

    正北为 0°，顺时针增加。
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_lambda = math.radians(lon2 - lon1)

    x = (
        math.sin(delta_lambda)
        * math.cos(phi2)
    )

    y = (
        math.cos(phi1)
        * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(delta_lambda)
    )

    if abs(x) < 1e-15 and abs(y) < 1e-15:
        return 0.0

    return (
        math.degrees(math.atan2(x, y))
        % 360.0
    )


def destination_position(
    lat,
    lon,
    distance_m,
    cog_deg,
):
    """
    根据起始经纬度、航行距离和航向计算终点。

    参数
    ----
    lat, lon
        起点纬度和经度。
    distance_m
        航行距离，单位 m。
    cog_deg
        航向角，正北为 0°，顺时针增加。

    返回
    ----
    lat2, lon2
        预测终点纬度和经度。
    """
    angular_distance = (
        distance_m / EARTH_RADIUS_M
    )

    bearing = math.radians(cog_deg)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    sin_lat2 = (
        math.sin(lat1)
        * math.cos(angular_distance)
        + math.cos(lat1)
        * math.sin(angular_distance)
        * math.cos(bearing)
    )

    sin_lat2 = max(
        -1.0,
        min(1.0, sin_lat2),
    )

    lat2 = math.asin(sin_lat2)

    lon2 = lon1 + math.atan2(
        math.sin(bearing)
        * math.sin(angular_distance)
        * math.cos(lat1),

        math.cos(angular_distance)
        - math.sin(lat1)
        * math.sin(lat2),
    )

    lon2 = (
        lon2 + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi

    return (
        math.degrees(lat2),
        math.degrees(lon2),
    )


def local_relative_position_m(
    own_lat,
    own_lon,
    target_lat,
    target_lon,
):
    """
    将目标船相对本船的位置转换为局部东、北方向距离。

    返回
    ----
    east_m, north_m
        目标船相对本船的东向和北向距离，单位 m。
    """
    mean_lat = math.radians(
        (own_lat + target_lat) / 2.0
    )

    east_m = (
        math.radians(target_lon - own_lon)
        * EARTH_RADIUS_M
        * math.cos(mean_lat)
    )

    north_m = (
        math.radians(target_lat - own_lat)
        * EARTH_RADIUS_M
    )

    return east_m, north_m


def velocity_components_mps(
    sog_kn,
    cog_deg,
):
    """
    将 SOG 和 COG 转换为东向、北向速度。

    返回单位为 m/s。
    """
    speed_mps = (
        max(0.0, float(sog_kn))
        * KNOT_TO_MPS
    )

    angle = math.radians(
        float(cog_deg) % 360.0
    )

    east_mps = (
        speed_mps
        * math.sin(angle)
    )

    north_mps = (
        speed_mps
        * math.cos(angle)
    )

    return east_mps, north_mps


# =========================================================
# 4. 根据历史位置估计 SOG 和 COG
# =========================================================

def infer_motion_from_history(
    frame,
    lat_col,
    lon_col,
    history_end_idx,
    lookback_steps,
):
    """
    仅使用历史位置估计船舶 SOG 和 COG。

    为避免 COG 在 0° 和 360° 附近出现直接平均错误，
    本函数先计算每一段的东向、北向速度，再分别取中位数。
    """
    start_idx = max(
        0,
        history_end_idx - lookback_steps,
    )

    east_velocity_list = []
    north_velocity_list = []

    for index in range(
        start_idx + 1,
        history_end_idx + 1,
    ):
        previous_row = frame.iloc[index - 1]
        current_row = frame.iloc[index]

        previous_time = previous_row["date"]
        current_time = current_row["date"]

        delta_time_s = (
            current_time - previous_time
        ).total_seconds()

        if (
            not np.isfinite(delta_time_s)
            or delta_time_s <= 0
        ):
            continue

        lat0 = float(
            previous_row[lat_col]
        )
        lon0 = float(
            previous_row[lon_col]
        )

        lat1 = float(
            current_row[lat_col]
        )
        lon1 = float(
            current_row[lon_col]
        )

        distance_m = haversine_m(
            lat0,
            lon0,
            lat1,
            lon1,
        )

        if distance_m <= 1e-9:
            east_velocity_list.append(0.0)
            north_velocity_list.append(0.0)
            continue

        course_deg = bearing_deg(
            lat0,
            lon0,
            lat1,
            lon1,
        )

        speed_mps = (
            distance_m / delta_time_s
        )

        course_rad = math.radians(
            course_deg
        )

        east_velocity_list.append(
            speed_mps
            * math.sin(course_rad)
        )

        north_velocity_list.append(
            speed_mps
            * math.cos(course_rad)
        )

    if not east_velocity_list:
        return 0.0, 0.0

    east_mps = float(
        np.median(east_velocity_list)
    )

    north_mps = float(
        np.median(north_velocity_list)
    )

    speed_mps = math.hypot(
        east_mps,
        north_mps,
    )

    if speed_mps <= 1e-9:
        return 0.0, 0.0

    sog_kn = (
        speed_mps
        * MPS_TO_KNOT
    )

    cog_deg = (
        math.degrees(
            math.atan2(
                east_mps,
                north_mps,
            )
        )
        % 360.0
    )

    return sog_kn, cog_deg


# =========================================================
# 5. ARPA-CPA 和 TCPA
# =========================================================

def arpa_cpa_calculation(
    own_lat,
    own_lon,
    own_sog,
    own_cog,
    target_lat,
    target_lon,
    target_sog,
    target_cog,
):
    """
    使用经典 ARPA 相对运动向量计算 DCPA 和 TCPA。

    定义：
    r = 目标船相对本船的位置向量
    v = 目标船相对本船的速度向量

    TCPA = -(r · v) / |v|²
    DCPA = |r + v × TCPA|

    返回
    ----
    distance_m
        当前船间距离，单位 m。

    dcpa_m
        最近会遇距离，单位 m。

    tcpa_s
        到最近会遇点的时间，单位 s。

    relative_speed_mps
        相对速度，单位 m/s。

    theta_rad
        相对位置向量与相对速度向量之间的夹角。
    """
    relative_x, relative_y = (
        local_relative_position_m(
            own_lat,
            own_lon,
            target_lat,
            target_lon,
        )
    )

    own_velocity_x, own_velocity_y = (
        velocity_components_mps(
            own_sog,
            own_cog,
        )
    )

    target_velocity_x, target_velocity_y = (
        velocity_components_mps(
            target_sog,
            target_cog,
        )
    )

    relative_velocity_x = (
        target_velocity_x
        - own_velocity_x
    )

    relative_velocity_y = (
        target_velocity_y
        - own_velocity_y
    )

    distance_m = math.hypot(
        relative_x,
        relative_y,
    )

    relative_speed_mps = math.hypot(
        relative_velocity_x,
        relative_velocity_y,
    )

    relative_speed_squared = (
        relative_velocity_x ** 2
        + relative_velocity_y ** 2
    )

    # 计算相对位置与相对速度夹角
    if (
        distance_m <= 1e-12
        or relative_speed_mps <= 1e-12
    ):
        theta_rad = 0.0
    else:
        cosine_theta = (
            relative_x
            * relative_velocity_x
            + relative_y
            * relative_velocity_y
        ) / (
            distance_m
            * relative_speed_mps
        )

        cosine_theta = max(
            -1.0,
            min(1.0, cosine_theta),
        )

        theta_rad = math.acos(
            cosine_theta
        )

    # 两船相对速度接近零
    if relative_speed_squared <= 1e-12:
        return {
            "distance_m": distance_m,
            "dcpa_m": distance_m,
            "tcpa_s": 0.0,
            "relative_speed_mps": relative_speed_mps,
            "theta_rad": theta_rad,
        }

    raw_tcpa_s = -(
        relative_x
        * relative_velocity_x
        + relative_y
        * relative_velocity_y
    ) / relative_speed_squared

    # TCPA 小于 0 表示最近会遇点已经过去
    if raw_tcpa_s <= 0.0:
        tcpa_s = 0.0
        dcpa_m = distance_m

    else:
        tcpa_s = raw_tcpa_s

        cpa_x = (
            relative_x
            + relative_velocity_x
            * tcpa_s
        )

        cpa_y = (
            relative_y
            + relative_velocity_y
            * tcpa_s
        )

        dcpa_m = math.hypot(
            cpa_x,
            cpa_y,
        )

    return {
        "distance_m": float(distance_m),
        "dcpa_m": float(dcpa_m),
        "tcpa_s": float(tcpa_s),
        "relative_speed_mps": float(
            relative_speed_mps
        ),
        "theta_rad": float(theta_rad),
    }


# =========================================================
# 6. CRI 计算
# =========================================================

def calculate_u_distance(distance_m):
    """
    计算空间风险因子 U(D)。
    """
    if distance_m >= D2_M:
        return 0.0

    if distance_m <= D1_M:
        return 1.0

    return (
        D2_M - distance_m
    ) / (
        D2_M - D1_M
    )


def calculate_u_time(
    dcpa_m,
    tcpa_s,
    relative_speed_mps,
    theta_rad,
):
    """
    按论文当前公式计算时间风险因子 U(t)。

    DCPA >= D2:
        U(t) = 0

    DCPA <= D1:
        U(t) = 1

    D1 < DCPA < D2:
        U(t) = exp(
            -abs(TCPA)
            + v_OT * tan(theta)
        )

    注意：
    此函数最终应与你原来生成 CRI 标签时使用的代码保持完全一致。
    """
    dcpa_m = abs(float(dcpa_m))

    if dcpa_m >= D2_M:
        return 0.0

    if dcpa_m <= D1_M:
        return 1.0

    cosine_theta = math.cos(
        theta_rad
    )

    # 防止 theta 接近 90° 时 tan(theta) 数值溢出
    if abs(cosine_theta) < 1e-8:
        tangent_theta = math.copysign(
            1e8,
            math.sin(theta_rad),
        )
    else:
        tangent_theta = math.tan(
            theta_rad
        )

    exponent = (
        -abs(float(tcpa_s))
        + float(relative_speed_mps)
        * tangent_theta
    )

    # 防止指数溢出
    exponent = max(
        -700.0,
        min(50.0, exponent),
    )

    u_time = math.exp(
        exponent
    )

    return float(
        np.clip(
            u_time,
            0.0,
            1.0,
        )
    )


def calculate_cri(
    distance_m,
    dcpa_m,
    tcpa_s,
    relative_speed_mps,
    theta_rad,
):
    """
    计算最终 CRI。

    返回
    ----
    cri
    u_time
    u_distance
    """
    u_time = calculate_u_time(
        dcpa_m,
        tcpa_s,
        relative_speed_mps,
        theta_rad,
    )

    u_distance = calculate_u_distance(
        distance_m
    )

    cri = (
        W_TIME * u_time
        + W_DISTANCE * u_distance
    )

    cri = float(
        np.clip(
            cri,
            0.0,
            1.0,
        )
    )

    return (
        cri,
        u_time,
        u_distance,
    )


# =========================================================
# 7. 评价指标
# =========================================================

def calculate_metrics(
    true_cri,
    predicted_cri,
):
    """
    计算论文 Table 11 所需指标。
    """
    y_true = np.asarray(
        true_cri,
        dtype=float,
    )

    y_pred = np.asarray(
        predicted_cri,
        dtype=float,
    )

    valid_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[
        valid_mask
    ]

    y_pred = y_pred[
        valid_mask
    ]

    if len(y_true) == 0:
        raise ValueError(
            "没有可用于计算指标的有效预测点。"
        )

    error = y_pred - y_true

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    true_high_risk = (
        y_true
        >= HIGH_RISK_THRESHOLD
    )

    predicted_high_risk = (
        y_pred
        >= HIGH_RISK_THRESHOLD
    )

    true_positive = int(
        np.sum(
            true_high_risk
            & predicted_high_risk
        )
    )

    false_positive = int(
        np.sum(
            ~true_high_risk
            & predicted_high_risk
        )
    )

    false_negative = int(
        np.sum(
            true_high_risk
            & ~predicted_high_risk
        )
    )

    true_negative = int(
        np.sum(
            ~true_high_risk
            & ~predicted_high_risk
        )
    )

    if (
        true_positive
        + false_positive
        > 0
    ):
        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
        )
    else:
        precision = 0.0

    if (
        true_positive
        + false_negative
        > 0
    ):
        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
        )
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1_score = 0.0

    return {
        "MAE": mae,
        "MSE": mse,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-score": float(f1_score),

        "TP": true_positive,
        "FP": false_positive,
        "FN": false_negative,
        "TN": true_negative,

        "FuturePoints": int(
            len(y_true)
        ),
    }


# =========================================================
# 8. CSV 读取
# =========================================================

def load_pair_csv(csv_path):
    """
    读取单个船舶对 CSV。
    """
    frame = pd.read_csv(
        csv_path
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"缺少字段："
            f"{sorted(missing_columns)}"
        )

    frame = frame.copy()

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )

    numeric_columns = [
        "lon",
        "lat",
        "target_lon",
        "target_lat",
        "target_sog",
        "target_cog",
        "CRI",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    # 确保按时间排序
    frame = frame.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # 目标船速度和航向允许为空，因为后续存在位置估计回退
    frame = frame.dropna(
        subset=[
            "date",
            "lon",
            "lat",
            "target_lon",
            "target_lat",
            "CRI",
        ]
    ).reset_index(
        drop=True
    )

    return frame


def get_csv_files():
    """
    获取输入文件夹中的所有 CSV。
    """
    if RECURSIVE:
        pattern = "**/*.csv"
    else:
        pattern = "*.csv"

    return sorted(
        path
        for path in INPUT_DIR.glob(pattern)
        if path.is_file()
    )


# =========================================================
# 9. 单个船舶对：40 步输入、20 步预测
# =========================================================

def evaluate_one_pair(csv_path):
    """
    对单个船舶对 CSV 执行 ARPA-CPA 滑动预测。
    """
    frame = load_pair_csv(
        csv_path
    )

    required_length = (
        HISTORY_STEPS
        + FUTURE_STEPS
    )

    if len(frame) < required_length:
        return (
            pd.DataFrame(),
            None,
        )

    prediction_rows = []
    valid_window_count = 0

    last_window_start = (
        len(frame)
        - required_length
    )

    for window_start in range(
        0,
        last_window_start + 1,
        WINDOW_STRIDE,
    ):
        history_end_index = (
            window_start
            + HISTORY_STEPS
            - 1
        )

        future_start_index = (
            history_end_index + 1
        )

        future_end_index = (
            future_start_index
            + FUTURE_STEPS
        )

        history_end_row = frame.iloc[
            history_end_index
        ]

        history_end_time = (
            history_end_row["date"]
        )

        # -------------------------------------------------
        # 本船 SOG 和 COG
        # CSV 没有本船速度和航向，因此仅根据历史位置估计
        # -------------------------------------------------
        own_sog_kn, own_cog_deg = (
            infer_motion_from_history(
                frame=frame,
                lat_col="lat",
                lon_col="lon",
                history_end_idx=history_end_index,
                lookback_steps=MOTION_LOOKBACK_STEPS,
            )
        )

        # -------------------------------------------------
        # 目标船 SOG 和 COG
        # 优先使用 target_sog 和 target_cog
        # -------------------------------------------------
        if (
            finite_number(
                history_end_row[
                    "target_sog"
                ]
            )
            and finite_number(
                history_end_row[
                    "target_cog"
                ]
            )
        ):
            target_sog_kn = max(
                0.0,
                float(
                    history_end_row[
                        "target_sog"
                    ]
                ),
            )

            target_cog_deg = (
                float(
                    history_end_row[
                        "target_cog"
                    ]
                )
                % 360.0
            )

        else:
            target_sog_kn, target_cog_deg = (
                infer_motion_from_history(
                    frame=frame,
                    lat_col="target_lat",
                    lon_col="target_lon",
                    history_end_idx=history_end_index,
                    lookback_steps=MOTION_LOOKBACK_STEPS,
                )
            )

        own_initial_lat = float(
            history_end_row["lat"]
        )

        own_initial_lon = float(
            history_end_row["lon"]
        )

        target_initial_lat = float(
            history_end_row[
                "target_lat"
            ]
        )

        target_initial_lon = float(
            history_end_row[
                "target_lon"
            ]
        )

        future_frame = frame.iloc[
            future_start_index:
            future_end_index
        ]

        if len(future_frame) != FUTURE_STEPS:
            continue

        window_id = (
            f"{csv_path.stem}"
            f"_w{window_start:06d}"
        )

        generated_steps = 0

        for future_step, (_, true_row) in enumerate(
            future_frame.iterrows(),
            start=1,
        ):
            future_time = (
                true_row["date"]
            )

            elapsed_seconds = (
                future_time
                - history_end_time
            ).total_seconds()

            if (
                not np.isfinite(
                    elapsed_seconds
                )
                or elapsed_seconds <= 0
            ):
                continue

            # =============================================
            # 恒航向、恒航速外推本船未来位置
            # =============================================
            own_travel_distance_m = (
                own_sog_kn
                * KNOT_TO_MPS
                * elapsed_seconds
            )

            (
                predicted_own_lat,
                predicted_own_lon,
            ) = destination_position(
                lat=own_initial_lat,
                lon=own_initial_lon,
                distance_m=own_travel_distance_m,
                cog_deg=own_cog_deg,
            )

            # =============================================
            # 恒航向、恒航速外推目标船未来位置
            # =============================================
            target_travel_distance_m = (
                target_sog_kn
                * KNOT_TO_MPS
                * elapsed_seconds
            )

            (
                predicted_target_lat,
                predicted_target_lon,
            ) = destination_position(
                lat=target_initial_lat,
                lon=target_initial_lon,
                distance_m=target_travel_distance_m,
                cog_deg=target_cog_deg,
            )

            # =============================================
            # ARPA CPA/TCPA
            # =============================================
            arpa_result = (
                arpa_cpa_calculation(
                    own_lat=predicted_own_lat,
                    own_lon=predicted_own_lon,
                    own_sog=own_sog_kn,
                    own_cog=own_cog_deg,

                    target_lat=predicted_target_lat,
                    target_lon=predicted_target_lon,
                    target_sog=target_sog_kn,
                    target_cog=target_cog_deg,
                )
            )

            # =============================================
            # CRI
            # =============================================
            (
                predicted_cri,
                predicted_u_time,
                predicted_u_distance,
            ) = calculate_cri(
                distance_m=arpa_result[
                    "distance_m"
                ],
                dcpa_m=arpa_result[
                    "dcpa_m"
                ],
                tcpa_s=arpa_result[
                    "tcpa_s"
                ],
                relative_speed_mps=arpa_result[
                    "relative_speed_mps"
                ],
                theta_rad=arpa_result[
                    "theta_rad"
                ],
            )

            true_cri = float(
                true_row["CRI"]
            )

            prediction_rows.append(
                {
                    "source_file":
                        csv_path.name,

                    "window_id":
                        window_id,

                    "window_start_index":
                        window_start,

                    "history_end_index":
                        history_end_index,

                    "history_end_time":
                        history_end_time,

                    "future_step":
                        future_step,

                    "future_time":
                        future_time,

                    "elapsed_seconds":
                        elapsed_seconds,

                    "own_sog_kn":
                        own_sog_kn,

                    "own_cog_deg":
                        own_cog_deg,

                    "target_sog_kn":
                        target_sog_kn,

                    "target_cog_deg":
                        target_cog_deg,

                    "pred_own_lon":
                        predicted_own_lon,

                    "pred_own_lat":
                        predicted_own_lat,

                    "pred_target_lon":
                        predicted_target_lon,

                    "pred_target_lat":
                        predicted_target_lat,

                    "pred_distance_m":
                        arpa_result[
                            "distance_m"
                        ],

                    "pred_DCPA_m":
                        arpa_result[
                            "dcpa_m"
                        ],

                    "pred_TCPA_s":
                        arpa_result[
                            "tcpa_s"
                        ],

                    "pred_v_OT_mps":
                        arpa_result[
                            "relative_speed_mps"
                        ],

                    "pred_theta_deg":
                        math.degrees(
                            arpa_result[
                                "theta_rad"
                            ]
                        ),

                    "pred_U_t":
                        predicted_u_time,

                    "pred_U_D":
                        predicted_u_distance,

                    "pred_CRI":
                        predicted_cri,

                    "true_CRI":
                        true_cri,

                    "absolute_error":
                        abs(
                            predicted_cri
                            - true_cri
                        ),

                    "squared_error":
                        (
                            predicted_cri
                            - true_cri
                        ) ** 2,

                    "true_high_risk":
                        int(
                            true_cri
                            >= HIGH_RISK_THRESHOLD
                        ),

                    "pred_high_risk":
                        int(
                            predicted_cri
                            >= HIGH_RISK_THRESHOLD
                        ),
                }
            )

            generated_steps += 1

        if generated_steps == FUTURE_STEPS:
            valid_window_count += 1

    predictions = pd.DataFrame(
        prediction_rows
    )

    if predictions.empty:
        return (
            predictions,
            None,
        )

    pair_metrics = calculate_metrics(
        true_cri=predictions[
            "true_CRI"
        ].to_numpy(),

        predicted_cri=predictions[
            "pred_CRI"
        ].to_numpy(),
    )

    pair_metrics.update(
        {
            "source_file":
                csv_path.name,

            "Windows":
                valid_window_count,
        }
    )

    return (
        predictions,
        pair_metrics,
    )


# =========================================================
# 10. 主程序
# =========================================================

def main():
    # 参数检查
    if HISTORY_STEPS <= 0:
        raise ValueError(
            "HISTORY_STEPS 必须大于 0。"
        )

    if FUTURE_STEPS <= 0:
        raise ValueError(
            "FUTURE_STEPS 必须大于 0。"
        )

    if WINDOW_STRIDE <= 0:
        raise ValueError(
            "WINDOW_STRIDE 必须大于 0。"
        )

    if MOTION_LOOKBACK_STEPS <= 0:
        raise ValueError(
            "MOTION_LOOKBACK_STEPS 必须大于 0。"
        )

    if D2_M <= D1_M:
        raise ValueError(
            "D2_M 必须大于 D1_M。"
        )

    if not math.isclose(
        W_TIME + W_DISTANCE,
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "W_TIME 和 W_DISTANCE "
            "之和必须等于 1。"
        )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"输入文件夹不存在："
            f"{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_files = get_csv_files()

    if not csv_files:
        raise FileNotFoundError(
            f"输入文件夹中没有找到 CSV："
            f"{INPUT_DIR}"
        )

    all_prediction_frames = []
    per_file_metric_rows = []
    skipped_rows = []

    print(
        f"共发现 {len(csv_files)} "
        f"个 CSV 文件。"
    )

    print("-" * 80)

    for csv_path in csv_files:
        try:
            (
                predictions,
                file_metrics,
            ) = evaluate_one_pair(
                csv_path
            )

        except Exception as error:
            reason = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            skipped_rows.append(
                {
                    "source_file":
                        csv_path.name,

                    "reason":
                        reason,
                }
            )

            print(
                f"[跳过] {csv_path.name}: "
                f"{reason}"
            )

            continue

        if (
            predictions.empty
            or file_metrics is None
        ):
            reason = (
                f"有效行数不足 "
                f"{HISTORY_STEPS + FUTURE_STEPS}"
            )

            skipped_rows.append(
                {
                    "source_file":
                        csv_path.name,

                    "reason":
                        reason,
                }
            )

            print(
                f"[跳过] {csv_path.name}: "
                f"{reason}"
            )

            continue

        all_prediction_frames.append(
            predictions
        )

        per_file_metric_rows.append(
            file_metrics
        )

        print(
            f"[完成] {csv_path.name}: "
            f"{int(file_metrics['Windows'])} "
            f"个窗口，"
            f"{int(file_metrics['FuturePoints'])} "
            f"个预测点"
        )

    if not all_prediction_frames:
        raise RuntimeError(
            "没有生成有效预测。"
            "请检查字段名称、时间格式和文件长度。"
        )

    # 汇总所有船舶对
    all_predictions = pd.concat(
        all_prediction_frames,
        ignore_index=True,
    )

    per_file_metrics = pd.DataFrame(
        per_file_metric_rows
    )

    overall_metrics = calculate_metrics(
        true_cri=all_predictions[
            "true_CRI"
        ].to_numpy(),

        predicted_cri=all_predictions[
            "pred_CRI"
        ].to_numpy(),
    )

    # 完整汇总结果
    full_result = pd.DataFrame(
        [
            {
                "Model":
                    "ARPA-CPA",

                "Dataset":
                    DATASET_NAME,

                "MAE":
                    overall_metrics[
                        "MAE"
                    ],

                "MSE":
                    overall_metrics[
                        "MSE"
                    ],

                "Precision":
                    overall_metrics[
                        "Precision"
                    ],

                "Recall":
                    overall_metrics[
                        "Recall"
                    ],

                "F1-score":
                    overall_metrics[
                        "F1-score"
                    ],

                "PairFiles":
                    int(
                        all_predictions[
                            "source_file"
                        ].nunique()
                    ),

                "Windows":
                    int(
                        all_predictions[
                            "window_id"
                        ].nunique()
                    ),

                "FuturePoints":
                    overall_metrics[
                        "FuturePoints"
                    ],

                "TP":
                    overall_metrics[
                        "TP"
                    ],

                "FP":
                    overall_metrics[
                        "FP"
                    ],

                "FN":
                    overall_metrics[
                        "FN"
                    ],

                "TN":
                    overall_metrics[
                        "TN"
                    ],

                "HistorySteps":
                    HISTORY_STEPS,

                "FutureSteps":
                    FUTURE_STEPS,

                "WindowStride":
                    WINDOW_STRIDE,

                "HighRiskThreshold":
                    HIGH_RISK_THRESHOLD,

                "D1_m":
                    D1_M,

                "D2_m":
                    D2_M,

                "W_time":
                    W_TIME,

                "W_distance":
                    W_DISTANCE,
            }
        ]
    )

    # 论文 Table 11 需要的字段
    manuscript_columns = [
        "Model",
        "Dataset",
        "MAE",
        "MSE",
        "Precision",
        "Recall",
        "F1-score",
    ]

    manuscript_table = full_result[
        manuscript_columns
    ].copy()

    manuscript_table_rounded = (
        manuscript_table.copy()
    )

    metric_columns = [
        "MAE",
        "MSE",
        "Precision",
        "Recall",
        "F1-score",
    ]

    manuscript_table_rounded[
        metric_columns
    ] = (
        manuscript_table_rounded[
            metric_columns
        ].round(
            ROUND_DIGITS
        )
    )

    # =====================================================
    # 输出文件
    # =====================================================

    # 每一个未来预测点
    prediction_output_path = (
        OUTPUT_DIR
        / "arpa_cpa_predictions.csv"
    )

    # 每一个船舶对的单独指标
    per_file_output_path = (
        OUTPUT_DIR
        / "arpa_cpa_per_file_metrics.csv"
    )

    # 完整汇总信息
    full_result_output_path = (
        OUTPUT_DIR
        / "arpa_cpa_full_metrics.csv"
    )

    # 论文 Table 11 需要的数据，完整精度
    manuscript_output_path = (
        OUTPUT_DIR
        / "arpa_cpa_manuscript_table.csv"
    )

    # 论文 Table 11 需要的数据，保留五位小数
    rounded_output_path = (
        OUTPUT_DIR
        / "arpa_cpa_manuscript_table_rounded.csv"
    )

    # 被跳过的文件
    skipped_output_path = (
        OUTPUT_DIR
        / "arpa_cpa_skipped_files.csv"
    )

    all_predictions.to_csv(
        prediction_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    per_file_metrics.to_csv(
        per_file_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    full_result.to_csv(
        full_result_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    manuscript_table.to_csv(
        manuscript_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    manuscript_table_rounded.to_csv(
        rounded_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    if skipped_rows:
        pd.DataFrame(
            skipped_rows
        ).to_csv(
            skipped_output_path,
            index=False,
            encoding="utf-8-sig",
        )

    # =====================================================
    # 控制台显示
    # =====================================================

    print()
    print("=" * 80)
    print(
        "论文 Table 11 所需 ARPA-CPA 结果"
    )
    print("=" * 80)

    print(
        manuscript_table_rounded.to_string(
            index=False
        )
    )

    print("=" * 80)

    print(
        f"逐预测点结果："
        f"{prediction_output_path}"
    )

    print(
        f"每个船舶对指标："
        f"{per_file_output_path}"
    )

    print(
        f"完整汇总结果："
        f"{full_result_output_path}"
    )

    print(
        f"论文表格结果："
        f"{manuscript_output_path}"
    )

    print(
        f"论文表格五位小数结果："
        f"{rounded_output_path}"
    )

    if skipped_rows:
        print(
            f"跳过文件记录："
            f"{skipped_output_path}"
        )


if __name__ == "__main__":
    main()