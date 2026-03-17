import os
import json
import cv2
import numpy as np
import glob
import pandas as pd


def build_arc_samples(length_mm, step_mm):
    """
    生成从 0 到 length_mm 的弧长采样点，确保终点一定包含。
    """
    if length_mm <= 0:
        return np.array([0.0], dtype=np.float32)

    samples = np.arange(0, length_mm, step_mm, dtype=np.float32)
    if len(samples) == 0 or abs(float(samples[-1]) - float(length_mm)) > 1e-6:
        samples = np.append(samples, np.float32(length_mm))
    return samples


def remove_consecutive_duplicate_points(points, tol=1e-6):
    """
    去掉连续重复点，避免 linestrip 出现完全重合控制点。
    """
    if not points:
        return points

    cleaned = [points[0]]
    for p in points[1:]:
        last = cleaned[-1]
        if abs(p[0] - last[0]) > tol or abs(p[1] - last[1]) > tol:
            cleaned.append(p)
    return cleaned


def normalize_map(arr):
    arr = arr.astype(np.float32)
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    p95 = np.percentile(finite_vals, 95)
    if p95 < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)

    out = arr / float(p95)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def build_blue_edge_score_maps(img_bgr):
    """
    为蓝色 ILM / 杯底界面构建两个评分图：
    1. blue_score_map : 蓝色可信度
    2. edge_score_map : 边界强度（纵向梯度）
    """
    img_f = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img_f)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 蓝色掩膜（适度放宽）
    mask_blue = cv2.inRange(
        hsv,
        np.array([85, 30, 30], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8)
    ).astype(np.float32) / 255.0

    # 蓝色超额分量
    blue_excess = np.maximum(b - 0.65 * g - 0.65 * r, 0.0)
    blue_excess = normalize_map(blue_excess)

    # 综合蓝色评分
    blue_score_map = 0.7 * blue_excess + 0.3 * mask_blue
    blue_score_map = np.clip(blue_score_map, 0.0, 1.0).astype(np.float32)

    # 边界强度：沿 y 的梯度更适合水平界面
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_score_map = normalize_map(sobel_y)

    return blue_score_map, edge_score_map


def search_best_y_on_column(
    x,
    y_seed,
    blue_score_map,
    edge_score_map,
    search_half_window=18,
    blue_w=1.0,
    edge_w=0.35,
    prior_w=0.75,
    prior_sigma=5.0,
    fallback_score_threshold=0.82,
    blue_accept_threshold=0.10
):
    """
    在固定 x 列上，沿 y 搜索最佳候选点。
    逻辑：
    - 蓝色优先
    - 边界辅助
    - 离 y_seed 越近先验越强
    - 若蓝色不明显/总分不够，则回退到 y_seed
    """
    h, w = blue_score_map.shape
    x = int(round(x))
    x = max(0, min(w - 1, x))

    y_seed = float(y_seed)
    y0 = int(round(y_seed))

    y_min = max(0, y0 - search_half_window)
    y_max = min(h - 1, y0 + search_half_window)

    ys = np.arange(y_min, y_max + 1, dtype=np.int32)
    if len(ys) == 0:
        return y_seed, 0.0, "red_fallback"

    dy = ys.astype(np.float32) - y_seed
    prior = np.exp(-0.5 * (dy / prior_sigma) ** 2).astype(np.float32)

    score = (
        blue_w * blue_score_map[ys, x] +
        edge_w * edge_score_map[ys, x] +
        prior_w * prior
    )

    best_idx = int(np.argmax(score))
    best_y = float(ys[best_idx])
    best_score = float(score[best_idx])
    best_blue = float(blue_score_map[ys[best_idx], x])

    # 蓝色不明显，或者总分太低 -> 回退到红线 seed
    if best_blue < blue_accept_threshold and best_score < fallback_score_threshold:
        return float(y_seed), best_score, "red_fallback"

    return best_y, best_score, "blue_guided"


def refine_path_with_seed_and_blue(
    x_seq,
    y_seed_curve,
    blue_score_map,
    edge_score_map,
    fixed_start_y=None,
    search_half_window=16,
    blue_w=1.0,
    edge_w=0.30,
    prior_w=0.70,
    prior_sigma=5.0,
    trans_penalty=0.08
):
    """
    在给定 x 序列上，基于：
    - 红线 seed
    - 蓝色评分
    - 边界评分
    - 连续性惩罚
    做一条连续最优路径搜索（动态规划）。

    返回:
        y_path: shape (len(x_seq),)
    """
    x_seq = np.asarray(x_seq, dtype=np.int32)
    if len(x_seq) == 0:
        return np.array([], dtype=np.float32)

    h, w = blue_score_map.shape

    candidate_ys = []
    unary_scores = []

    for i, x in enumerate(x_seq):
        x = int(np.clip(x, 0, w - 1))

        if i == 0 and fixed_start_y is not None:
            ys = np.array([int(round(fixed_start_y))], dtype=np.int32)
            ys = np.clip(ys, 0, h - 1)
        else:
            y_seed = float(y_seed_curve[x])
            y0 = int(round(y_seed))
            ys = np.arange(y0 - search_half_window, y0 + search_half_window + 1, dtype=np.int32)
            ys = np.clip(ys, 0, h - 1)
            ys = np.unique(ys)

        y_seed = float(y_seed_curve[x])
        dy = ys.astype(np.float32) - y_seed
        prior = np.exp(-0.5 * (dy / prior_sigma) ** 2).astype(np.float32)

        unary = (
            blue_w * blue_score_map[ys, x] +
            edge_w * edge_score_map[ys, x] +
            prior_w * prior
        ).astype(np.float32)

        candidate_ys.append(ys)
        unary_scores.append(unary)

    n = len(candidate_ys)
    dp_scores = [None] * n
    back_ptrs = [None] * n

    dp_scores[0] = unary_scores[0].copy()
    back_ptrs[0] = -np.ones(len(candidate_ys[0]), dtype=np.int32)

    for i in range(1, n):
        ys_cur = candidate_ys[i].astype(np.float32)
        ys_prev = candidate_ys[i - 1].astype(np.float32)

        # 当前候选 j 与前一列候选 k 的连续性惩罚
        trans = trans_penalty * np.abs(ys_cur[:, None] - ys_prev[None, :])

        total = unary_scores[i][:, None] + dp_scores[i - 1][None, :] - trans
        best_prev_idx = np.argmax(total, axis=1)

        dp_scores[i] = total[np.arange(len(ys_cur)), best_prev_idx]
        back_ptrs[i] = best_prev_idx.astype(np.int32)

    # 回溯
    y_path = np.zeros(n, dtype=np.float32)
    last_idx = int(np.argmax(dp_scores[-1]))
    y_path[-1] = float(candidate_ys[-1][last_idx])

    for i in range(n - 1, 0, -1):
        last_idx = int(back_ptrs[i][last_idx])
        y_path[i - 1] = float(candidate_ys[i - 1][last_idx])

    return y_path


def compute_polyline_cum_arc_mm(xs, ys, scale_x_mm, scale_y_mm):
    """
    计算 2D 折线累计弧长（单位 mm）
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    if len(xs) <= 1:
        return np.array([0.0], dtype=np.float64)

    dx_mm = np.diff(xs) * scale_x_mm
    dy_mm = np.diff(ys) * scale_y_mm
    seg_len_mm = np.sqrt(dx_mm ** 2 + dy_mm ** 2).astype(np.float64)

    cum_arc = np.concatenate(([0.0], np.cumsum(seg_len_mm)))
    return cum_arc


def resample_polyline_xy_by_arc_mm(xs, ys, sample_s_mm, scale_x_mm, scale_y_mm):
    """
    按真实弧长对 2D 折线进行重采样
    返回:
        pts = [[x1,y1],[x2,y2],...]
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    sample_s_mm = np.asarray(sample_s_mm, dtype=np.float32)

    if len(xs) == 0:
        return []

    if len(xs) == 1:
        return [[float(xs[0]), float(ys[0])] for _ in sample_s_mm]

    cum_arc = compute_polyline_cum_arc_mm(xs, ys, scale_x_mm, scale_y_mm)
    total_len = float(cum_arc[-1])

    sample_s_mm = np.clip(sample_s_mm, 0.0, total_len)

    x_new = np.interp(sample_s_mm, cum_arc, xs).astype(np.float32)
    y_new = np.interp(sample_s_mm, cum_arc, ys).astype(np.float32)

    return [[float(px), float(py)] for px, py in zip(x_new, y_new)]


def auto_annotate_ilm_v2(patient_folder, excel_path, patient_id):
    print("🤖 启动 AI 前置打标引擎 V3 (红线引导 + 蓝色修正 + 中心轴Y向搜索)...")

    # =========================
    # 0. 读取 Excel，计算 scale_X
    # =========================
    try:
        df_baseline = pd.read_excel(excel_path)

        if 'Patient_ID' not in df_baseline.columns:
            print("  ❌ Excel 中缺少列: Patient_ID")
            return None
        if 'Axial_Length' not in df_baseline.columns:
            print("  ❌ Excel 中缺少列: Axial_Length")
            return None
        if 'Laterality' not in df_baseline.columns:
            print("  ❌ Excel 中缺少列: Laterality")
            return None

        patient_data = df_baseline[df_baseline['Patient_ID'] == patient_id]
        if patient_data.empty:
            print(f"  ❌ 严重错误: Excel 中未找到患者 [{patient_id}]")
            return None

        axial_length = float(patient_data.iloc[0]['Axial_Length'])
        laterality = str(patient_data.iloc[0]['Laterality']).strip().upper()

    except Exception as e:
        print(f"  ❌ Excel 读取失败: {e}")
        return None

    # =========================
    # 1. 比例尺
    # =========================
    native_spacing_X = 0.00586  # mm/pixel
    native_spacing_Y = 0.00529  # mm/pixel

    M = (0.01306 * (axial_length - 1.82)) / (0.01306 * (24.0 - 1.82))

    scale_X = native_spacing_X * M
    scale_Y = native_spacing_Y

    print(f"  👁️ 患者: {patient_id} | 眼别: {laterality} | 眼轴: {axial_length:.2f} mm")
    print(f"  📏 scale_X = {scale_X:.6f} mm/pixel | scale_Y = {scale_Y:.6f} mm/pixel | M = {M:.4f}")

    json_dir = os.path.join(patient_folder, "JSONs")
    image_dir = os.path.join(patient_folder, "伪彩")

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print(f"❌ 找不到 JSON 文件: {json_dir}")
        return None

    # =========================
    # 2. ROI 参数
    # =========================
    outer_mm = 0.10
    inner_target_mm = 0.50
    max_total_mm = 0.70
    step_mm = 0.02

    # 红线 / 蓝色搜索参数
    center_search_half_window = 18
    local_search_half_window = 16
    search_margin_mm = 0.12   # 为路径搜索多留一点 x 范围，最终仍按弧长裁剪

    auto_labels_to_remove = [
        'Center_ILM',
        'ILM_ROI',
        'ILM_ROI_Anchor',
        'ILM_ROI_OuterEnd',
        'ILM_ROI_InnerEnd'
    ]

    for json_path in json_files:
        filename = os.path.basename(json_path)
        file_num = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, f"{file_num}.png")

        if not os.path.exists(image_path):
            print(f"  ⚠️ 跳过 {filename}: 找不到对应图片 {image_path}")
            continue

        # 1. 绕过中文路径读取图片
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ⚠️ 跳过 {filename}: 图片读取失败")
            continue

        # 2. 读取现有 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 提取 BMO X 坐标
        bmo_x_list = []
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            pts = shape.get('points', [])
            if 'BMO' in label and len(pts) > 0:
                try:
                    bmo_x_list.append(float(pts[0][0]))
                except Exception:
                    pass

        if len(bmo_x_list) == 0:
            print(f"  ⚠️ 跳过 {filename}: 未找到 BMO 点")
            continue

        bmo_x_list = sorted(bmo_x_list)

        # 4. 提取粉红中心轴
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_pink = cv2.inRange(
            hsv,
            np.array([140, 50, 50], dtype=np.uint8),
            np.array([170, 255, 255], dtype=np.uint8)
        )
        col_sums = np.sum(mask_pink, axis=0)

        img_height, img_width = img.shape[:2]

        if np.max(col_sums) > 0:
            x_candidates = np.where(col_sums > np.max(col_sums) * 0.5)[0]
            x_center = int(np.mean(x_candidates))
        else:
            x_center = img_width // 2

        # 5. 提取全图红色 ILM 掩膜，构建 seed 曲线
        mask_red1 = cv2.inRange(
            hsv,
            np.array([0, 100, 100], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8)
        )
        mask_red2 = cv2.inRange(
            hsv,
            np.array([170, 100, 100], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8)
        )
        mask_red = mask_red1 | mask_red2

        top_y_array = np.full(img_width, np.nan, dtype=np.float32)

        red_y, red_x = np.where(mask_red > 0)
        for x in np.unique(red_x):
            top_y_array[x] = np.min(red_y[red_x == x])

        valid_x = np.where(~np.isnan(top_y_array))[0]
        all_x = np.arange(img_width, dtype=np.float32)

        if len(valid_x) > 0:
            interpolated_y = np.interp(all_x, valid_x, top_y_array[valid_x]).astype(np.float32)
        else:
            print(f"  ⚠️ {filename}: 未检测到红色 ILM，使用默认高度 y=100 托底")
            interpolated_y = np.full(img_width, 100.0, dtype=np.float32)

        # 6. 建蓝色/边界评分图
        blue_score_map, edge_score_map = build_blue_edge_score_maps(img)

        # 7. 中心锚点：固定中心轴，沿 y 搜索
        x_center = max(0, min(img_width - 1, int(round(x_center))))
        y_center_seed = float(interpolated_y[x_center])

        y_center_ilm, center_score, center_source = search_best_y_on_column(
            x=x_center,
            y_seed=y_center_seed,
            blue_score_map=blue_score_map,
            edge_score_map=edge_score_map,
            search_half_window=center_search_half_window
        )

        shape_center_ilm = {
            "label": "Center_ILM",
            "points": [[float(x_center), float(y_center_ilm)]],
            "group_id": None,
            "shape_type": "point",
            "flags": {}
        }

        # 8. 预计算 seed 曲线的累计弧长（用于定义搜索 x 范围）
        seed_cum_arc = compute_polyline_cum_arc_mm(all_x, interpolated_y, scale_X, scale_Y)

        new_shapes = []
        roi_reports = []

        # 9. 逐个 BMO 生成 ROI
        for group_id, bmo_x_raw in enumerate(bmo_x_list, start=1):
            anchor_x_int = int(round(bmo_x_raw))
            anchor_x_int = max(0, min(img_width - 1, anchor_x_int))

            # 先在该列上做一次单点修正，得到更可信的 anchor_y
            anchor_y_seed = float(interpolated_y[anchor_x_int])
            anchor_y_refined, anchor_score, anchor_source = search_best_y_on_column(
                x=anchor_x_int,
                y_seed=anchor_y_seed,
                blue_score_map=blue_score_map,
                edge_score_map=edge_score_map,
                search_half_window=local_search_half_window
            )

            # ------------------------------------------
            # 左侧 BMO：外侧向左，杯内向右
            # 右侧 BMO：外侧向右，杯内向左
            # ------------------------------------------
            if anchor_x_int < x_center:
                outward_x_full = np.arange(anchor_x_int, -1, -1, dtype=np.int32)
                inward_x_full = np.arange(anchor_x_int, img_width, 1, dtype=np.int32)
                side_name = "LEFT_BMO"
            else:
                outward_x_full = np.arange(anchor_x_int, img_width, 1, dtype=np.int32)
                inward_x_full = np.arange(anchor_x_int, -1, -1, dtype=np.int32)
                side_name = "RIGHT_BMO"

            outward_s_full = np.abs(seed_cum_arc[outward_x_full] - seed_cum_arc[anchor_x_int])
            inward_s_full = np.abs(seed_cum_arc[inward_x_full] - seed_cum_arc[anchor_x_int])

            available_outer_mm = float(outward_s_full[-1]) if len(outward_s_full) > 0 else 0.0
            available_inner_mm = float(inward_s_full[-1]) if len(inward_s_full) > 0 else 0.0

            actual_outer_mm = min(outer_mm, available_outer_mm)
            allowed_inner_by_total = max(0.0, max_total_mm - actual_outer_mm)
            actual_inner_mm = min(inner_target_mm, available_inner_mm, allowed_inner_by_total)

            if actual_outer_mm <= 0 and actual_inner_mm <= 0:
                roi_reports.append(f"G{group_id}: 无有效弧长")
                continue

            # 搜索时给一点额外余量，但最终仍按 actual_outer/inner 裁样
            outward_search_mm = min(actual_outer_mm + search_margin_mm, available_outer_mm)
            inward_search_mm = min(actual_inner_mm + search_margin_mm, available_inner_mm)

            outward_keep = outward_s_full <= outward_search_mm + 1e-6
            inward_keep = inward_s_full <= inward_search_mm + 1e-6

            outward_x = outward_x_full[outward_keep]
            inward_x = inward_x_full[inward_keep]

            if len(outward_x) < 2:
                outward_x = outward_x_full[:2] if len(outward_x_full) >= 2 else outward_x_full
            if len(inward_x) < 2:
                inward_x = inward_x_full[:2] if len(inward_x_full) >= 2 else inward_x_full

            # 沿两个方向分别做连续路径优化
            y_path_outward = refine_path_with_seed_and_blue(
                x_seq=outward_x,
                y_seed_curve=interpolated_y,
                blue_score_map=blue_score_map,
                edge_score_map=edge_score_map,
                fixed_start_y=anchor_y_refined,
                search_half_window=local_search_half_window
            )

            y_path_inward = refine_path_with_seed_and_blue(
                x_seq=inward_x,
                y_seed_curve=interpolated_y,
                blue_score_map=blue_score_map,
                edge_score_map=edge_score_map,
                fixed_start_y=anchor_y_refined,
                search_half_window=local_search_half_window
            )

            # 按真实弧长采样
            sample_s_outer = build_arc_samples(actual_outer_mm, step_mm)
            sample_s_inner = build_arc_samples(actual_inner_mm, step_mm)

            outer_points_anchor_to_end = resample_polyline_xy_by_arc_mm(
                xs=outward_x.astype(np.float32),
                ys=y_path_outward.astype(np.float32),
                sample_s_mm=sample_s_outer,
                scale_x_mm=scale_X,
                scale_y_mm=scale_Y
            )

            inner_points_anchor_to_end = resample_polyline_xy_by_arc_mm(
                xs=inward_x.astype(np.float32),
                ys=y_path_inward.astype(np.float32),
                sample_s_mm=sample_s_inner,
                scale_x_mm=scale_X,
                scale_y_mm=scale_Y
            )

            # 拼成：外侧端 -> ... -> 锚点 -> ... -> 杯内端
            roi_points = outer_points_anchor_to_end[::-1]
            if len(inner_points_anchor_to_end) > 1:
                roi_points.extend(inner_points_anchor_to_end[1:])  # 去掉重复锚点

            roi_points = remove_consecutive_duplicate_points(roi_points)

            if len(roi_points) < 2:
                roi_reports.append(f"G{group_id}: ROI点数不足")
                continue

            outer_end_point = outer_points_anchor_to_end[-1]
            inner_end_point = inner_points_anchor_to_end[-1]
            total_mm = actual_outer_mm + actual_inner_mm

            # 主 ROI 线
            new_shapes.append({
                "label": "ILM_ROI",
                "points": roi_points,
                "group_id": group_id,
                "description": (
                    f"{side_name}; "
                    f"anchor_source={anchor_source}; "
                    f"outer_mm={actual_outer_mm:.3f}; "
                    f"inner_mm={actual_inner_mm:.3f}; "
                    f"total_mm={total_mm:.3f}"
                ),
                "shape_type": "linestrip",
                "flags": {}
            })

            # 锚点
            new_shapes.append({
                "label": "ILM_ROI_Anchor",
                "points": [[float(anchor_x_int), float(anchor_y_refined)]],
                "group_id": group_id,
                "description": f"{side_name}; 锚点; source={anchor_source}",
                "shape_type": "point",
                "flags": {}
            })

            # 外侧止点
            new_shapes.append({
                "label": "ILM_ROI_OuterEnd",
                "points": [[float(outer_end_point[0]), float(outer_end_point[1])]],
                "group_id": group_id,
                "description": f"{side_name}; 外侧止点; outer_mm={actual_outer_mm:.3f}",
                "shape_type": "point",
                "flags": {}
            })

            # 杯内止点
            new_shapes.append({
                "label": "ILM_ROI_InnerEnd",
                "points": [[float(inner_end_point[0]), float(inner_end_point[1])]],
                "group_id": group_id,
                "description": f"{side_name}; 杯内止点; inner_mm={actual_inner_mm:.3f}",
                "shape_type": "point",
                "flags": {}
            })

            roi_reports.append(
                f"G{group_id}({side_name}): "
                f"anchor={anchor_source}; 外{actual_outer_mm:.3f} mm | "
                f"内{actual_inner_mm:.3f} mm | 总{total_mm:.3f} mm"
            )

        # 10. 清理旧自动标签并写回
        data['shapes'] = [
            s for s in data.get('shapes', [])
            if s.get('label') not in auto_labels_to_remove
        ]

        data['shapes'].append(shape_center_ilm)
        data['shapes'].extend(new_shapes)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if len(roi_reports) > 0:
            print(f"  ✅ {filename} 处理完毕 | Center={center_source} | center_score={center_score:.3f}")
            for rep in roi_reports:
                print(f"     -> {rep}")
        else:
            print(f"  ⚠️ {filename} 未生成有效 ILM_ROI | Center={center_source} | center_score={center_score:.3f}")

    print("\n🎉 V3 引擎打标完成！")


# ================= 调用区 =================
patient_folder = r"D:\桌面来的文件\测试集png格式\xu lingxi"
my_excel = r"D:\桌面来的文件\测试集png格式\基线数据.xls"
patient_id = "xu lingxi"

auto_annotate_ilm_v2(patient_folder, my_excel, patient_id)
