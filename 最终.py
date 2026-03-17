import json
import os
import math
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import importlib.util
from datetime import datetime

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from matplotlib.path import Path
from scipy.interpolate import splprep, splev

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(errors='replace')
except Exception:
    pass


def normalize_identifier(name):
    return ''.join(ch for ch in str(name).strip().lower() if ch.isalnum())


def choose_image_dir(patient_folder, json_dir=None):
    preferred_names = ['增强', '澧炲己', '伪彩', '原图', '反色']
    name_bonus = {name: (len(preferred_names) - idx) * 100 for idx, name in enumerate(preferred_names)}
    json_count = 0
    if json_dir and os.path.isdir(json_dir):
        json_count = len(glob.glob(os.path.join(json_dir, '*.json')))

    candidates = []
    if os.path.isdir(patient_folder):
        for entry in sorted(os.scandir(patient_folder), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            if entry.name == 'JSONs':
                continue
            png_count = len(glob.glob(os.path.join(entry.path, '*.png')))
            if png_count <= 0:
                continue

            score = png_count
            score += name_bonus.get(entry.name, 0)
            if json_count > 0:
                score -= abs(png_count - json_count) * 3
                if png_count == json_count:
                    score += 120

            candidates.append({
                'path': entry.path,
                'name': entry.name,
                'png_count': png_count,
                'score': score
            })

    if not candidates:
        return {'image_dir': None, 'png_count': 0, 'score': -1, 'reason': 'no_png_subdir'}

    best = max(candidates, key=lambda x: (x['score'], x['png_count'], x['name']))
    reason = f"name={best['name']}, png_count={best['png_count']}, score={best['score']}, candidates={len(candidates)}"
    return {
        'image_dir': best['path'],
        'png_count': best['png_count'],
        'score': best['score'],
        'reason': reason
    }


def auto_discover_paths(base_dir, patient_id_hint=None, excel_hint=None):
    discovered = {
        'base_dir': base_dir,
        'excel_path': None,
        'patient_folder': None,
        'json_dir': None,
        'image_dir': None,
        'notes': [],
        'patient_score': None,
        'image_reason': None,
    }

    # 1) Excel discovery
    excel_candidates = []
    if excel_hint and os.path.isfile(excel_hint):
        excel_candidates.append(excel_hint)

    for ext in ('*.xls', '*.xlsx'):
        excel_candidates.extend(glob.glob(os.path.join(base_dir, ext)))

    if not excel_candidates:
        for ext in ('*.xls', '*.xlsx'):
            excel_candidates.extend(glob.glob(os.path.join(base_dir, '*', ext)))

    excel_candidates = sorted(set(excel_candidates))
    if excel_candidates:
        scored_excel = []
        for pth in excel_candidates:
            base = os.path.basename(pth).lower()
            ext = os.path.splitext(base)[1]
            depth = pth.replace('\\', '/').count('/') - base_dir.replace('\\', '/').count('/')
            score = 0
            if base in ('data.xls', 'data.xlsx', 'baseline.xls', 'baseline.xlsx'):
                score += 300
            if 'data' in base:
                score += 120
            if 'base' in base:
                score += 80
            if ext == '.xls':
                score += 40
            score -= max(depth, 0) * 10
            scored_excel.append((score, pth))

        scored_excel.sort(key=lambda x: (x[0], x[1]), reverse=True)
        discovered['excel_path'] = scored_excel[0][1]
        discovered['notes'].append(f"excel_selected={discovered['excel_path']}")
    else:
        discovered['notes'].append("excel_selected=None")

    # 2) Patient folder + JSON/image discovery
    hint_norm = normalize_identifier(patient_id_hint)
    patient_candidates = []
    skip_names = {'.idea', '.venv', '__pycache__', 'Final_Analysis_Output'}

    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if entry.name in skip_names:
            continue

        patient_dir = entry.path
        json_dir = os.path.join(patient_dir, 'JSONs')
        json_count = len(glob.glob(os.path.join(json_dir, '*.json'))) if os.path.isdir(json_dir) else 0
        image_pick = choose_image_dir(patient_dir, json_dir)
        image_dir = image_pick['image_dir']
        png_count = image_pick['png_count']

        if json_count <= 0 and image_dir is None:
            continue

        folder_norm = normalize_identifier(entry.name)
        score = 0
        if hint_norm:
            if folder_norm == hint_norm:
                score += 300
            elif hint_norm in folder_norm or folder_norm in hint_norm:
                score += 120
        if json_count > 0:
            score += 140 + min(json_count, 100)
        if image_dir:
            score += 120 + min(png_count, 100)
            score += image_pick['score']
        if json_count > 0 and png_count > 0:
            score -= abs(json_count - png_count) * 5
            if json_count == png_count:
                score += 150

        patient_candidates.append({
            'score': score,
            'patient_folder': patient_dir,
            'json_dir': json_dir if os.path.isdir(json_dir) else None,
            'image_dir': image_dir,
            'json_count': json_count,
            'png_count': png_count,
            'image_reason': image_pick['reason'],
        })

    if patient_candidates:
        best = max(patient_candidates, key=lambda x: (x['score'], x['json_count'], x['png_count']))
        discovered['patient_folder'] = best['patient_folder']
        discovered['json_dir'] = best['json_dir']
        discovered['image_dir'] = best['image_dir']
        discovered['patient_score'] = best['score']
        discovered['image_reason'] = best['image_reason']
        discovered['notes'].append(
            f"patient_selected={best['patient_folder']} (json={best['json_count']}, png={best['png_count']}, score={best['score']})"
        )
    else:
        discovered['notes'].append("patient_selected=None")

    return discovered


def process_full_eye_to_3d_point_cloud(patient_folder, excel_path, patient_id, json_dir=None, image_dir=None):
    print(f"\n🚀 启动全眼 3D 重建矩阵 | 当前患者: [{patient_id}]")

    # 路径拼接
    if json_dir is None:
        json_dir = os.path.join(patient_folder, "JSONs")
    if image_dir is None:
        image_pick = choose_image_dir(patient_folder, json_dir)
        image_dir = image_pick['image_dir'] if image_pick['image_dir'] else os.path.join(patient_folder, "增强")

    # 基础路径检查
    if not os.path.exists(patient_folder):
        print(f"  ❌ 患者文件夹不存在: {patient_folder}")
        return None
    if not os.path.exists(json_dir):
        print(f"  ❌ JSON 文件夹不存在: {json_dir}")
        return None
    if not os.path.exists(image_dir):
        print(f"  ❌ 图像文件夹不存在: {image_dir}")
        return None
    if not os.path.exists(excel_path):
        print(f"  ❌ Excel 文件不存在: {excel_path}")
        return None

    # ==========================================
    # 0. 自动查阅 Excel 基线数据
    # ==========================================
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

    # 比例尺校正
    native_spacing_X = 0.00586  # mm/pixel
    native_spacing_Z = 0.00529  # mm/pixel
    M = (0.01306 * (axial_length - 1.82)) / (0.01306 * (24.0 - 1.82))
    scale_X = native_spacing_X * M
    scale_Z = native_spacing_Z

    print(f"  👁️ 眼别: {laterality} | 眼轴: {axial_length} mm | M系数: {M:.4f}")

    # ==========================================
    # 大循环：准备收纳所有 3D 点
    # ==========================================
    cloud_3d = {
        'BMO': [],
        'ALI': [],
        'ALCS': [],
        'ILM_ROI': [],
        'BMO_META': [],
        'ILM_META': [],
        'ALI_META': [],
        'ALCS_META': [],
        'SLICE_META': {},
        'laterality': laterality,
        'axial_length': float(axial_length),
        'z_stabilization_status': 'unknown',
    }

    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    json_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not json_files:
        print(f"  ❌ 文件夹内未找到 JSON 文件: {json_dir}")
        return None

    # =========================================================
    # 第一遍扫描 (Pass 1) - 提前提取 Z 轴防抖基准
    # =========================================================
    center_z_list = []
    file_center_map = {}
    for j_path in json_files:
        try:
            with open(j_path, 'r', encoding='utf-8') as f:
                temp_data = json.load(f)
            for shape in temp_data.get('shapes', []):
                if shape.get('label') == 'Center_ILM':
                    z_phys = shape['points'][0][1] * scale_Z
                    center_z_list.append(z_phys)
                    file_center_map[j_path] = z_phys
        except Exception:
            pass

    if center_z_list:
        ref_center_z = np.median(center_z_list)  # 用中位数代替均值，更稳
        drift_um = (np.max(center_z_list) - np.min(center_z_list)) * 1000
        cloud_3d['z_stabilization_status'] = 'active_center_ilm'

        print(f"\n  🔬 【防抖质控报告 (Z-axis Motion Audit)】")
        print(f"     -> 成功提取中心锚点: {len(center_z_list)}/12 张")
        print(f"     -> Center_ILM 中位参考深度: {ref_center_z:.4f} mm")
        print(f"     -> 最大扫描漂移极差: {drift_um:.1f} μm")
        print(f"     -> 补偿策略: <20 μm 不补偿 | 20–50 μm 半量补偿 | >50 μm 全量补偿")
    else:
        ref_center_z = 0
        cloud_3d['z_stabilization_status'] = 'skipped_no_center_ilm'
        print("\n  ⚠️ 警告: 未检测到 Center_ILM，Z 轴防抖补偿将跳过。")

    mrw_segments_list = []   # 初始化 24 条真实 MRW 段收集器

    for json_path in json_files:
        filename = os.path.basename(json_path)

        try:
            file_num = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"  ⚠️ 跳过无法解析编号的 JSON 文件: {filename}")
            continue

        image_path = os.path.join(image_dir, f"{file_num}.png")
        angle_radian = math.radians((file_num - 1) * 15)

        if not os.path.exists(image_path):
            print(f"  ⚠️ 跳过 {file_num}: 对应图片不存在 {image_path}")
            continue

        # ----------------------------------------
        # OpenCV 寻找旋转中心
        # ----------------------------------------
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if img is None:
            print(f"  ⚠️ 跳过 {file_num}: 图片解码失败 {image_path}")
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([170, 255, 255]))
        column_sums = np.sum(mask, axis=0)

        if np.max(column_sums) > 0:
            pink_cols = np.where(column_sums > (np.max(column_sums) * 0.5))[0]
            if len(pink_cols) > 0:
                x_center = np.mean(pink_cols)
            else:
                x_center = img.shape[1] / 2
        else:
            x_center = img.shape[1] / 2

        # ----------------------------------------
        # JSON 坐标读取与映射
        # ----------------------------------------
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️ 跳过 {file_num}: JSON 读取失败 -> {e}")
            continue

        # =========================================================
        # 获取防抖差值并建立原位切片缓存
        # =========================================================
        current_center_z = file_center_map.get(json_path, ref_center_z)
        raw_delta_z = current_center_z - ref_center_z  # mm
        raw_delta_um = raw_delta_z * 1000  # μm

        # 阈值触发 + 限幅补偿
        if abs(raw_delta_um) < 20:
            delta_z = 0.0
        elif abs(raw_delta_um) < 50:
            delta_z = 0.5 * raw_delta_z
        else:
            delta_z = raw_delta_z

        cloud_3d['SLICE_META'][file_num] = {
            'slice_id': int(file_num),
            'image_path': image_path,
            'json_path': json_path,
            'x_center': float(x_center),
            'delta_z': float(delta_z),
            'scale_X': float(scale_X),
            'scale_Z': float(scale_Z),
            'angle_deg': float((file_num - 1) * 15.0),
            'laterality': laterality,
        }

        slice_bmo = []  # 当前切片的 BMO 点，后面按 pixel_x 分左右
        slice_ilm = []  # 当前切片的 ILM_ROI 点，后面按 pixel_x 分成左右两段

        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])

            if not points:
                continue

            for pt in points:
                if len(pt) < 2:
                    continue

                pixel_x, pixel_y = pt[0], pt[1]

                # 核心 3D 转换矩阵
                r_mm = (pixel_x - x_center) * scale_X
                X_3D = r_mm * math.cos(angle_radian)
                Y_3D = r_mm * math.sin(angle_radian)
                Z_3D = pixel_y * scale_Z

                if laterality == 'L':
                    X_3D = -X_3D

                # 应用 Z 轴防抖
                Z_3D = Z_3D - delta_z

                pt3d = (X_3D, Y_3D, Z_3D)

                if 'BMO' in label:
                    cloud_3d['BMO'].append(pt3d)
                    cloud_3d['BMO_META'].append({
                        'slice_id': file_num,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'point_3d': pt3d
                    })
                    slice_bmo.append({
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'point_3d': pt3d
                    })

                elif 'ALI' in label:
                    cloud_3d['ALI'].append(pt3d)
                    cloud_3d['ALI_META'].append({
                        'slice_id': file_num,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'point_3d': pt3d
                    })

                elif 'ALCS' in label:
                    cloud_3d['ALCS'].append(pt3d)
                    cloud_3d['ALCS_META'].append({
                        'slice_id': file_num,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'point_3d': pt3d
                    })

                elif 'ILM_ROI' in label:
                    cloud_3d['ILM_ROI'].append(pt3d)
                    cloud_3d['ILM_META'].append({
                        'slice_id': file_num,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'point_3d': pt3d
                    })
                    slice_ilm.append({
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'point_3d': pt3d
                    })

        # =========================================================
        # 在同切片内计算真实 MRW：
        # 左 BMO -> 左侧 ILM_ROI
        # 右 BMO -> 右侧 ILM_ROI
        # =========================================================
        if len(slice_bmo) >= 2 and len(slice_ilm) >= 4:
            # 1. BMO 按 pixel_x 区分左右端点
            slice_bmo_sorted = sorted(slice_bmo, key=lambda d: d['pixel_x'])
            left_bmo = slice_bmo_sorted[0]
            right_bmo = slice_bmo_sorted[-1]

            left_bmo_pt = np.array(left_bmo['point_3d'], dtype=float)
            right_bmo_pt = np.array(right_bmo['point_3d'], dtype=float)

            # 同切片 BMO 基线（仅用于角度质控）
            bmo_baseline_vec = right_bmo_pt - left_bmo_pt

            # 2. ILM_ROI 按 pixel_x 排序后，用最大间隙切成左右两段
            slice_ilm_sorted = sorted(slice_ilm, key=lambda d: d['pixel_x'])
            ilm_x = np.array([item['pixel_x'] for item in slice_ilm_sorted], dtype=float)

            gaps = np.diff(ilm_x)
            if len(gaps) > 0:
                split_idx = int(np.argmax(gaps)) + 1

                left_ilm_items = slice_ilm_sorted[:split_idx]
                right_ilm_items = slice_ilm_sorted[split_idx:]

                if len(left_ilm_items) >= 2 and len(right_ilm_items) >= 2:
                    left_ilm_polyline = np.array(
                        [item['point_3d'] for item in left_ilm_items],
                        dtype=float
                    )
                    right_ilm_polyline = np.array(
                        [item['point_3d'] for item in right_ilm_items],
                        dtype=float
                    )

                    # 3. 左 BMO -> 左 ILM_ROI
                    closest_left_ilm_pt, left_dist_mm = closest_point_on_polyline_3d(
                        left_bmo_pt, left_ilm_polyline
                    )
                    if closest_left_ilm_pt is not None:
                        left_dist_um = left_dist_mm * 1000.0
                        if left_dist_um < 800:
                            left_vec = closest_left_ilm_pt - left_bmo_pt
                            left_angle_deg = acute_angle_between_vectors(left_vec, bmo_baseline_vec)
                            abs_angle = anatomical_angle_deg(left_bmo_pt[0], left_bmo_pt[1])

                            mrw_segments_list.append({
                                'slice_id': file_num,
                                'side': 'L',
                                'angle': float(abs_angle),
                                'bmo_pt': tuple(left_bmo_pt.tolist()),
                                'ilm_pt': tuple(closest_left_ilm_pt.tolist()),
                                'mrw_vector': tuple(left_vec.tolist()),
                                'mrw_len': float(left_dist_um),
                                'mrw_angle_deg': float(left_angle_deg)
                            })

                    # 4. 右 BMO -> 右 ILM_ROI
                    closest_right_ilm_pt, right_dist_mm = closest_point_on_polyline_3d(
                        right_bmo_pt, right_ilm_polyline
                    )
                    if closest_right_ilm_pt is not None:
                        right_dist_um = right_dist_mm * 1000.0
                        if right_dist_um < 800:
                            right_vec = closest_right_ilm_pt - right_bmo_pt
                            right_angle_deg = acute_angle_between_vectors(right_vec, bmo_baseline_vec)
                            abs_angle = anatomical_angle_deg(right_bmo_pt[0], right_bmo_pt[1])

                            mrw_segments_list.append({
                                'slice_id': file_num,
                                'side': 'R',
                                'angle': float(abs_angle),
                                'bmo_pt': tuple(right_bmo_pt.tolist()),
                                'ilm_pt': tuple(closest_right_ilm_pt.tolist()),
                                'mrw_vector': tuple(right_vec.tolist()),
                                'mrw_len': float(right_dist_um),
                                'mrw_angle_deg': float(right_angle_deg)
                            })

        print(f"  ✅ 切片 {file_num} ({(file_num - 1) * 15}°) 处理完毕 | X_center: {x_center:.1f}")

    # ==========================================
    # 汇总报告
    # ==========================================
    print(f"\n🎉 完美收官！全眼 3D 点云坐标库构建完毕：")
    print(f"   -> BMO 环点云共计: {len(cloud_3d['BMO'])} 个点")
    print(f"   -> ALI 边界共计: {len(cloud_3d['ALI'])} 个点")
    print(f"   -> 筛板表面 ALCS 共计: {len(cloud_3d['ALCS'])} 个点")

    # =========================================================
    # 偏心度审计 (Eccentricity Audit)
    # =========================================================
    if len(cloud_3d['BMO']) >= 3:
        bmo_arr = np.array(cloud_3d['BMO'])
        bmo_centroid_2d = np.mean(bmo_arr[:, :2], axis=0)
        eccentricity = np.linalg.norm(bmo_centroid_2d)
        print(f"\n  🎯 【偏心度审计 (Eccentricity Audit)】")
        print(f"     -> BMO 质心偏离机器扫描中心距离: {eccentricity:.3f} mm")
        if eccentricity > 0.3:
            print(f"     ⚠️ 提示: 固视偏差较大 (偏心度 > 0.3 mm)")

    if len(cloud_3d['BMO']) < 3:
        print("  ❌ BMO 点不足，后续无法进行最佳拟合平面")
        return None

    return cloud_3d, mrw_segments_list


def fit_bmo_bfp_geometry(cloud_3d):
    if cloud_3d is None or 'BMO' not in cloud_3d or len(cloud_3d['BMO']) < 3:
        return None, None

    bmo_pts = np.array(cloud_3d['BMO'], dtype=float)
    centroid = np.mean(bmo_pts, axis=0)
    centered_bmo = bmo_pts - centroid
    _, _, Vt = np.linalg.svd(centered_bmo)
    normal = Vt[-1, :]

    if normal[2] < 0:
        normal = -normal

    return centroid, normal


def align_to_bmo_bfp(cloud_3d):
    print("\n⚖️ 启动 BMO 最佳拟合平面 (BFP) 空间矫正引擎...")

    if cloud_3d is None:
        print("  ❌ 错误: 输入 cloud_3d 为 None，通常是前一步读取 Excel 或生成点云失败。")
        return None

    if 'BMO' not in cloud_3d or not cloud_3d['BMO'] or len(cloud_3d['BMO']) < 3:
        print("  ❌ 错误: BMO 点数量不足以拟合平面 (至少需要 3 个)！")
        return None

    centroid, normal = fit_bmo_bfp_geometry(cloud_3d)
    if centroid is None or normal is None:
        print("  ❌ 错误: BMO-BFP 几何拟合失败。")
        return None

    print(f"  📍 原始 BMO 质心坐标: X={centroid[0]:.3f}, Y={centroid[1]:.3f}, Z={centroid[2]:.3f}")
    print(f"  📐 BFP 原始法向量: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

    target_z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target_z)
    axis_length = np.linalg.norm(axis)

    if axis_length > 1e-8:
        axis = axis / axis_length
        angle = np.arccos(np.clip(np.dot(normal, target_z), -1.0, 1.0))
        rotation_vector = axis * angle
        rotation = R.from_rotvec(rotation_vector)
    else:
        rotation = R.from_rotvec(np.array([0.0, 0.0, 0.0]))

    aligned_cloud = {
        'BMO': [],
        'ALI': [],
        'ALCS': [],
        'ILM_ROI': [],
        'BMO_META': [],
        'ILM_META': [],
        'ALI_META': [],
        'ALCS_META': [],
        'SLICE_META': cloud_3d.get('SLICE_META', {}),
        'laterality': cloud_3d.get('laterality', 'R'),
        'axial_length': cloud_3d.get('axial_length', np.nan),
        'z_stabilization_status': cloud_3d.get('z_stabilization_status', 'unknown'),
        'ALIGNMENT': {
            'centroid': tuple(np.asarray(centroid, dtype=float).tolist()),
            'rotation_matrix': rotation.as_matrix().tolist(),
        }
    }

    # 普通点云旋转
    for key in ['BMO', 'ALI', 'ALCS', 'ILM_ROI']:
        if key in cloud_3d and cloud_3d[key]:
            pts = np.array(cloud_3d[key], dtype=float)
            pts_centered = pts - centroid
            pts_aligned = rotation.apply(pts_centered)
            aligned_cloud[key] = pts_aligned.tolist()

    # BMO_META 单独旋转
    if 'BMO_META' in cloud_3d and cloud_3d['BMO_META']:
        for item in cloud_3d['BMO_META']:
            pt = np.array(item['point_3d'], dtype=float)
            pt_centered = pt - centroid
            pt_aligned = rotation.apply(pt_centered)
            aligned_cloud['BMO_META'].append({
                'slice_id': item['slice_id'],
                'pixel_x': item['pixel_x'],
                'pixel_y': item.get('pixel_y', np.nan),
                'point_3d': tuple(pt_aligned.tolist())
            })

    # ILM_META 单独旋转
    if 'ILM_META' in cloud_3d and cloud_3d['ILM_META']:
        for item in cloud_3d['ILM_META']:
            pt = np.array(item['point_3d'], dtype=float)
            pt_centered = pt - centroid
            pt_aligned = rotation.apply(pt_centered)
            aligned_cloud['ILM_META'].append({
                'slice_id': item['slice_id'],
                'pixel_x': item['pixel_x'],
                'pixel_y': item.get('pixel_y', np.nan),
                'point_3d': tuple(pt_aligned.tolist())
            })

    # ALI_META 单独旋转
    if 'ALI_META' in cloud_3d and cloud_3d['ALI_META']:
        for item in cloud_3d['ALI_META']:
            pt = np.array(item['point_3d'], dtype=float)
            pt_centered = pt - centroid
            pt_aligned = rotation.apply(pt_centered)
            aligned_cloud['ALI_META'].append({
                'slice_id': item['slice_id'],
                'pixel_x': item['pixel_x'],
                'pixel_y': item.get('pixel_y', np.nan),
                'point_3d': tuple(pt_aligned.tolist())
            })


    # ALCS_META ????
    if 'ALCS_META' in cloud_3d and cloud_3d['ALCS_META']:
        for item in cloud_3d['ALCS_META']:
            pt = np.array(item['point_3d'], dtype=float)
            pt_centered = pt - centroid
            pt_aligned = rotation.apply(pt_centered)
            aligned_cloud['ALCS_META'].append({
                'slice_id': item['slice_id'],
                'pixel_x': item['pixel_x'],
                'pixel_y': item.get('pixel_y', np.nan),
                'point_3d': tuple(pt_aligned.tolist())
            })

    new_bmo_centroid = np.mean(np.array(aligned_cloud['BMO']), axis=0)

    print(f"  ✅ 空间矫正完毕！当前 BMO 质心已被锚定在: "
          f"[{new_bmo_centroid[0]:.3f}, {new_bmo_centroid[1]:.3f}, {new_bmo_centroid[2]:.3f}]")
    print("  🌊 整个视盘已放平，BMO 所在的平面现在就是绝对零度海平面 (Z ≈ 0)！")

    return aligned_cloud


def anatomical_angle_deg(x, y):
    """
    输入已经统一到同一解剖坐标系后的 x, y
    输出 0~360° 的角度
    """
    ang = np.degrees(np.arctan2(y, x))
    if np.isscalar(ang):
        return ang + 360 if ang < 0 else ang
    return np.where(ang < 0, ang + 360, ang)


def closest_point_on_segment_3d(p, a, b):
    """
    点 p 到 3D 线段 ab 的最近点
    返回:
        closest_pt, distance_mm
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    ab2 = np.dot(ab, ab)

    if ab2 < 1e-12:
        closest = a.copy()
        dist = np.linalg.norm(p - closest)
        return closest, dist

    t = np.dot(p - a, ab) / ab2
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab
    dist = np.linalg.norm(p - closest)
    return closest, dist


def closest_point_on_polyline_3d(p, polyline_pts):
    """
    点到 3D 折线的最近点
    polyline_pts: shape (N, 3)
    返回:
        best_point, best_dist_mm
    """
    polyline_pts = np.asarray(polyline_pts, dtype=float)

    if len(polyline_pts) < 2:
        if len(polyline_pts) == 1:
            only_pt = polyline_pts[0]
            return only_pt, np.linalg.norm(np.asarray(p) - only_pt)
        return None, np.inf

    best_dist = np.inf
    best_point = None

    for i in range(len(polyline_pts) - 1):
        a = polyline_pts[i]
        b = polyline_pts[i + 1]
        cp, dist = closest_point_on_segment_3d(p, a, b)
        if dist < best_dist:
            best_dist = dist
            best_point = cp

    return best_point, best_dist


def acute_angle_between_vectors(v1, v2):
    """
    计算两个向量的锐角夹角，范围 0~90°
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-12 or n2 < 1e-12:
        return np.nan

    cos_theta = np.dot(v1, v2) / (n1 * n2)
    cos_theta = np.clip(np.abs(cos_theta), 0.0, 1.0)

    theta_deg = np.degrees(np.arccos(cos_theta))
    return theta_deg


def build_slice_bmo_dict(bmo_meta):
    slice_dict = {}
    for item in bmo_meta:
        sid = item['slice_id']
        slice_dict.setdefault(sid, []).append(item)

    out = {}
    for sid, items in slice_dict.items():
        items_sorted = sorted(items, key=lambda d: d['pixel_x'])
        if len(items_sorted) >= 2:
            out[sid] = {
                'L': items_sorted[0],
                'R': items_sorted[-1]
            }
    return out


def split_slice_ilm_into_left_right(slice_items):
    """
    每张 B-scan 有两段 ILM_ROI，左右不相连。
    这里按 pixel_x 排序后，用最大间隙切成两段。
    """
    if not slice_items or len(slice_items) < 4:
        return None, None

    items_sorted = sorted(slice_items, key=lambda d: d['pixel_x'])
    x_vals = np.array([it['pixel_x'] for it in items_sorted], dtype=float)
    gaps = np.diff(x_vals)

    if len(gaps) == 0:
        return None, None

    split_idx = int(np.argmax(gaps)) + 1

    left_items = items_sorted[:split_idx]
    right_items = items_sorted[split_idx:]

    if len(left_items) < 2 or len(right_items) < 2:
        return None, None

    left_poly = np.array([it['point_3d'] for it in left_items], dtype=float)
    right_poly = np.array([it['point_3d'] for it in right_items], dtype=float)

    return left_poly, right_poly


def build_slice_ilm_segment_dict(ilm_meta):
    slice_dict = {}
    for item in ilm_meta:
        sid = item['slice_id']
        slice_dict.setdefault(sid, []).append(item)

    out = {}
    for sid, items in slice_dict.items():
        left_poly, right_poly = split_slice_ilm_into_left_right(items)
        if left_poly is not None and right_poly is not None:
            out[sid] = {
                'L': left_poly,
                'R': right_poly
            }
    return out


def build_ordered_bmo24_from_meta(bmo_meta):
    """
    固定顺序：
        1L -> 2L -> ... -> 12L -> 1R -> 2R -> ... -> 12R
    """
    bmo_dict = build_slice_bmo_dict(bmo_meta)
    ordered = []

    for sid in sorted(bmo_dict.keys()):
        if 'L' in bmo_dict[sid]:
            ordered.append({
                'slice_id': sid,
                'side': 'L',
                'point_3d': np.array(bmo_dict[sid]['L']['point_3d'], dtype=float)
            })

    for sid in sorted(bmo_dict.keys()):
        if 'R' in bmo_dict[sid]:
            ordered.append({
                'slice_id': sid,
                'side': 'R',
                'point_3d': np.array(bmo_dict[sid]['R']['point_3d'], dtype=float)
            })

    return ordered


def build_bmo24_index_map(ordered_bmo24):
    index_map = {}
    for idx, item in enumerate(ordered_bmo24):
        index_map[(item['slice_id'], item['side'])] = idx
    return index_map


def convert_points_to_slice_ref_2d(points_3d, origin_3d, ref_dir_xy):
    """
    将 3D 点投到当前 B-scan 的 2D 参考坐标中：
        x: 沿当前切片左右 BMO 连线方向
        y: 深度方向（默认让“朝 ILM 方向”为正）
    """
    pts = np.asarray(points_3d, dtype=float)
    origin_3d = np.asarray(origin_3d, dtype=float)
    ref_dir_xy = np.asarray(ref_dir_xy, dtype=float)

    if pts.ndim == 1:
        pts = pts[None, :]

    n = np.linalg.norm(ref_dir_xy)
    if n < 1e-12:
        return None

    ref_dir_xy = ref_dir_xy / n

    rel_xy = pts[:, :2] - origin_3d[:2]
    x = rel_xy @ ref_dir_xy

    # 让 ILM 通常位于 y>0 方向
    y = -(pts[:, 2] - origin_3d[2])
    if np.nanmax(y) < 0:
        y = -y

    out = np.column_stack([x, y])
    return out


def ray_segment_intersection_2d(ray_dir, p0, p1, eps=1e-10):
    """
    射线: origin=(0,0), direction=ray_dir
    线段: p0 -> p1
    若相交，返回射线上参数 t 和线段参数 u
    """
    d = np.asarray(ray_dir, dtype=float)
    s = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    p0 = np.asarray(p0, dtype=float)

    mat = np.array([[d[0], -s[0]],
                    [d[1], -s[1]]], dtype=float)
    rhs = p0

    det = np.linalg.det(mat)
    if abs(det) < eps:
        return None

    try:
        t, u = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return None

    if t >= 0 and 0 <= u <= 1:
        return t, u
    return None


def intersect_ray_with_polyline_in_slice_2d(origin_3d, ref_dir_xy, polyline_3d, angle_deg):
    """
    在当前 B-scan 的 2D 截面中，求从 BMO 点出发、
    与参考线夹角为 angle_deg 的射线，与 ILM 折线的最近交点。

    约定：
        0°  = 平行于该 B-scan 内左右 BMO 连线
        90° = 垂直向 ILM 方向
    返回:
        rw_mm, hit_point_3d
    """
    polyline_3d = np.asarray(polyline_3d, dtype=float)
    if len(polyline_3d) < 2:
        return None, None

    poly2d = convert_points_to_slice_ref_2d(polyline_3d, origin_3d, ref_dir_xy)
    if poly2d is None or len(poly2d) < 2:
        return None, None

    theta = np.deg2rad(angle_deg)
    ray_dir = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    best_t = np.inf
    best_hit = None

    for i in range(len(poly2d) - 1):
        p0 = poly2d[i]
        p1 = poly2d[i + 1]
        hit = ray_segment_intersection_2d(ray_dir, p0, p1)
        if hit is None:
            continue

        t, u = hit
        if t < best_t:
            seg_pt_3d = polyline_3d[i] + u * (polyline_3d[i + 1] - polyline_3d[i])
            best_t = t
            best_hit = seg_pt_3d

    if best_hit is None:
        return None, None

    return float(best_t), np.asarray(best_hit, dtype=float)


def calculate_gardiner_mra(aligned_cloud, n_sectors=24, phi_step_deg=0.5):
    """
    改后逻辑：
    - 输入必须是已经 align_to_bmo_bfp() 后的 aligned_cloud
    - 所有 BMO 点已共面（en face）
    - 每个局部 MRA 单元对应一个真实 BMO 点
    - 底边宽度：环周相邻两个 BMO 点半段之和
    - 参考线：当前 B-scan 内左右两个 BMO 点的连线
    - phi 直接在该切片 2D 截面中搜索
    """
    print("\n🧮 启动改良 Gardiner-style BMO-MRA 计算引擎...")

    if aligned_cloud is None:
        print("  ❌ 错误: aligned_cloud 为 None")
        return [], 0.0

    bmo_meta = aligned_cloud.get('BMO_META', [])
    ilm_meta = aligned_cloud.get('ILM_META', [])

    if not bmo_meta or not ilm_meta:
        print("  ❌ 错误: BMO_META 或 ILM_META 为空")
        return [], 0.0

    bmo_dict = build_slice_bmo_dict(bmo_meta)
    ilm_dict = build_slice_ilm_segment_dict(ilm_meta)

    if not bmo_dict or not ilm_dict:
        print("  ❌ 错误: BMO 或 ILM 切片字典构建失败")
        return [], 0.0

    ordered_bmo24 = build_ordered_bmo24_from_meta(bmo_meta)
    if len(ordered_bmo24) < 6:
        print("  ❌ 错误: 有效 BMO 点不足，无法进行 MRA 计算")
        return [], 0.0

    if len(ordered_bmo24) != n_sectors:
        print(f"  ⚠️ 警告: 当前有效 BMO 点数 = {len(ordered_bmo24)}，与 n_sectors={n_sectors} 不一致。")
        print("     将按当前实际有效 BMO 点顺序进行局部 MRA 计算。")

    index_map = build_bmo24_index_map(ordered_bmo24)

    all_bmo = np.array(aligned_cloud['BMO'], dtype=float)
    center_xy = np.mean(all_bmo[:, :2], axis=0)

    gardiner_local_list = []
    global_mra_mm2 = 0.0

    for slice_id in sorted(bmo_dict.keys()):
        if slice_id not in ilm_dict:
            continue

        if 'L' not in bmo_dict[slice_id] or 'R' not in bmo_dict[slice_id]:
            continue

        # 当前 B-scan 内参考线：左右两个 BMO 点连线
        left_bmo_pt = np.array(bmo_dict[slice_id]['L']['point_3d'], dtype=float)
        right_bmo_pt = np.array(bmo_dict[slice_id]['R']['point_3d'], dtype=float)

        ref_dir_xy = right_bmo_pt[:2] - left_bmo_pt[:2]
        ref_norm = np.linalg.norm(ref_dir_xy)
        if ref_norm < 1e-12:
            continue
        ref_dir_xy = ref_dir_xy / ref_norm

        for side in ['L', 'R']:
            if side not in bmo_dict[slice_id] or side not in ilm_dict[slice_id]:
                continue

            # 关键修正：
            # 左侧朝 +ref_dir_xy（朝盘中心）
            # 右侧朝 -ref_dir_xy（朝盘中心）
            if side == 'L':
                side_ref_dir_xy = ref_dir_xy.copy()
            else:
                side_ref_dir_xy = -ref_dir_xy.copy()

            key = (slice_id, side)
            if key not in index_map:
                continue

            idx = index_map[key]

            bmo_pt = np.array(bmo_dict[slice_id][side]['point_3d'], dtype=float)
            ilm_polyline = np.array(ilm_dict[slice_id][side], dtype=float)

            if len(ilm_polyline) < 2:
                continue

            # 当前 BMO 在 en face 上的半径 r
            r_mm = np.linalg.norm(bmo_pt[:2] - center_xy)
            if r_mm < 1e-12:
                continue

            # 当前局部底边 = 环周前后相邻 BMO 点半段之和
            prev_pt = ordered_bmo24[(idx - 1) % len(ordered_bmo24)]['point_3d']
            curr_pt = ordered_bmo24[idx]['point_3d']
            next_pt = ordered_bmo24[(idx + 1) % len(ordered_bmo24)]['point_3d']

            bottom_len_mm = (
                    0.5 * np.linalg.norm(curr_pt[:2] - prev_pt[:2]) +
                    0.5 * np.linalg.norm(next_pt[:2] - curr_pt[:2])
            )

            if bottom_len_mm < 1e-12:
                continue

            best_area_mm2 = np.inf
            best_phi_deg = None
            best_rw_phi_mm = None
            best_top_len_mm = None
            best_hit_pt = None

            phi_values = np.arange(0.0, 90.0 + phi_step_deg, phi_step_deg)

            for phi_deg in phi_values:
                rw_phi_mm, hit_pt = intersect_ray_with_polyline_in_slice_2d(
                    bmo_pt, side_ref_dir_xy, ilm_polyline, phi_deg
                )
                if rw_phi_mm is None:
                    continue

                phi_rad = np.deg2rad(phi_deg)

                top_radius_mm = r_mm - rw_phi_mm * np.cos(phi_rad)
                if top_radius_mm <= 0:
                    continue

                top_len_mm = bottom_len_mm * (top_radius_mm / (r_mm + 1e-12))
                if top_len_mm <= 0:
                    continue

                area_mm2 = 0.5 * (bottom_len_mm + top_len_mm) * rw_phi_mm

                if area_mm2 < best_area_mm2:
                    best_area_mm2 = area_mm2
                    best_phi_deg = phi_deg
                    best_rw_phi_mm = rw_phi_mm
                    best_top_len_mm = top_len_mm
                    best_hit_pt = hit_pt

            if best_phi_deg is None or not np.isfinite(best_area_mm2):
                continue

            gardiner_local_list.append({
                'slice_id': slice_id,
                'side': side,
                'scan_angle_deg': float((slice_id - 1) * 15.0),
                'anatomical_angle_deg': float(anatomical_angle_deg(bmo_pt[0], bmo_pt[1])),
                'r_mm': float(r_mm),
                'bottom_len_mm': float(bottom_len_mm),
                'mra_phi_deg': float(best_phi_deg),
                'rw_phi_um': float(best_rw_phi_mm * 1000.0),
                'top_len_mm': float(best_top_len_mm),
                'local_area_mm2': float(best_area_mm2),
                'bmo_pt': tuple(bmo_pt.tolist()),
                'ilm_hit_pt': tuple(best_hit_pt.tolist()) if best_hit_pt is not None else None
            })

            global_mra_mm2 += best_area_mm2

    print(f"  ✅ 成功计算局部梯形数: {len(gardiner_local_list)}")
    print(f"  ✅ 全局改良 Gardiner-style BMO-MRA: {global_mra_mm2:.4f} mm²")

    return gardiner_local_list, global_mra_mm2


def build_ordered_boundary_from_slices(ali_meta):
    """
    针对径向扫描：
    图1~图12按逆时针每15°旋转一次，
    每张切片两个ALI端点分别属于圆环两侧。

    边界顺序采用：
    1L -> 2L -> ... -> 12L -> 1R -> 2R -> ... -> 12R -> 回到1L
    """
    if not ali_meta:
        return None

    slice_dict = {}
    for item in ali_meta:
        sid = item['slice_id']
        slice_dict.setdefault(sid, []).append(item)

    left_chain = []
    right_chain = []

    for sid in sorted(slice_dict.keys()):
        pts = slice_dict[sid]
        if len(pts) < 2:
            continue

        # 按原图横坐标区分左右端点
        pts_sorted = sorted(pts, key=lambda d: d['pixel_x'])
        left_pt = pts_sorted[0]['point_3d']
        right_pt = pts_sorted[-1]['point_3d']

        left_chain.append(left_pt[:2])
        right_chain.append(right_pt[:2])

    if len(left_chain) < 3 or len(right_chain) < 3:
        return None

    boundary_xy = np.array(left_chain + right_chain, dtype=float)
    return boundary_xy


def smooth_closed_boundary(boundary_xy, n_points=300, smooth_factor=0.001):
    """
    将已按正确顺序排列的闭合边界点，平滑成闭合样条曲线。
    boundary_xy: shape (N, 2)
    返回: shape (n_points, 2) 的平滑闭合边界
    """
    if boundary_xy is None or len(boundary_xy) < 3:
        return boundary_xy

    pts = np.array(boundary_xy, dtype=float)

    try:
        # 闭合处理：首点不重复传入，per=1 表示周期闭合
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smooth_factor, per=1)
        u_new = np.linspace(0, 1, n_points, endpoint=False)
        x_new, y_new = splev(u_new, tck)
        smoothed_xy = np.column_stack((x_new, y_new))
        return smoothed_xy
    except Exception as e:
        print(f"  ⚠️ ALI 边界样条平滑失败，降级使用原始折线: {e}")
        return boundary_xy


def build_ali_mask_from_boundary(boundary_xy, grid_x, grid_y):
    if boundary_xy is None or len(boundary_xy) < 3:
        return None

    ali_path = Path(boundary_xy)
    ali_mask = ali_path.contains_points(
        np.column_stack((grid_x.ravel(), grid_y.ravel()))
    ).reshape(grid_x.shape)

    return ali_mask


def gaussian_filter_nanaware_2d(grid_z, valid_mask, sigma=1.5):
    """
    只在 valid_mask 内进行 NaN-aware 高斯平滑。
    原理：
    1. 数值图中 NaN 置 0
    2. 同时对权重图（有效=1，无效=0）做同样高斯滤波
    3. 用 value_blur / weight_blur 得到只受有效区贡献的平滑结果
    """
    if grid_z is None or valid_mask is None:
        return grid_z

    value = np.where(valid_mask, grid_z, 0.0)
    weight = valid_mask.astype(float)

    value_blur = gaussian_filter(value, sigma=sigma)
    weight_blur = gaussian_filter(weight, sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        smoothed = value_blur / weight_blur

    smoothed[weight_blur < 1e-6] = np.nan
    smoothed[~valid_mask] = np.nan

    return smoothed


def calculate_3d_lcd_parameters(aligned_cloud, grid_resolution=200, interp_method='linear',
                                support_radius_mm=0.10, smooth_sigma=1.5,
                                show_debug_plot=False, debug_plot_path=None):
    print("\n🧵 启动 3D 曲面重构引擎：基于 ALCS 点云编织 LCD 曲面网格...")

    if aligned_cloud is None:
        print("  ❌ 错误: aligned_cloud 为 None")
        return None, None, None

    if 'ALCS' not in aligned_cloud or not aligned_cloud['ALCS']:
        print("  ❌ 错误: ALCS 点为空，无法进行曲面重构")
        return None, None, None

    if 'ALI' not in aligned_cloud or len(aligned_cloud['ALI']) < 3:
        print("  ❌ 错误: ALI 点不足，无法构建边界 mask")
        return None, None, None

    alcs_pts = np.array(aligned_cloud['ALCS'], dtype=float)

    if alcs_pts.shape[0] < 3:
        print("  ❌ 错误: ALCS 点数量不足，至少需要 3 个点进行插值")
        return None, None, None

    x = alcs_pts[:, 0]
    y = alcs_pts[:, 1]
    z = alcs_pts[:, 2]

    # 1. 建立规则网格
    xi = np.linspace(np.min(x), np.max(x), grid_resolution)
    yi = np.linspace(np.min(y), np.max(y), grid_resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # 2. 2D 插值生成初始曲面
    try:
        grid_z = griddata(
            points=(x, y),
            values=z,
            xi=(grid_x, grid_y),
            method=interp_method
        )
    except Exception as e:
        print(f"  ❌ 插值失败: {e}")
        return None, None, None

    # 3. 用 nearest 填补插值内部残留 NaN（仅为初始表面完整性）
    if np.isnan(grid_z).any():
        try:
            grid_z_nearest = griddata(
                points=(x, y),
                values=z,
                xi=(grid_x, grid_y),
                method='nearest'
            )
            grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
        except Exception as e:
            print(f"  ⚠️ nearest 填补 NaN 失败: {e}")

    # 4. 重建 ALI 边界
    ali_boundary_xy_raw = build_ordered_boundary_from_slices(aligned_cloud.get('ALI_META', []))
    if ali_boundary_xy_raw is None:
        print("  ❌ ALI 边界重建失败，无法构建 mask")
        return None, None, None

    ali_boundary_xy = smooth_closed_boundary(
        ali_boundary_xy_raw,
        n_points=300,
        smooth_factor=0.001
    )

    ali_mask = build_ali_mask_from_boundary(ali_boundary_xy, grid_x, grid_y)
    if ali_mask is None:
        print("  ❌ ALI mask 构建失败")
        return None, None, None

    # 5. 基于 ALCS 原始点到网格点最近距离，构建 support mask
    alcs_xy = np.column_stack((x, y))
    tree_xy = cKDTree(alcs_xy)

    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dist_xy, _ = tree_xy.query(grid_points, k=1)
    dist_xy = dist_xy.reshape(grid_x.shape)

    support_mask = dist_xy <= support_radius_mm

    # 6. 双 mask 交集：先定义最终有效区
    final_mask = ali_mask & support_mask

    # 7. 先裁剪，再平滑（关键改动）
    grid_z_masked = grid_z.copy()
    grid_z_masked[~final_mask] = np.nan

    print("  🪄 正在应用 mask 内 NaN-aware 高斯平滑...")
    grid_z_smoothed = gaussian_filter_nanaware_2d(
        grid_z_masked,
        valid_mask=final_mask,
        sigma=smooth_sigma
    )

    ali_count = np.sum(ali_mask)
    support_count = np.sum(support_mask)
    final_count = np.sum(final_mask)
    total_count = grid_z_masked.size

    print(f"  ✅ 3D 曲面网格生成完成 | 分辨率: {grid_resolution} x {grid_resolution}")
    print(f"  ✅ ALI mask 已应用")
    print(f"  ✅ ALCS support mask 已应用 | support_radius = {support_radius_mm:.2f} mm")
    print(f"  ✅ ALI 内点数: {ali_count}/{total_count}")
    print(f"  ✅ support 区点数: {support_count}/{total_count}")
    print(f"  ✅ 最终交集点数: {final_count}/{total_count}")

    # 调试可视化
    if show_debug_plot or debug_plot_path:
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(ali_mask, origin='lower', cmap='gray')
        plt.title('ALI Mask')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(support_mask, origin='lower', cmap='gray')
        plt.title(f'Support Mask (r <= {support_radius_mm:.2f} mm)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(final_mask, origin='lower', cmap='gray')
        plt.title('Final Mask = ALI ? Support')
        plt.axis('off')

        plt.tight_layout()
        if debug_plot_path:
            plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
        if show_debug_plot:
            plt.show()
        else:
            plt.close()


    return grid_x, grid_y, grid_z_smoothed


def order_mrw_segments_like_boundary(mrw_segments_list):
    """
    按和 ALI/BMO 一样的逻辑排序：
    1L -> 2L -> ... -> 12L -> 1R -> 2R -> ... -> 12R
    """
    if not mrw_segments_list:
        return []

    left_chain = [item for item in mrw_segments_list if item['side'] == 'L']
    right_chain = [item for item in mrw_segments_list if item['side'] == 'R']

    left_chain = sorted(left_chain, key=lambda d: d['slice_id'])
    right_chain = sorted(right_chain, key=lambda d: d['slice_id'])

    return left_chain + right_chain


def calculate_anatomical_sector_parameters(grid_x, grid_y, grid_z):
    print("\n🔪 启动真实解剖学 3D 空间分区引擎 (Garway-Heath 流行病学补偿版)")
    print("  👁️ 注: 底层数据已完成左眼镜像映射，当前统一按标准右眼拓扑进行切割 (0°=鼻侧水平, 187.6°=真实颞侧 FoBMO 轴)")

    if grid_x is None or grid_y is None or grid_z is None:
        print("  ❌ 错误: 输入网格为空，无法进行分区分析")
        return None

    angles_deg = anatomical_angle_deg(grid_x, grid_y)

    sectors_8 = {
        'NS (Nasal Superior / 鼻上侧)': [],
        'SN (Superior Nasal / 上鼻侧)': [],
        'ST (Superior Temporal / 上颞侧)': [],
        'TS (Temporal Superior / 颞上侧)': [],
        'TI (Temporal Inferior / 颞下侧)': [],
        'IT (Inferior Temporal / 下颞侧)': [],
        'IN (Inferior Nasal / 下鼻侧)': [],
        'NI (Nasal Inferior / 鼻下侧)': []
    }

    valid_mask = ~np.isnan(grid_z)

    for i in range(grid_z.shape[0]):
        for j in range(grid_z.shape[1]):
            if not valid_mask[i, j]:
                continue

            depth_um = grid_z[i, j] * 1000
            angle = angles_deg[i, j]

            if 0.0 <= angle < 63.5:
                sectors_8['NS (Nasal Superior / 鼻上侧)'].append(depth_um)
            elif 63.5 <= angle < 109.1:
                sectors_8['SN (Superior Nasal / 上鼻侧)'].append(depth_um)
            elif 109.1 <= angle < 144.7:
                sectors_8['ST (Superior Temporal / 上颞侧)'].append(depth_um)
            elif 144.7 <= angle < 187.6:
                sectors_8['TS (Temporal Superior / 颞上侧)'].append(depth_um)
            elif 187.6 <= angle < 224.1:
                sectors_8['TI (Temporal Inferior / 颞下侧)'].append(depth_um)
            elif 224.1 <= angle < 260.1:
                sectors_8['IT (Inferior Temporal / 下颞侧)'].append(depth_um)
            elif 260.1 <= angle < 310.1:
                sectors_8['IN (Inferior Nasal / 下鼻侧)'].append(depth_um)
            elif 310.1 <= angle <= 360.0:
                sectors_8['NI (Nasal Inferior / 鼻下侧)'].append(depth_um)

    sectors_4 = {
        'Nasal (鼻侧)': sectors_8['NS (Nasal Superior / 鼻上侧)'] + sectors_8['NI (Nasal Inferior / 鼻下侧)'],
        'Superior (上方)': sectors_8['SN (Superior Nasal / 上鼻侧)'] + sectors_8['ST (Superior Temporal / 上颞侧)'],
        'Temporal (颞侧)': sectors_8['TS (Temporal Superior / 颞上侧)'] + sectors_8['TI (Temporal Inferior / 颞下侧)'],
        'Inferior (下方)': sectors_8['IT (Inferior Temporal / 下颞侧)'] + sectors_8['IN (Inferior Nasal / 下鼻侧)']
    }

    sectors_2 = {
        'Superior_Half (上半部)': sectors_8['NS (Nasal Superior / 鼻上侧)'] +
                                  sectors_4['Superior (上方)'] +
                                  sectors_8['TS (Temporal Superior / 颞上侧)'],
        'Inferior_Half (下半部)': sectors_8['TI (Temporal Inferior / 颞下侧)'] +
                                  sectors_4['Inferior (下方)'] +
                                  sectors_8['NI (Nasal Inferior / 鼻下侧)']
    }

    print("\n📊 【八分区 (Garway-Heath 真实解剖学) 局部 3D-LCD 深度】")
    for k, v in sectors_8.items():
        print(f"   -> {k:<35}: {np.mean(v) if v else 0:.2f} μm (网格数: {len(v)})")

    print("\n📊 【四分区 (嵌套合并) 局部 3D-LCD 深度】")
    for k, v in sectors_4.items():
        print(f"   -> {k:<35}: {np.mean(v) if v else 0:.2f} μm (网格数: {len(v)})")

    print("\n📊 【二分区 (嵌套合并) 局部 3D-LCD 深度】")
    for k, v in sectors_2.items():
        print(f"   -> {k:<35}: {np.mean(v) if v else 0:.2f} μm (网格数: {len(v)})")

    return {
        'sectors_8': sectors_8,
        'sectors_4': sectors_4,
        'sectors_2': sectors_2
    }




def check_required_packages(excel_path=None):
    required = ['numpy', 'pandas', 'cv2', 'matplotlib', 'scipy', 'openpyxl']
    if excel_path and str(excel_path).lower().endswith('.xls'):
        required.append('xlrd')
    records = []
    ok = True
    for name in required:
        found = importlib.util.find_spec(name) is not None
        records.append({'Check': f'package:{name}', 'Status': 'PASS' if found else 'FAIL',
                        'Detail': 'installed' if found else 'missing'})
        ok = ok and found
    return ok, records


def check_input_paths(config):
    patient_folder = config['patient_folder']
    excel_path = config['excel_path']
    json_dir = config.get('json_dir') or os.path.join(patient_folder, 'JSONs')
    image_dir = config.get('image_dir')
    if not image_dir:
        image_pick = choose_image_dir(patient_folder, json_dir)
        image_dir = image_pick['image_dir']
    if not image_dir:
        image_dir = os.path.join(patient_folder, '增强')

    checks = [
        ('path:patient_folder', patient_folder),
        ('path:excel', excel_path),
        ('path:json_dir', json_dir),
        ('path:image_dir', image_dir),
    ]

    records = []
    ok = True
    for name, path in checks:
        exists = os.path.exists(path)
        records.append({'Check': name, 'Status': 'PASS' if exists else 'FAIL',
                        'Detail': path})
        ok = ok and exists
    return ok, records, {'json_dir': json_dir, 'image_dir': image_dir}


def check_patient_folder_id_match(config):
    patient_folder = config.get('patient_folder', '')
    patient_id = config.get('patient_id', '')
    folder_name = os.path.basename(os.path.normpath(patient_folder))
    folder_norm = normalize_identifier(folder_name)
    patient_norm = normalize_identifier(patient_id)
    is_match = bool(folder_norm) and bool(patient_norm) and folder_norm == patient_norm
    record = {
        'Check': 'patient_id:folder_match',
        'Status': 'PASS' if is_match else 'FAIL',
        'Detail': f"folder='{folder_name}', patient_id='{patient_id}'"
    }
    return is_match, [record]


def check_excel_columns(excel_path, patient_id):
    required_cols = ['Patient_ID', 'Axial_Length', 'Laterality']
    records = []
    ok = True
    try:
        df = pd.read_excel(excel_path)
        for col in required_cols:
            has_col = col in df.columns
            records.append({'Check': f'excel_col:{col}', 'Status': 'PASS' if has_col else 'FAIL',
                            'Detail': 'present' if has_col else 'missing'})
            ok = ok and has_col

        has_patient = ('Patient_ID' in df.columns) and (df['Patient_ID'] == patient_id).any()
        records.append({'Check': 'excel_patient_row', 'Status': 'PASS' if has_patient else 'FAIL',
                        'Detail': f'patient_id={patient_id}'})
        ok = ok and has_patient
    except Exception as e:
        ok = False
        records.append({'Check': 'excel_read', 'Status': 'FAIL', 'Detail': str(e)})
    return ok, records


def check_json_png_pairing(json_dir, image_dir):
    records = []
    ok = True

    json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    png_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))

    json_stems = {os.path.splitext(os.path.basename(p))[0] for p in json_files}
    png_stems = {os.path.splitext(os.path.basename(p))[0] for p in png_files}

    missing_png = sorted(json_stems - png_stems)
    extra_png = sorted(png_stems - json_stems)

    records.append({'Check': 'pairing:json_count', 'Status': 'PASS' if len(json_files) > 0 else 'FAIL',
                    'Detail': f'{len(json_files)} json'})
    records.append({'Check': 'pairing:png_count', 'Status': 'PASS' if len(png_files) > 0 else 'FAIL',
                    'Detail': f'{len(png_files)} png'})

    if missing_png:
        ok = False
    records.append({'Check': 'pairing:missing_png_for_json',
                    'Status': 'PASS' if not missing_png else 'FAIL',
                    'Detail': ','.join(missing_png[:10]) if missing_png else 'none'})

    if extra_png:
        records.append({'Check': 'pairing:extra_png_without_json',
                        'Status': 'WARN',
                        'Detail': ','.join(extra_png[:10])})
    else:
        records.append({'Check': 'pairing:extra_png_without_json',
                        'Status': 'PASS', 'Detail': 'none'})

    ok = ok and (len(json_files) > 0) and (len(png_files) > 0) and (len(missing_png) == 0)
    return ok, records


def check_label_presence(json_dir):
    required = ['BMO', 'ILM_ROI', 'ALI', 'ALCS', 'Center_ILM']
    counts = {k: 0 for k in required}

    json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    for pth in json_files:
        try:
            with open(pth, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                label = str(shape.get('label', ''))
                for key in required:
                    if key in label:
                        counts[key] += 1
        except Exception:
            pass

    records = []
    ok = True
    min_req = {'BMO': 2, 'ILM_ROI': 2, 'ALI': 2, 'ALCS': 3, 'Center_ILM': 1}
    for key in required:
        passed = counts[key] >= min_req[key]
        records.append({'Check': f'label:{key}', 'Status': 'PASS' if passed else 'FAIL',
                        'Detail': f"count={counts[key]}, min={min_req[key]}"})
        ok = ok and passed
    return ok, records


def check_output_writable(output_dir):
    records = []
    ok = True
    try:
        os.makedirs(output_dir, exist_ok=True)
        probe = os.path.join(output_dir, '__write_probe__.tmp')
        with open(probe, 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(probe)
        records.append({'Check': 'output_writable', 'Status': 'PASS', 'Detail': output_dir})
    except Exception as e:
        ok = False
        records.append({'Check': 'output_writable', 'Status': 'FAIL', 'Detail': str(e)})
    return ok, records


def startup_self_check(config):
    all_records = []

    ok_pkg, rec_pkg = check_required_packages(config.get('excel_path'))
    all_records.extend(rec_pkg)

    ok_path, rec_path, path_ctx = check_input_paths(config)
    all_records.extend(rec_path)

    ok_patient, rec_patient = check_patient_folder_id_match(config)
    all_records.extend(rec_patient)

    ok_excel = False
    ok_pair = False
    ok_label = False

    if ok_path:
        ok_excel, rec_excel = check_excel_columns(config['excel_path'], config['patient_id'])
        all_records.extend(rec_excel)

        ok_pair, rec_pair = check_json_png_pairing(path_ctx['json_dir'], path_ctx['image_dir'])
        all_records.extend(rec_pair)

        ok_label, rec_label = check_label_presence(path_ctx['json_dir'])
        all_records.extend(rec_label)

    ok_out, rec_out = check_output_writable(config['output_dir'])
    all_records.extend(rec_out)

    self_check_df = pd.DataFrame(all_records)
    overall_ok = ok_pkg and ok_path and ok_patient and ok_excel and ok_pair and ok_label and ok_out

    return overall_ok, self_check_df, path_ctx if ok_path else {}


def build_slice_ali_dict(ali_meta):
    out = {}
    slice_dict = {}
    for item in ali_meta:
        sid = item['slice_id']
        slice_dict.setdefault(sid, []).append(item)

    for sid, items in slice_dict.items():
        items_sorted = sorted(items, key=lambda d: d['pixel_x'])
        if len(items_sorted) >= 2:
            out[sid] = {
                'L': items_sorted[0],
                'R': items_sorted[-1]
            }
    return out


def build_slice_alcs_dict(alcs_meta):
    out = {}
    slice_dict = {}
    for item in alcs_meta:
        sid = item['slice_id']
        slice_dict.setdefault(sid, []).append(item)

    for sid, items in slice_dict.items():
        items_sorted = sorted(items, key=lambda d: d['pixel_x'])
        pts = np.array([it['point_3d'] for it in items_sorted], dtype=float)
        if len(pts) >= 3:
            out[sid] = pts
    return out


def build_slice_projection_context(left_bmo_pt, right_bmo_pt):
    left_bmo_pt = np.asarray(left_bmo_pt, dtype=float)
    right_bmo_pt = np.asarray(right_bmo_pt, dtype=float)

    ref_dir_xy = right_bmo_pt[:2] - left_bmo_pt[:2]
    ref_norm = np.linalg.norm(ref_dir_xy)
    if ref_norm < 1e-12:
        return None
    ref_dir_xy = ref_dir_xy / ref_norm

    right_2d = convert_points_to_slice_ref_2d(right_bmo_pt, left_bmo_pt, ref_dir_xy)
    if right_2d is None:
        return None
    right_2d = right_2d[0]

    chord_vec = right_2d - np.array([0.0, 0.0], dtype=float)
    D = np.linalg.norm(chord_vec)
    if D < 1e-12:
        return None

    ec = chord_vec / D
    ep = np.array([-ec[1], ec[0]], dtype=float)

    return {
        'origin_3d': left_bmo_pt,
        'ref_dir_xy': ref_dir_xy,
        'ec': ec,
        'ep': ep,
        'D': float(D),
    }


def project_points_with_slice_context(points_3d, ctx):
    pts = np.asarray(points_3d, dtype=float)
    single = False
    if pts.ndim == 1:
        pts = pts[None, :]
        single = True

    pts2d = convert_points_to_slice_ref_2d(pts, ctx['origin_3d'], ctx['ref_dir_xy'])
    if pts2d is None:
        return None

    rel = pts2d
    u = rel @ ctx['ec']
    v = rel @ ctx['ep']
    uv = np.column_stack([u, v])

    return uv[0] if single else uv


def clear_formal_output_dir(output_dir):
    if not os.path.isdir(output_dir):
        return
    for entry in os.scandir(output_dir):
        path = entry.path
        if entry.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _radial_unit_for_slice(theta_deg, laterality):
    theta = np.deg2rad(theta_deg)
    if str(laterality).upper() == 'L':
        return np.array([-np.cos(theta), np.sin(theta)], dtype=float)
    return np.array([np.cos(theta), np.sin(theta)], dtype=float)


def original_3d_to_image_px(point_3d, slice_meta, laterality):
    if slice_meta is None:
        return None

    pt = np.asarray(point_3d, dtype=float)
    scale_x = float(slice_meta.get('scale_X', np.nan))
    scale_z = float(slice_meta.get('scale_Z', np.nan))
    x_center = float(slice_meta.get('x_center', np.nan))
    delta_z = float(slice_meta.get('delta_z', 0.0))
    theta = float(slice_meta.get('angle_deg', np.nan))

    if not np.isfinite(scale_x) or abs(scale_x) < 1e-12:
        return None
    if not np.isfinite(scale_z) or abs(scale_z) < 1e-12:
        return None
    if not np.isfinite(x_center) or not np.isfinite(theta):
        return None

    radial = _radial_unit_for_slice(theta, laterality)
    r_mm = float(np.dot(pt[:2], radial))
    px = r_mm / scale_x + x_center
    py = (float(pt[2]) + delta_z) / scale_z
    return np.array([px, py], dtype=float)


def aligned_3d_to_original_3d(point_3d, alignment):
    if alignment is None:
        return None
    if 'rotation_matrix' not in alignment or 'centroid' not in alignment:
        return None

    pt = np.asarray(point_3d, dtype=float)
    rot = R.from_matrix(np.asarray(alignment['rotation_matrix'], dtype=float))
    cen = np.asarray(alignment['centroid'], dtype=float)
    return rot.inv().apply(pt) + cen


def aligned_3d_to_image_px(point_3d, slice_id, aligned_cloud):
    sl = aligned_cloud.get('SLICE_META', {}).get(int(slice_id))
    if sl is None:
        return None
    orig_pt = aligned_3d_to_original_3d(point_3d, aligned_cloud.get('ALIGNMENT'))
    if orig_pt is None:
        return None
    return original_3d_to_image_px(orig_pt, sl, aligned_cloud.get('laterality', 'R'))


def build_uv_to_aligned_xyz_coef(uv_pts, xyz_pts):
    uv = np.asarray(uv_pts, dtype=float)
    xyz = np.asarray(xyz_pts, dtype=float)
    if uv.ndim != 2 or xyz.ndim != 2 or uv.shape[0] < 3 or xyz.shape[0] != uv.shape[0]:
        return None

    A = np.column_stack([uv[:, 0], uv[:, 1], np.ones(uv.shape[0], dtype=float)])
    try:
        coef, *_ = np.linalg.lstsq(A, xyz, rcond=None)
    except Exception:
        return None
    return coef  # shape (3,3), [u,v,1] @ coef -> [x,y,z]


def uv_to_aligned_3d(uv_pts, coef):
    if coef is None:
        return None
    uv = np.asarray(uv_pts, dtype=float)
    single = False
    if uv.ndim == 1:
        uv = uv[None, :]
        single = True
    A = np.column_stack([uv[:, 0], uv[:, 1], np.ones(uv.shape[0], dtype=float)])
    out = A @ coef
    return out[0] if single else out


def fit_alcs_arc_prefer_circle(points_2d):
    pts = np.asarray(points_2d, dtype=float)
    if pts.shape[0] < 3:
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)

    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None

    a, b_, c = coef
    cx = -a / 2.0
    cy = -b_ / 2.0
    r2 = cx * cx + cy * cy - c

    if not np.isfinite(r2) or r2 <= 1e-12:
        return None

    r = float(np.sqrt(r2))

    term = r2 - (x - cx) ** 2
    valid = term >= 0
    if np.sum(valid) < max(3, int(0.6 * len(x))):
        return None

    y1 = np.full_like(y, np.nan)
    y2 = np.full_like(y, np.nan)
    y1[valid] = cy + np.sqrt(np.maximum(term[valid], 0.0))
    y2[valid] = cy - np.sqrt(np.maximum(term[valid], 0.0))

    err1 = np.nanmean((y1[valid] - y[valid]) ** 2)
    err2 = np.nanmean((y2[valid] - y[valid]) ** 2)
    branch = 1.0 if err1 <= err2 else -1.0

    y_fit_obs = y1 if branch > 0 else y2
    rmse = float(np.sqrt(np.nanmean((y_fit_obs[valid] - y[valid]) ** 2)))

    if not np.isfinite(rmse):
        return None

    y_std = float(np.nanstd(y))
    if y_std > 1e-9 and rmse > max(0.20, 3.0 * y_std):
        return None

    def eval_y(xq):
        xq = np.asarray(xq, dtype=float)
        tq = np.maximum(r2 - (xq - cx) ** 2, 0.0)
        return cy + branch * np.sqrt(tq)

    return {
        'method': 'circle',
        'rmse': rmse,
        'radius': r,
        'center': (float(cx), float(cy)),
        'eval_y': eval_y,
    }


def fit_alcs_quadratic_fallback(points_2d):
    pts = np.asarray(points_2d, dtype=float)
    if pts.shape[0] < 3:
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    try:
        coef = np.polyfit(x, y, 2)
    except Exception:
        return None

    poly = np.poly1d(coef)
    y_pred = poly(x)
    rmse = float(np.sqrt(np.mean((y_pred - y) ** 2)))

    def eval_y(xq):
        return poly(np.asarray(xq, dtype=float))

    return {
        'method': 'quadratic',
        'rmse': rmse,
        'coef': coef,
        'eval_y': eval_y,
    }


def compute_slice_traditional_lcd_lcci(slice_id, bmo_lr, ali_lr, alcs_pts):
    row = {
        'slice_id': int(slice_id),
        'scan_angle_deg': float((int(slice_id) - 1) * 15.0),
        'fit_method': 'none',
        'fit_status': 'fit_failed',
        'fit_quality': np.nan,
        'Length_D_mm': np.nan,
        'Area_S_mm2': np.nan,
        'LCD_area_mm': np.nan,
        'LCD_direct_mm': np.nan,
        'ALID1_mm': np.nan,
        'ALID2_mm': np.nan,
        'mALID_mm': np.nan,
        'LCCI_area_mm': np.nan,
        'LCCI_direct_mm': np.nan,
        'aLCCI_area_pct': np.nan,
        'aLCCI_direct_pct': np.nan,
        'status': 'FAIL',
        'reason': '',
    }

    left_bmo = np.array(bmo_lr['L']['point_3d'], dtype=float)
    right_bmo = np.array(bmo_lr['R']['point_3d'], dtype=float)

    ctx = build_slice_projection_context(left_bmo, right_bmo)
    if ctx is None:
        row['reason'] = 'invalid BMO chord'
        return row, None

    D = ctx['D']
    if D <= 1e-12:
        row['reason'] = 'zero chord length'
        return row, None

    alcs_uv = project_points_with_slice_context(alcs_pts, ctx)
    if alcs_uv is None or len(alcs_uv) < 3:
        row['reason'] = 'insufficient ALCS points'
        return row, None

    x_raw = alcs_uv[:, 0]
    keep = (x_raw >= -0.10 * D) & (x_raw <= 1.10 * D)
    if np.sum(keep) >= 3:
        alcs_uv_fit = alcs_uv[keep]
    else:
        alcs_uv_fit = alcs_uv

    fit = fit_alcs_arc_prefer_circle(alcs_uv_fit)
    if fit is None:
        fit = fit_alcs_quadratic_fallback(alcs_uv_fit)

    if fit is None:
        row['reason'] = 'fit failed'
        return row, None

    fit_method = fit['method']
    fit_quality = float(fit['rmse'])
    fit_status = 'circle_ok' if fit_method == 'circle' else 'quadratic_fallback'

    x_fit = np.linspace(0.0, D, 300)
    y_fit = fit['eval_y'](x_fit)

    if np.any(~np.isfinite(y_fit)):
        row['reason'] = 'fit produced invalid values'
        return row, None

    uv_fit = np.column_stack([x_fit, y_fit])
    uv_to_xyz_coef = build_uv_to_aligned_xyz_coef(alcs_uv, alcs_pts)

    area_s = float(np.trapezoid(np.abs(y_fit), x_fit))
    lcd_area = area_s / D

    idx_deep = int(np.argmax(np.abs(y_fit)))
    x_deep = float(x_fit[idx_deep])
    y_deep = float(y_fit[idx_deep])
    lcd_direct = float(abs(y_deep))

    left_ali = np.array(ali_lr['L']['point_3d'], dtype=float)
    right_ali = np.array(ali_lr['R']['point_3d'], dtype=float)
    ali_uv = project_points_with_slice_context(np.vstack([left_ali, right_ali]), ctx)
    if ali_uv is None:
        row['reason'] = 'ALI projection failed'
        return row, None

    ALID1 = float(abs(ali_uv[0, 1]))
    ALID2 = float(abs(ali_uv[1, 1]))
    mALID = 0.5 * (ALID1 + ALID2)

    LCCI_area = lcd_area - mALID
    LCCI_direct = lcd_direct - mALID
    aLCCI_area = (LCCI_area / D) * 100.0
    aLCCI_direct = (LCCI_direct / D) * 100.0

    row.update({
        'fit_method': fit_method,
        'fit_status': fit_status,
        'fit_quality': fit_quality,
        'Length_D_mm': float(D),
        'Area_S_mm2': area_s,
        'LCD_area_mm': float(lcd_area),
        'LCD_direct_mm': float(lcd_direct),
        'ALID1_mm': ALID1,
        'ALID2_mm': ALID2,
        'mALID_mm': float(mALID),
        'LCCI_area_mm': float(LCCI_area),
        'LCCI_direct_mm': float(LCCI_direct),
        'aLCCI_area_pct': float(aLCCI_area),
        'aLCCI_direct_pct': float(aLCCI_direct),
        'status': 'PASS',
        'reason': '',
    })

    plot_payload = {
        'slice_id': int(slice_id),
        'ctx': ctx,
        'alcs_uv': alcs_uv,
        'alcs_xyz': np.asarray(alcs_pts, dtype=float),
        'fit_x': x_fit,
        'fit_y': y_fit,
        'fit_uv': uv_fit,
        'uv_to_xyz_coef': uv_to_xyz_coef,
        'bmo_uv': np.array([[0.0, 0.0], [D, 0.0]], dtype=float),
        'ali_uv': ali_uv,
        'bmo_lr_xyz': np.vstack([left_bmo, right_bmo]),
        'ali_lr_xyz': np.vstack([left_ali, right_ali]),
        'deepest_uv': np.array([x_deep, y_deep], dtype=float),
        'lcd_line_uv': np.array([[x_deep, 0.0], [x_deep, y_deep]], dtype=float),
        'alid_lines_uv': np.array([
            [[ali_uv[0, 0], 0.0], [ali_uv[0, 0], ali_uv[0, 1]]],
            [[ali_uv[1, 0], 0.0], [ali_uv[1, 0], ali_uv[1, 1]]],
        ], dtype=float),
        'deepest_xyz': uv_to_aligned_3d(np.array([x_deep, y_deep], dtype=float), uv_to_xyz_coef)
        if uv_to_xyz_coef is not None else None,
        'fit_method': fit_method,
        'fit_status': fit_status,
        'fit_quality': fit_quality,
    }

    return row, plot_payload


def compute_traditional_lcd_lcci_all_slices(aligned_cloud):
    bmo_meta = aligned_cloud.get('BMO_META', [])
    ali_meta = aligned_cloud.get('ALI_META', [])
    alcs_meta = aligned_cloud.get('ALCS_META', [])

    bmo_dict = build_slice_bmo_dict(bmo_meta)
    ali_dict = build_slice_ali_dict(ali_meta)
    alcs_dict = build_slice_alcs_dict(alcs_meta)

    slice_ids = sorted(set(bmo_dict.keys()) | set(ali_dict.keys()) | set(alcs_dict.keys()))

    rows = []
    payload_map = {}

    for sid in slice_ids:
        if sid not in bmo_dict:
            rows.append({'slice_id': int(sid), 'scan_angle_deg': float((int(sid) - 1) * 15.0),
                         'status': 'FAIL', 'reason': 'missing BMO',
                         'fit_method': 'none', 'fit_quality': np.nan,
                         'fit_status': 'fit_failed',
                         'Length_D_mm': np.nan, 'Area_S_mm2': np.nan,
                         'LCD_area_mm': np.nan, 'LCD_direct_mm': np.nan,
                         'ALID1_mm': np.nan, 'ALID2_mm': np.nan, 'mALID_mm': np.nan,
                         'LCCI_area_mm': np.nan, 'LCCI_direct_mm': np.nan,
                         'aLCCI_area_pct': np.nan, 'aLCCI_direct_pct': np.nan})
            continue
        if sid not in ali_dict:
            rows.append({'slice_id': int(sid), 'scan_angle_deg': float((int(sid) - 1) * 15.0),
                         'status': 'FAIL', 'reason': 'missing ALI',
                         'fit_method': 'none', 'fit_quality': np.nan,
                         'fit_status': 'fit_failed',
                         'Length_D_mm': np.nan, 'Area_S_mm2': np.nan,
                         'LCD_area_mm': np.nan, 'LCD_direct_mm': np.nan,
                         'ALID1_mm': np.nan, 'ALID2_mm': np.nan, 'mALID_mm': np.nan,
                         'LCCI_area_mm': np.nan, 'LCCI_direct_mm': np.nan,
                         'aLCCI_area_pct': np.nan, 'aLCCI_direct_pct': np.nan})
            continue
        if sid not in alcs_dict:
            rows.append({'slice_id': int(sid), 'scan_angle_deg': float((int(sid) - 1) * 15.0),
                         'status': 'FAIL', 'reason': 'missing ALCS',
                         'fit_method': 'none', 'fit_quality': np.nan,
                         'fit_status': 'fit_failed',
                         'Length_D_mm': np.nan, 'Area_S_mm2': np.nan,
                         'LCD_area_mm': np.nan, 'LCD_direct_mm': np.nan,
                         'ALID1_mm': np.nan, 'ALID2_mm': np.nan, 'mALID_mm': np.nan,
                         'LCCI_area_mm': np.nan, 'LCCI_direct_mm': np.nan,
                         'aLCCI_area_pct': np.nan, 'aLCCI_direct_pct': np.nan})
            continue

        row, payload = compute_slice_traditional_lcd_lcci(
            sid,
            bmo_dict[sid],
            ali_dict[sid],
            alcs_dict[sid]
        )
        rows.append(row)
        if payload is not None:
            payload_map[int(sid)] = payload

    df = pd.DataFrame(rows).sort_values('slice_id').reset_index(drop=True)
    return df, payload_map


def _imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or '.png'
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def _as_int_point(pt, w=None, h=None):
    if pt is None:
        return None
    arr = np.asarray(pt, dtype=float).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None
    x = int(np.rint(arr[0]))
    y = int(np.rint(arr[1]))
    if w is not None and h is not None:
        x = int(np.clip(x, 0, max(w - 1, 0)))
        y = int(np.clip(y, 0, max(h - 1, 0)))
    return (x, y)


def _draw_line_if_valid(img, p0, p1, color, thickness=2):
    h, w = img.shape[:2]
    q0 = _as_int_point(p0, w, h)
    q1 = _as_int_point(p1, w, h)
    if q0 is not None and q1 is not None:
        cv2.line(img, q0, q1, color, thickness, lineType=cv2.LINE_AA)


def _draw_points_if_valid(img, pts, color, radius=3, thickness=-1):
    h, w = img.shape[:2]
    for p in pts:
        q = _as_int_point(p, w, h)
        if q is not None:
            cv2.circle(img, q, radius, color, thickness, lineType=cv2.LINE_AA)


def _draw_polyline_if_valid(img, pts, color, thickness=2):
    if pts is None:
        return
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return
    h, w = img.shape[:2]
    int_pts = []
    for p in arr:
        q = _as_int_point(p, w, h)
        if q is not None:
            int_pts.append(q)
    if len(int_pts) >= 2:
        cv2.polylines(img, [np.array(int_pts, dtype=np.int32)], False, color, thickness, lineType=cv2.LINE_AA)


def save_slice_qc_figures(payload_map, mrw_segments_list, gardiner_local_list, output_dir, final_cloud, aligned_cloud):
    os.makedirs(output_dir, exist_ok=True)

    slice_meta = final_cloud.get('SLICE_META', {})
    laterality = final_cloud.get('laterality', 'R')

    bmo_raw = build_slice_bmo_dict(final_cloud.get('BMO_META', []))
    ali_raw = build_slice_ali_dict(final_cloud.get('ALI_META', []))

    alcs_raw = {}
    for it in final_cloud.get('ALCS_META', []):
        alcs_raw.setdefault(int(it['slice_id']), []).append((it.get('pixel_x', np.nan), it.get('pixel_y', np.nan)))
    ilm_raw = {}
    for it in final_cloud.get('ILM_META', []):
        ilm_raw.setdefault(int(it['slice_id']), []).append((it.get('pixel_x', np.nan), it.get('pixel_y', np.nan)))

    mrw_by_slice = {}
    for item in mrw_segments_list:
        mrw_by_slice.setdefault(int(item['slice_id']), []).append(item)

    mra_by_slice = {}
    for item in gardiner_local_list:
        mra_by_slice.setdefault(int(item['slice_id']), []).append(item)

    all_sids = sorted(set(slice_meta.keys()) | set(payload_map.keys()) | set(mrw_by_slice.keys()) | set(mra_by_slice.keys()))

    for sid in all_sids:
        sm = slice_meta.get(int(sid))
        if sm is None:
            continue
        image_path = sm.get('image_path')
        if not image_path or not os.path.exists(image_path):
            continue

        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Raw annotation overlays (direct pixel coordinates from JSON when available).
        _draw_points_if_valid(img, ilm_raw.get(int(sid), []), color=(180, 180, 0), radius=1, thickness=-1)
        _draw_points_if_valid(img, alcs_raw.get(int(sid), []), color=(0, 255, 0), radius=2, thickness=-1)

        bmo_sid = bmo_raw.get(int(sid), {})
        if 'L' in bmo_sid and 'R' in bmo_sid:
            lpt = (bmo_sid['L'].get('pixel_x', np.nan), bmo_sid['L'].get('pixel_y', np.nan))
            rpt = (bmo_sid['R'].get('pixel_x', np.nan), bmo_sid['R'].get('pixel_y', np.nan))
            _draw_points_if_valid(img, [lpt, rpt], color=(0, 255, 255), radius=4, thickness=-1)
            _draw_line_if_valid(img, lpt, rpt, color=(0, 255, 255), thickness=2)

        ali_sid = ali_raw.get(int(sid), {})
        if 'L' in ali_sid and 'R' in ali_sid:
            lpt = (ali_sid['L'].get('pixel_x', np.nan), ali_sid['L'].get('pixel_y', np.nan))
            rpt = (ali_sid['R'].get('pixel_x', np.nan), ali_sid['R'].get('pixel_y', np.nan))
            _draw_points_if_valid(img, [lpt, rpt], color=(255, 120, 0), radius=4, thickness=-1)

        p = payload_map.get(int(sid))
        if p is not None:
            coef = p.get('uv_to_xyz_coef')

            # Fitted ALCS arc (derived geometry -> inverse projection).
            fit_uv = p.get('fit_uv')
            fit_xyz = uv_to_aligned_3d(fit_uv, coef) if fit_uv is not None and coef is not None else None
            fit_px = []
            if fit_xyz is not None:
                for pt in fit_xyz:
                    px = aligned_3d_to_image_px(pt, sid, aligned_cloud)
                    if px is not None:
                        fit_px.append(px)
            _draw_polyline_if_valid(img, fit_px, color=(0, 140, 255), thickness=2)

            # Deepest point and LCD_direct line.
            deep_xyz = p.get('deepest_xyz')
            deep_px = aligned_3d_to_image_px(deep_xyz, sid, aligned_cloud) if deep_xyz is not None else None
            if deep_px is not None:
                _draw_points_if_valid(img, [deep_px], color=(0, 0, 255), radius=5, thickness=2)

            lcd_uv = p.get('lcd_line_uv')
            lcd_xyz = uv_to_aligned_3d(lcd_uv.reshape(-1, 2), coef) if lcd_uv is not None and coef is not None else None
            if lcd_xyz is not None and len(lcd_xyz) >= 2:
                p0 = aligned_3d_to_image_px(lcd_xyz[0], sid, aligned_cloud)
                p1 = aligned_3d_to_image_px(lcd_xyz[1], sid, aligned_cloud)
                _draw_line_if_valid(img, p0, p1, color=(0, 0, 255), thickness=2)

            # ALID projected lines.
            alid_uv = p.get('alid_lines_uv')
            if alid_uv is not None and coef is not None:
                for seg in np.asarray(alid_uv, dtype=float):
                    seg_xyz = uv_to_aligned_3d(seg.reshape(-1, 2), coef)
                    if seg_xyz is None or len(seg_xyz) < 2:
                        continue
                    p0 = aligned_3d_to_image_px(seg_xyz[0], sid, aligned_cloud)
                    p1 = aligned_3d_to_image_px(seg_xyz[1], sid, aligned_cloud)
                    _draw_line_if_valid(img, p0, p1, color=(255, 120, 0), thickness=1)

        # MRW lines: BMO endpoint from original label px when possible, ILM hit is derived.
        for m in mrw_by_slice.get(int(sid), []):
            side = m.get('side')
            bmo_px = None
            if side in bmo_sid:
                bmo_px = np.array([bmo_sid[side].get('pixel_x', np.nan), bmo_sid[side].get('pixel_y', np.nan)], dtype=float)
            if bmo_px is None or not np.all(np.isfinite(bmo_px)):
                bmo_px = original_3d_to_image_px(np.array(m['bmo_pt'], dtype=float), sm, laterality)
            ilm_px = original_3d_to_image_px(np.array(m['ilm_pt'], dtype=float), sm, laterality)
            _draw_line_if_valid(img, bmo_px, ilm_px, color=(190, 0, 255), thickness=2)

        # MRA final selected optimal ray: BMO label px -> derived ILM hit px.
        for g in mra_by_slice.get(int(sid), []):
            if g.get('ilm_hit_pt') is None:
                continue
            side = g.get('side')
            bmo_px = None
            if side in bmo_sid:
                bmo_px = np.array([bmo_sid[side].get('pixel_x', np.nan), bmo_sid[side].get('pixel_y', np.nan)], dtype=float)
            if bmo_px is None or not np.all(np.isfinite(bmo_px)):
                bmo_px = aligned_3d_to_image_px(np.array(g['bmo_pt'], dtype=float), sid, aligned_cloud)
            hit_px = aligned_3d_to_image_px(np.array(g['ilm_hit_pt'], dtype=float), sid, aligned_cloud)
            _draw_line_if_valid(img, bmo_px, hit_px, color=(60, 30, 200), thickness=1)

        fit_status = p.get('fit_status', 'na') if p is not None else 'na'
        cv2.putText(img, f"slice {int(sid):02d}  fit:{fit_status}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)

        out_path = os.path.join(output_dir, f'slice_{int(sid):02d}_qc.png')
        _imwrite_unicode(out_path, img)


def save_lcd_lcci_summary_plot(lcd_lcci_df, out_path):
    if lcd_lcci_df is None or len(lcd_lcci_df) == 0:
        return

    df = lcd_lcci_df.copy()
    if 'slice_id' not in df.columns:
        return

    metrics = [
        (['lcd_area_mm', 'LCD_area_mm'], 'LCD_area (mm)'),
        (['lcd_direct_mm', 'LCD_direct_mm'], 'LCD_direct (mm)'),
        (['lcci_area_mm', 'LCCI_area_mm'], 'LCCI_area (mm)'),
        (['lcci_direct_mm', 'LCCI_direct_mm'], 'LCCI_direct (mm)'),
        (['alcci_area', 'aLCCI_area_pct'], 'aLCCI_area (%)'),
        (['alcci_direct', 'aLCCI_direct_pct'], 'aLCCI_direct (%)'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, (candidates, title) in zip(axes, metrics):
        col = next((c for c in candidates if c in df.columns), None)
        y = df[col] if col else np.nan
        ax.plot(df['slice_id'], y, marker='o', linewidth=1.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    for ax in axes[-2:]:
        ax.set_xlabel('Slice ID')

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


SECTOR_8_BOUNDS = [
    (0.0, 63.5, 'NS (Nasal Superior / 鼻上侧)'),
    (63.5, 109.1, 'SN (Superior Nasal / 上鼻侧)'),
    (109.1, 144.7, 'ST (Superior Temporal / 上颞侧)'),
    (144.7, 187.6, 'TS (Temporal Superior / 颞上侧)'),
    (187.6, 224.1, 'TI (Temporal Inferior / 颞下侧)'),
    (224.1, 260.1, 'IT (Inferior Temporal / 下颞侧)'),
    (260.1, 310.1, 'IN (Inferior Nasal / 下鼻侧)'),
    (310.1, 360.0, 'NI (Nasal Inferior / 鼻下侧)'),
]

SECTOR_4_MAP = {
    'NS (Nasal Superior / 鼻上侧)': 'Nasal (Nasal / 鼻侧)',
    'NI (Nasal Inferior / 鼻下侧)': 'Nasal (Nasal / 鼻侧)',
    'SN (Superior Nasal / 上鼻侧)': 'Superior (Superior / 上方)',
    'ST (Superior Temporal / 上颞侧)': 'Superior (Superior / 上方)',
    'TS (Temporal Superior / 颞上侧)': 'Temporal (Temporal / 颞侧)',
    'TI (Temporal Inferior / 颞下侧)': 'Temporal (Temporal / 颞侧)',
    'IT (Inferior Temporal / 下颞侧)': 'Inferior (Inferior / 下方)',
    'IN (Inferior Nasal / 下鼻侧)': 'Inferior (Inferior / 下方)',
}

SECTOR_2_MAP = {
    'NS (Nasal Superior / 鼻上侧)': 'Superior Half (Superior Half / 上半部)',
    'SN (Superior Nasal / 上鼻侧)': 'Superior Half (Superior Half / 上半部)',
    'ST (Superior Temporal / 上颞侧)': 'Superior Half (Superior Half / 上半部)',
    'TS (Temporal Superior / 颞上侧)': 'Superior Half (Superior Half / 上半部)',
    'TI (Temporal Inferior / 颞下侧)': 'Inferior Half (Inferior Half / 下半部)',
    'IT (Inferior Temporal / 下颞侧)': 'Inferior Half (Inferior Half / 下半部)',
    'IN (Inferior Nasal / 下鼻侧)': 'Inferior Half (Inferior Half / 下半部)',
    'NI (Nasal Inferior / 鼻下侧)': 'Inferior Half (Inferior Half / 下半部)',
}


def sector_names_from_angle(angle_deg):
    if angle_deg is None or not np.isfinite(angle_deg):
        return None, None, None

    a = float(angle_deg) % 360.0
    if np.isclose(a, 360.0):
        a = 0.0

    sector_8 = None
    for idx, (lo, hi, name) in enumerate(SECTOR_8_BOUNDS):
        if idx == len(SECTOR_8_BOUNDS) - 1:
            if lo <= a <= hi:
                sector_8 = name
                break
        elif lo <= a < hi:
            sector_8 = name
            break

    if sector_8 is None:
        return None, None, None

    return sector_8, SECTOR_4_MAP.get(sector_8), SECTOR_2_MAP.get(sector_8)


def attach_sector_labels(df, angle_col):
    out = df.copy()
    s8, s4, s2 = [], [], []
    for v in out.get(angle_col, pd.Series(dtype=float)):
        a8, a4, a2 = sector_names_from_angle(v)
        s8.append(a8)
        s4.append(a4)
        s2.append(a2)
    out['sector_8_name'] = s8
    out['sector_4_name'] = s4
    out['sector_2_name'] = s2
    return out


def prepare_mrw_dataframe(ordered_mrw_segments, final_cloud):
    if not ordered_mrw_segments:
        cols = [
            'slice_id', 'side', 'scan_angle_deg', 'anatomical_angle_deg',
            'mrw_len_um', 'mrw_angle_deg',
            'bmo_x_px', 'bmo_y_px', 'ilm_x_px', 'ilm_y_px',
            'sector_8_name', 'sector_4_name', 'sector_2_name'
        ]
        return pd.DataFrame(columns=cols)

    bmo_dict = build_slice_bmo_dict(final_cloud.get('BMO_META', []))
    slice_meta = final_cloud.get('SLICE_META', {})
    laterality = final_cloud.get('laterality', 'R')

    rows = []
    for item in ordered_mrw_segments:
        sid = int(item['slice_id'])
        side = item.get('side')
        scan_angle = float((sid - 1) * 15.0)
        anat = float(item['angle']) if np.isfinite(item.get('angle', np.nan)) else np.nan
        sec8, sec4, sec2 = sector_names_from_angle(anat)

        bmo_x_px, bmo_y_px = np.nan, np.nan
        bmo_item = bmo_dict.get(sid, {}).get(side)
        if bmo_item is not None:
            bmo_x_px = float(bmo_item.get('pixel_x', np.nan))
            bmo_y_px = float(bmo_item.get('pixel_y', np.nan))
        else:
            bmo_px = original_3d_to_image_px(np.array(item['bmo_pt'], dtype=float), slice_meta.get(sid), laterality)
            if bmo_px is not None:
                bmo_x_px, bmo_y_px = float(bmo_px[0]), float(bmo_px[1])

        ilm_px = original_3d_to_image_px(np.array(item['ilm_pt'], dtype=float), slice_meta.get(sid), laterality)
        ilm_x_px = float(ilm_px[0]) if ilm_px is not None else np.nan
        ilm_y_px = float(ilm_px[1]) if ilm_px is not None else np.nan

        rows.append({
            'slice_id': sid,
            'side': side,
            'scan_angle_deg': scan_angle,
            'anatomical_angle_deg': anat,
            'mrw_len_um': float(item.get('mrw_len', np.nan)),
            'mrw_angle_deg': float(item.get('mrw_angle_deg', np.nan)),
            'bmo_x_px': bmo_x_px,
            'bmo_y_px': bmo_y_px,
            'ilm_x_px': ilm_x_px,
            'ilm_y_px': ilm_y_px,
            'sector_8_name': sec8,
            'sector_4_name': sec4,
            'sector_2_name': sec2,
        })

    df = pd.DataFrame(rows)
    cols = [
        'slice_id', 'side', 'scan_angle_deg', 'anatomical_angle_deg',
        'mrw_len_um', 'mrw_angle_deg',
        'bmo_x_px', 'bmo_y_px', 'ilm_x_px', 'ilm_y_px',
        'sector_8_name', 'sector_4_name', 'sector_2_name'
    ]
    return df[cols]


def prepare_mra_dataframe(gardiner_local_list, final_cloud, aligned_cloud):
    if not gardiner_local_list:
        cols = [
            'slice_id', 'side', 'scan_angle_deg', 'anatomical_angle_deg', 'r_mm',
            'bottom_len_mm', 'mra_phi_deg', 'rw_phi_um', 'top_len_mm', 'local_area_mm2',
            'bmo_x_px', 'bmo_y_px', 'ilm_hit_x_px', 'ilm_hit_y_px',
            'sector_8_name', 'sector_4_name', 'sector_2_name'
        ]
        return pd.DataFrame(columns=cols)

    bmo_dict = build_slice_bmo_dict(final_cloud.get('BMO_META', []))
    rows = []
    for item in gardiner_local_list:
        sid = int(item.get('slice_id'))
        side = item.get('side')
        anat = float(item.get('anatomical_angle_deg', np.nan))
        sec8, sec4, sec2 = sector_names_from_angle(anat)

        bmo_x_px, bmo_y_px = np.nan, np.nan
        bmo_item = bmo_dict.get(sid, {}).get(side)
        if bmo_item is not None:
            bmo_x_px = float(bmo_item.get('pixel_x', np.nan))
            bmo_y_px = float(bmo_item.get('pixel_y', np.nan))
        else:
            bmo_px = aligned_3d_to_image_px(np.array(item['bmo_pt'], dtype=float), sid, aligned_cloud)
            if bmo_px is not None:
                bmo_x_px, bmo_y_px = float(bmo_px[0]), float(bmo_px[1])

        hit_px = None
        if item.get('ilm_hit_pt') is not None:
            hit_px = aligned_3d_to_image_px(np.array(item['ilm_hit_pt'], dtype=float), sid, aligned_cloud)

        rows.append({
            'slice_id': sid,
            'side': side,
            'scan_angle_deg': float(item.get('scan_angle_deg', (sid - 1) * 15.0)),
            'anatomical_angle_deg': anat,
            'r_mm': float(item.get('r_mm', np.nan)),
            'bottom_len_mm': float(item.get('bottom_len_mm', np.nan)),
            'mra_phi_deg': float(item.get('mra_phi_deg', np.nan)),
            'rw_phi_um': float(item.get('rw_phi_um', np.nan)),
            'top_len_mm': float(item.get('top_len_mm', np.nan)),
            'local_area_mm2': float(item.get('local_area_mm2', np.nan)),
            'bmo_x_px': bmo_x_px,
            'bmo_y_px': bmo_y_px,
            'ilm_hit_x_px': float(hit_px[0]) if hit_px is not None else np.nan,
            'ilm_hit_y_px': float(hit_px[1]) if hit_px is not None else np.nan,
            'sector_8_name': sec8,
            'sector_4_name': sec4,
            'sector_2_name': sec2,
        })

    df = pd.DataFrame(rows)
    cols = [
        'slice_id', 'side', 'scan_angle_deg', 'anatomical_angle_deg', 'r_mm',
        'bottom_len_mm', 'mra_phi_deg', 'rw_phi_um', 'top_len_mm', 'local_area_mm2',
        'bmo_x_px', 'bmo_y_px', 'ilm_hit_x_px', 'ilm_hit_y_px',
        'sector_8_name', 'sector_4_name', 'sector_2_name'
    ]
    return df[cols]


def prepare_lcd_lcci_dataframe(lcd_lcci_df, payload_map, final_cloud, aligned_cloud):
    if lcd_lcci_df is None or len(lcd_lcci_df) == 0:
        cols = [
            'slice_id', 'scan_angle_deg', 'length_D_mm', 'area_S_mm2',
            'lcd_area_mm', 'lcd_direct_mm', 'alid1_mm', 'alid2_mm', 'malid_mm',
            'lcci_area_mm', 'lcci_direct_mm', 'alcci_area_percent', 'alcci_direct_percent',
            'fit_method', 'fit_status', 'fit_quality', 'status', 'reason',
            'bmo_left_x_px', 'bmo_left_y_px', 'bmo_right_x_px', 'bmo_right_y_px',
            'ali_left_x_px', 'ali_left_y_px', 'ali_right_x_px', 'ali_right_y_px',
            'deepest_x_px', 'deepest_y_px',
            'sector_8_name', 'sector_4_name', 'sector_2_name'
        ]
        return pd.DataFrame(columns=cols)

    df = lcd_lcci_df.copy()
    if 'scan_angle_deg' not in df.columns:
        df['scan_angle_deg'] = (df['slice_id'].astype(float) - 1.0) * 15.0
    else:
        fallback = (df['slice_id'].astype(float) - 1.0) * 15.0
        df['scan_angle_deg'] = pd.to_numeric(df['scan_angle_deg'], errors='coerce').fillna(fallback)

    rename_map = {
        'Length_D_mm': 'length_D_mm',
        'Area_S_mm2': 'area_S_mm2',
        'LCD_area_mm': 'lcd_area_mm',
        'LCD_direct_mm': 'lcd_direct_mm',
        'ALID1_mm': 'alid1_mm',
        'ALID2_mm': 'alid2_mm',
        'mALID_mm': 'malid_mm',
        'LCCI_area_mm': 'lcci_area_mm',
        'LCCI_direct_mm': 'lcci_direct_mm',
        'aLCCI_area_pct': 'alcci_area_percent',
        'aLCCI_direct_pct': 'alcci_direct_percent',
    }
    df = df.rename(columns=rename_map)

    bmo_dict = build_slice_bmo_dict(final_cloud.get('BMO_META', []))
    ali_dict = build_slice_ali_dict(final_cloud.get('ALI_META', []))

    bmo_left_x, bmo_left_y, bmo_right_x, bmo_right_y = [], [], [], []
    ali_left_x, ali_left_y, ali_right_x, ali_right_y = [], [], [], []
    deep_x, deep_y = [], []

    for _, row in df.iterrows():
        sid = int(row['slice_id'])
        bmo_sid = bmo_dict.get(sid, {})
        ali_sid = ali_dict.get(sid, {})

        l_bmo = bmo_sid.get('L')
        r_bmo = bmo_sid.get('R')
        l_ali = ali_sid.get('L')
        r_ali = ali_sid.get('R')

        bmo_left_x.append(float(l_bmo.get('pixel_x', np.nan)) if l_bmo else np.nan)
        bmo_left_y.append(float(l_bmo.get('pixel_y', np.nan)) if l_bmo else np.nan)
        bmo_right_x.append(float(r_bmo.get('pixel_x', np.nan)) if r_bmo else np.nan)
        bmo_right_y.append(float(r_bmo.get('pixel_y', np.nan)) if r_bmo else np.nan)

        ali_left_x.append(float(l_ali.get('pixel_x', np.nan)) if l_ali else np.nan)
        ali_left_y.append(float(l_ali.get('pixel_y', np.nan)) if l_ali else np.nan)
        ali_right_x.append(float(r_ali.get('pixel_x', np.nan)) if r_ali else np.nan)
        ali_right_y.append(float(r_ali.get('pixel_y', np.nan)) if r_ali else np.nan)

        payload = payload_map.get(sid)
        if payload is None or payload.get('deepest_xyz') is None:
            deep_x.append(np.nan)
            deep_y.append(np.nan)
        else:
            px = aligned_3d_to_image_px(np.asarray(payload['deepest_xyz'], dtype=float), sid, aligned_cloud)
            deep_x.append(float(px[0]) if px is not None else np.nan)
            deep_y.append(float(px[1]) if px is not None else np.nan)

    df['bmo_left_x_px'] = bmo_left_x
    df['bmo_left_y_px'] = bmo_left_y
    df['bmo_right_x_px'] = bmo_right_x
    df['bmo_right_y_px'] = bmo_right_y
    df['ali_left_x_px'] = ali_left_x
    df['ali_left_y_px'] = ali_left_y
    df['ali_right_x_px'] = ali_right_x
    df['ali_right_y_px'] = ali_right_y
    df['deepest_x_px'] = deep_x
    df['deepest_y_px'] = deep_y

    df = attach_sector_labels(df, 'scan_angle_deg')

    cols = [
        'slice_id', 'scan_angle_deg', 'length_D_mm', 'area_S_mm2',
        'lcd_area_mm', 'lcd_direct_mm', 'alid1_mm', 'alid2_mm', 'malid_mm',
        'lcci_area_mm', 'lcci_direct_mm', 'alcci_area_percent', 'alcci_direct_percent',
        'fit_method', 'fit_status', 'fit_quality', 'status', 'reason',
        'bmo_left_x_px', 'bmo_left_y_px', 'bmo_right_x_px', 'bmo_right_y_px',
        'ali_left_x_px', 'ali_left_y_px', 'ali_right_x_px', 'ali_right_y_px',
        'deepest_x_px', 'deepest_y_px',
        'sector_8_name', 'sector_4_name', 'sector_2_name'
    ]
    return df[cols]


def build_sector_summary_from_tables(mrw_df, mra_df, lcd_lcci_df):
    rows = []
    specs = [
        ('MRW_24', mrw_df, [('mrw_len_um', 'MRW_um')]),
        ('MRA_24', mra_df, [('local_area_mm2', 'MRA_local_area_mm2')]),
        ('LCD_LCCI_12', lcd_lcci_df, [
            ('lcd_area_mm', 'LCD_area_mm'),
            ('lcd_direct_mm', 'LCD_direct_mm'),
            ('lcci_area_mm', 'LCCI_area_mm'),
            ('lcci_direct_mm', 'LCCI_direct_mm'),
            ('alcci_area_percent', 'aLCCI_area_percent'),
            ('alcci_direct_percent', 'aLCCI_direct_percent'),
        ]),
    ]

    for source_table, df, metric_specs in specs:
        if df is None or len(df) == 0:
            continue

        for level_col in ['sector_8_name', 'sector_4_name', 'sector_2_name']:
            level_name = level_col.replace('_name', '')
            if level_col not in df.columns:
                continue

            for metric_col, metric_name in metric_specs:
                if metric_col not in df.columns:
                    continue
                work = df[[level_col, metric_col]].copy()
                work[metric_col] = pd.to_numeric(work[metric_col], errors='coerce')
                work = work.dropna(subset=[level_col, metric_col])
                if len(work) == 0:
                    continue

                grouped = work.groupby(level_col, dropna=False)[metric_col]
                for sector_name, vals in grouped:
                    vals = vals.dropna()
                    if len(vals) == 0:
                        continue
                    rows.append({
                        'source_table': source_table,
                        'level': level_name,
                        'sector_name': sector_name,
                        'metric_name': metric_name,
                        'mean_value': float(vals.mean()),
                        'count': int(vals.shape[0]),
                    })

    return pd.DataFrame(rows)


def read_patient_baseline(excel_path, patient_id):
    out = {'axial_length': np.nan, 'laterality': None}
    try:
        df = pd.read_excel(excel_path)
        if 'Patient_ID' not in df.columns:
            return out
        row = df[df['Patient_ID'] == patient_id]
        if row.empty:
            return out
        if 'Axial_Length' in row.columns:
            out['axial_length'] = float(row.iloc[0]['Axial_Length'])
        if 'Laterality' in row.columns:
            out['laterality'] = str(row.iloc[0]['Laterality']).strip().upper()
    except Exception:
        return out
    return out


def export_results_excel(workbook_path, run_summary_df, self_check_df, mrw_df, mra_df, lcd_lcci_df, sector_df):
    sc = self_check_df.rename(columns={'Check': 'item', 'Status': 'status', 'Detail': 'detail'}).copy()
    for col in ['item', 'status', 'detail']:
        if col not in sc.columns:
            sc[col] = np.nan
    sc = sc[['item', 'status', 'detail']]

    with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
        run_summary_df.to_excel(writer, sheet_name='Run_Summary', index=False)
        sc.to_excel(writer, sheet_name='Self_Check', index=False)
        mrw_df.to_excel(writer, sheet_name='MRW_24', index=False)
        mra_df.to_excel(writer, sheet_name='MRA_24', index=False)
        lcd_lcci_df.to_excel(writer, sheet_name='LCD_LCCI_12', index=False)
        sector_df.to_excel(writer, sheet_name='Sector_Summary', index=False)


def main():
    # ================= user config =================
    my_excel = r"D:/code experiment/pycahrm code/PyCharmMiscProject/data.xls"
    patient_folder = r"D:/code experiment/pycahrm code/PyCharmMiscProject/xu lingxi"
    current_patient = "xu lingxi"

    # Auto-discover usable local paths inside current project folder.
    configured_paths = {
        'excel_path': my_excel,
        'patient_folder': patient_folder,
        'patient_id': current_patient,
    }
    discovered = auto_discover_paths(os.getcwd(), patient_id_hint=current_patient, excel_hint=my_excel)

    if discovered.get('excel_path'):
        my_excel = discovered['excel_path']
    if discovered.get('patient_folder'):
        patient_folder = discovered['patient_folder']
    json_dir = discovered.get('json_dir') or os.path.join(patient_folder, 'JSONs')
    image_dir = discovered.get('image_dir')
    if not image_dir:
        image_pick = choose_image_dir(patient_folder, json_dir)
        image_dir = image_pick['image_dir']

    output_dir = os.path.join(os.getcwd(), 'Final_Analysis_Output')
    qc_slice_dir = os.path.join(output_dir, 'QC_Slices')
    workbook_path = os.path.join(output_dir, f"Final_Results_{current_patient}.xlsx")

    print("\n===== PATH CONFIG (configured) =====")
    print(f"excel_path: {configured_paths['excel_path']}")
    print(f"patient_folder: {configured_paths['patient_folder']}")
    print(f"patient_id: {configured_paths['patient_id']}")

    print("\n===== PATH CONFIG (auto-discovered) =====")
    print(f"excel_path: {my_excel}")
    print(f"patient_folder: {patient_folder}")
    print(f"json_dir: {json_dir}")
    print(f"image_dir: {image_dir}")
    print(f"patient_score: {discovered.get('patient_score')}")
    print(f"image_reason: {discovered.get('image_reason')}")
    for note in discovered.get('notes', []):
        print(f"note: {note}")

    # 0) Startup self-check
    config = {
        'patient_folder': patient_folder,
        'excel_path': my_excel,
        'patient_id': current_patient,
        'output_dir': output_dir,
        'json_dir': json_dir,
        'image_dir': image_dir,
    }

    ok, self_check_df, path_ctx = startup_self_check(config)

    print("\n===== STARTUP SELF-CHECK =====")
    for _, r in self_check_df.iterrows():
        print(f"[{r['Status']}] {r['Check']} :: {r['Detail']}")

    if not ok:
        print("\nSelf-check failed. Stop gracefully.")
        return {
            'status': 'SELF_CHECK_FAILED',
            'self_check': self_check_df,
        }

    print("\nSelf-check passed. Continue to full pipeline...")

    # Clear only the formal output directory before this run.
    clear_formal_output_dir(output_dir)
    os.makedirs(qc_slice_dir, exist_ok=True)

    # 1) Point cloud + Z-axis motion correction + traditional MRW extraction
    ret = process_full_eye_to_3d_point_cloud(
        patient_folder,
        my_excel,
        current_patient,
        json_dir=path_ctx.get('json_dir', json_dir),
        image_dir=path_ctx.get('image_dir', image_dir),
    )
    if ret is None:
        print("final_cloud generation failed. abort")
        return None
    final_cloud, mrw_segments_list = ret

    # 2) BMO best-fit-plane alignment
    aligned_cloud = align_to_bmo_bfp(final_cloud)
    if aligned_cloud is None:
        print("aligned_cloud generation failed. abort")
        return None

    # 3) MRW
    ordered_mrw_segments = order_mrw_segments_like_boundary(mrw_segments_list)
    mrw_df = prepare_mrw_dataframe(ordered_mrw_segments, final_cloud)
    real_mean_mrw = np.mean([item['mrw_len'] for item in mrw_segments_list]) if mrw_segments_list else 0.0

    print("\n[MRW] 24-segment QC")
    for item in ordered_mrw_segments:
        print(
            f"  slice={item['slice_id']:>2} side={item['side']} "
            f"MRW={item['mrw_len']:.2f} um angle={item['mrw_angle_deg']:.2f} deg"
        )
    print(f"[MRW] mean={real_mean_mrw:.2f} um")

    # 4) Current working MRA (formula unchanged)
    gardiner_local_list, global_mra_mm2 = calculate_gardiner_mra(
        aligned_cloud,
        n_sectors=24,
        phi_step_deg=0.5
    )
    mra_df = prepare_mra_dataframe(gardiner_local_list, final_cloud, aligned_cloud)

    print(f"\n[MRA] global={global_mra_mm2:.4f} mm^2")

    # 5) Traditional per-slice LCD/LCCI (formal source)
    lcd_lcci_raw_df, payload_map = compute_traditional_lcd_lcci_all_slices(aligned_cloud)
    lcd_lcci_df = prepare_lcd_lcci_dataframe(lcd_lcci_raw_df, payload_map, final_cloud, aligned_cloud)

    # 6) Sectoral analysis from row-wise angle mapping
    sector_df = build_sector_summary_from_tables(mrw_df, mra_df, lcd_lcci_df)

    # 7) Visualization-based QC
    save_slice_qc_figures(payload_map, mrw_segments_list, gardiner_local_list, qc_slice_dir, final_cloud, aligned_cloud)

    # 8) Export one final Excel workbook
    baseline = read_patient_baseline(my_excel, current_patient)
    laterality = aligned_cloud.get('laterality') or baseline.get('laterality')
    axial_length = aligned_cloud.get('axial_length')
    if axial_length is None or not np.isfinite(axial_length):
        axial_length = baseline.get('axial_length')
    z_status = aligned_cloud.get('z_stabilization_status', 'unknown')

    pass_mask = lcd_lcci_df['status'] == 'PASS' if 'status' in lcd_lcci_df.columns else pd.Series(dtype=bool)
    lcd_pass = lcd_lcci_df[pass_mask] if len(lcd_lcci_df) > 0 and len(pass_mask) == len(lcd_lcci_df) else lcd_lcci_df

    run_summary_df = pd.DataFrame([
        {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'patient_id': current_patient,
            'laterality': laterality,
            'axial_length': float(axial_length) if axial_length is not None and np.isfinite(axial_length) else np.nan,
            'z_stabilization_status': z_status,
            'mean_mrw_um': float(real_mean_mrw),
            'global_mra_mm2': float(global_mra_mm2),
            'mean_lcd_area_mm': float(lcd_pass['lcd_area_mm'].mean()) if 'lcd_area_mm' in lcd_pass.columns and len(lcd_pass) > 0 else np.nan,
            'mean_lcd_direct_mm': float(lcd_pass['lcd_direct_mm'].mean()) if 'lcd_direct_mm' in lcd_pass.columns and len(lcd_pass) > 0 else np.nan,
            'mean_lcci_area_mm': float(lcd_pass['lcci_area_mm'].mean()) if 'lcci_area_mm' in lcd_pass.columns and len(lcd_pass) > 0 else np.nan,
            'mean_lcci_direct_mm': float(lcd_pass['lcci_direct_mm'].mean()) if 'lcci_direct_mm' in lcd_pass.columns and len(lcd_pass) > 0 else np.nan,
            'mean_alcci_area_percent': float(lcd_pass['alcci_area_percent'].mean()) if 'alcci_area_percent' in lcd_pass.columns and len(lcd_pass) > 0 else np.nan,
            'mean_alcci_direct_percent': float(lcd_pass['alcci_direct_percent'].mean()) if 'alcci_direct_percent' in lcd_pass.columns and len(lcd_pass) > 0 else np.nan,
            'lcd_lcci_source': 'per-slice fitted ALCS arc',
            'lcd_lcci_pass_slices': int((lcd_lcci_df['status'] == 'PASS').sum()) if 'status' in lcd_lcci_df.columns else 0,
            'lcd_lcci_total_slices': int(len(lcd_lcci_df)),
            'excel_output': workbook_path,
            'qc_slice_dir': qc_slice_dir,
        }
    ])

    export_results_excel(
        workbook_path,
        run_summary_df,
        self_check_df,
        mrw_df,
        mra_df,
        lcd_lcci_df,
        sector_df,
    )

    print("\nPipeline completed.")
    print(f"Workbook: {workbook_path}")
    print(f"QC slice figures: {qc_slice_dir}")

    return {
        'status': 'OK',
        'workbook': workbook_path,
        'mrw_mean_um': float(real_mean_mrw),
        'mra_global_mm2': float(global_mra_mm2),
        'self_check': self_check_df,
    }


if __name__ == "__main__":
    main()
