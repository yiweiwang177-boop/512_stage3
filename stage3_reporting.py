from datetime import datetime
import re
import unicodedata

import numpy as np
import pandas as pd


SECTOR_8_BOUNDS = [
    (0.0, 63.5, 'NS (Nasal Superior / 榧讳笂渚?)'),
    (63.5, 109.1, 'SN (Superior Nasal / 涓婇蓟渚?)'),
    (109.1, 144.7, 'ST (Superior Temporal / 涓婇渚?)'),
    (144.7, 187.6, 'TS (Temporal Superior / 棰炰笂渚?)'),
    (187.6, 224.1, 'TI (Temporal Inferior / 棰炰笅渚?)'),
    (224.1, 260.1, 'IT (Inferior Temporal / 涓嬮渚?)'),
    (260.1, 310.1, 'IN (Inferior Nasal / 涓嬮蓟渚?)'),
    (310.1, 360.0, 'NI (Nasal Inferior / 榧讳笅渚?)'),
]

SECTOR_4_MAP = {
    'NS (Nasal Superior / 榧讳笂渚?)': 'Nasal (Nasal / 榧讳晶)',
    'NI (Nasal Inferior / 榧讳笅渚?)': 'Nasal (Nasal / 榧讳晶)',
    'SN (Superior Nasal / 涓婇蓟渚?)': 'Superior (Superior / 涓婃柟)',
    'ST (Superior Temporal / 涓婇渚?)': 'Superior (Superior / 涓婃柟)',
    'TS (Temporal Superior / 棰炰笂渚?)': 'Temporal (Temporal / 棰炰晶)',
    'TI (Temporal Inferior / 棰炰笅渚?)': 'Temporal (Temporal / 棰炰晶)',
    'IT (Inferior Temporal / 涓嬮渚?)': 'Inferior (Inferior / 涓嬫柟)',
    'IN (Inferior Nasal / 涓嬮蓟渚?)': 'Inferior (Inferior / 涓嬫柟)',
}

SECTOR_2_MAP = {
    'NS (Nasal Superior / 榧讳笂渚?)': 'Superior Half (Superior Half / 涓婂崐閮?)',
    'SN (Superior Nasal / 涓婇蓟渚?)': 'Superior Half (Superior Half / 涓婂崐閮?)',
    'ST (Superior Temporal / 涓婇渚?)': 'Superior Half (Superior Half / 涓婂崐閮?)',
    'TS (Temporal Superior / 棰炰笂渚?)': 'Superior Half (Superior Half / 涓婂崐閮?)',
    'TI (Temporal Inferior / 棰炰笅渚?)': 'Inferior Half (Inferior Half / 涓嬪崐閮?)',
    'IT (Inferior Temporal / 涓嬮渚?)': 'Inferior Half (Inferior Half / 涓嬪崐閮?)',
    'IN (Inferior Nasal / 涓嬮蓟渚?)': 'Inferior Half (Inferior Half / 涓嬪崐閮?)',
    'NI (Nasal Inferior / 榧讳笅渚?)': 'Inferior Half (Inferior Half / 涓嬪崐閮?)',
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


def build_sector_summary_from_tables(mrw_df, mra_df, lcd_lcci_df):
    """Build the legacy Stage3 sector summary table.

    This preserves the existing compatibility behavior, including the legacy
    mean-style aggregation used by the old MRA reporting path. The future 512
    ONH3D pipeline should not rely on this MRA aggregation semantics; its
    Sector_Summary is expected to be produced directly by the report adapter.
    """
    rows = []
    specs = [
        ('MRW_detail', mrw_df, [('mrw_len_um', 'MRW_um')]),
        ('MRA_detail', mra_df, [('local_area_mm2', 'MRA_local_area_mm2')]),
        ('LCD_LCCI_detail', lcd_lcci_df, [
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

    cols = ['source_table', 'level', 'sector_name', 'metric_name', 'mean_value', 'count']
    return pd.DataFrame(rows, columns=cols)


def slugify_sector_label(label):
    txt = unicodedata.normalize('NFKD', str(label)).encode('ascii', 'ignore').decode('ascii')
    txt = txt.lower()
    txt = re.sub(r'[^a-z0-9]+', '_', txt)
    txt = re.sub(r'_+', '_', txt).strip('_')
    return txt or 'unknown_sector'


def _ordered_unique_non_null(values):
    out = []
    seen = set()
    for value in values:
        if value is None:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def get_expected_master_sector_schema():
    sector_8_labels = [name for _, _, name in SECTOR_8_BOUNDS]
    sector_4_labels = _ordered_unique_non_null(SECTOR_4_MAP.get(name) for name in sector_8_labels)
    sector_2_labels = _ordered_unique_non_null(SECTOR_2_MAP.get(name) for name in sector_8_labels)
    labels_by_level = {
        'sector_8': sector_8_labels,
        'sector_4': sector_4_labels,
        'sector_2': sector_2_labels,
    }

    ordered_columns = []
    column_map = {}
    specs = [
        ('MRW_detail', 'MRW_um', 'MRW', 'um'),
        ('MRA_detail', 'MRA_local_area_mm2', 'MRA', 'mm2'),
    ]
    for source_table, metric_name, prefix, unit in specs:
        for level in ['sector_8', 'sector_4', 'sector_2']:
            for sector_name in labels_by_level[level]:
                col = f'{prefix}_{level}_{slugify_sector_label(sector_name)}_{unit}'
                ordered_columns.append(col)
                column_map[(source_table, metric_name, level, sector_name)] = col

    return {
        'labels_by_level': labels_by_level,
        'ordered_columns': ordered_columns,
        'column_map': column_map,
    }


def summarize_review_status(slice_meta):
    statuses = []
    for item in slice_meta.values():
        status = item.get('review_status')
        if status is None:
            continue
        status_txt = str(status).strip()
        if status_txt:
            statuses.append(status_txt)
    if not statuses:
        return np.nan
    unique = sorted(set(statuses))
    if len(unique) == 1:
        return unique[0]
    return '|'.join(unique)


def summarize_lcd_lcci_pass_metrics(lcd_lcci_df):
    if lcd_lcci_df is None or len(lcd_lcci_df) == 0 or 'status' not in lcd_lcci_df.columns:
        empty = pd.DataFrame(columns=lcd_lcci_df.columns if lcd_lcci_df is not None else [])
        return {
            'pass_df': empty,
            'pass_slices': 0,
            'total_slices': int(len(lcd_lcci_df)) if lcd_lcci_df is not None else 0,
            'pass_available': False,
            'lcd_area_mean': np.nan,
            'lcci_area_mean': np.nan,
            'lcd_direct_mean': np.nan,
            'lcci_direct_mean': np.nan,
            'alcci_area_mean': np.nan,
            'alcci_direct_mean': np.nan,
        }

    pass_df = lcd_lcci_df[lcd_lcci_df['status'].astype(str) == 'PASS'].copy()

    def metric_mean(df, col):
        if col not in df.columns or len(df) == 0:
            return np.nan
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.mean())

    return {
        'pass_df': pass_df,
        'pass_slices': int(len(pass_df)),
        'total_slices': int(len(lcd_lcci_df)),
        'pass_available': bool(len(pass_df) > 0),
        'lcd_area_mean': metric_mean(pass_df, 'lcd_area_mm'),
        'lcci_area_mean': metric_mean(pass_df, 'lcci_area_mm'),
        'lcd_direct_mean': metric_mean(pass_df, 'lcd_direct_mm'),
        'lcci_direct_mean': metric_mean(pass_df, 'lcci_direct_mm'),
        'alcci_area_mean': metric_mean(pass_df, 'alcci_area_percent'),
        'alcci_direct_mean': metric_mean(pass_df, 'alcci_direct_percent'),
    }


def build_master_sector_columns(sector_df):
    schema = get_expected_master_sector_schema()
    master_sector_values = {col: np.nan for col in schema['ordered_columns']}
    if sector_df is None or len(sector_df) == 0:
        return master_sector_values

    for _, row in sector_df.iterrows():
        source_table = row.get('source_table')
        metric_name = row.get('metric_name')
        level = row.get('level')
        sector_name = row.get('sector_name')

        if source_table not in {'MRW_detail', 'MRA_detail'}:
            continue

        col = schema['column_map'].get((source_table, metric_name, level, sector_name))
        if col is None:
            print(
                f"[Master_Table] ignoring unexpected sector row: "
                f"source={source_table}, metric={metric_name}, level={level}, sector={sector_name}"
            )
            continue

        mean_value = pd.to_numeric(pd.Series([row.get('mean_value')]), errors='coerce').iloc[0]
        master_sector_values[col] = float(mean_value) if pd.notna(mean_value) else np.nan

    return master_sector_values


def build_master_table(
    case_id,
    patient_id,
    laterality,
    axial_length,
    baseline_row,
    stage2_schema_version,
    z_stabilization_status,
    self_check_df,
    final_cloud,
    mrw_df,
    global_mra_mm2,
    lcd_lcci_df,
    sector_df,
):
    # Master_Table is the canonical source of truth for this output layer.
    baseline_row = baseline_row or {}
    slice_meta = final_cloud.get('SLICE_META', {}) if final_cloud is not None else {}
    expected_slice_count = 12
    fail_count = 0
    if self_check_df is not None and 'Status' in self_check_df.columns:
        fail_count = int((self_check_df['Status'].astype(str) == 'FAIL').sum())

    mrw_global_mean = np.nan
    if mrw_df is not None and 'mrw_len_um' in mrw_df.columns and len(mrw_df) > 0:
        vals = pd.to_numeric(mrw_df['mrw_len_um'], errors='coerce').dropna()
        if len(vals) > 0:
            mrw_global_mean = float(vals.mean())

    pass_metrics = summarize_lcd_lcci_pass_metrics(lcd_lcci_df)

    row = {
        'case_id': case_id,
        'patient_id': patient_id,
        'laterality': laterality,
        'axial_length': float(axial_length) if axial_length is not None and np.isfinite(axial_length) else np.nan,
    }

    for key, value in baseline_row.items():
        if key in {'Patient_ID', 'Laterality', 'Axial_Length'}:
            continue
        row[key] = value

    row.update({
        'MRW_global_mean': mrw_global_mean,
        'MRA_global_mm2': float(global_mra_mm2) if global_mra_mm2 is not None and np.isfinite(global_mra_mm2) else np.nan,
        # Area-based LCD/LCCI globals are canonical in Master_Table for this pass.
        'LCD_global_mean': pass_metrics['lcd_area_mean'],
        'LCCI_global_mean': pass_metrics['lcci_area_mean'],
    })

    row.update(build_master_sector_columns(sector_df))

    row.update({
        'review_status_summary': summarize_review_status(slice_meta),
        'slice_count': int(len(slice_meta)),
        'expected_slice_count': int(expected_slice_count),
        'slice_count_complete': bool(len(slice_meta) == expected_slice_count),
        'stage2_schema_version': stage2_schema_version if stage2_schema_version is not None else np.nan,
        'z_stabilization_status': z_stabilization_status,
        'self_check_fail_count': int(fail_count),
        'lcd_lcci_pass_slices': int(pass_metrics['pass_slices']),
        'lcd_lcci_total_slices': int(pass_metrics['total_slices']),
        'lcd_lcci_pass_available': bool(pass_metrics['pass_available']),
    })

    return pd.DataFrame([row])


def build_run_summary_df(master_df, lcd_lcci_df, workbook_path, qc_slice_dir, timestamp=None):
    # Run_Summary is a narrow downstream compatibility sheet only.
    master_row = master_df.iloc[0].to_dict() if master_df is not None and len(master_df) > 0 else {}
    pass_metrics = summarize_lcd_lcci_pass_metrics(lcd_lcci_df)
    out = {
        'timestamp': timestamp or datetime.now().isoformat(timespec='seconds'),
        'case_id': master_row.get('case_id'),
        'patient_id': master_row.get('patient_id'),
        'laterality': master_row.get('laterality'),
        'axial_length': master_row.get('axial_length'),
        'z_stabilization_status': master_row.get('z_stabilization_status'),
        'mean_mrw_um': master_row.get('MRW_global_mean'),
        'global_mra_mm2': master_row.get('MRA_global_mm2'),
        'mean_lcd_area_mm': master_row.get('LCD_global_mean'),
        'mean_lcci_area_mm': master_row.get('LCCI_global_mean'),
        'LCD_direct_global_mean': pass_metrics['lcd_direct_mean'],
        'LCCI_direct_global_mean': pass_metrics['lcci_direct_mean'],
        'mean_alcci_area_percent': pass_metrics['alcci_area_mean'],
        'mean_alcci_direct_percent': pass_metrics['alcci_direct_mean'],
        'lcd_lcci_source': 'per-slice fitted ALCS arc',
        'lcd_lcci_pass_slices': master_row.get('lcd_lcci_pass_slices'),
        'lcd_lcci_total_slices': master_row.get('lcd_lcci_total_slices'),
        'excel_output': workbook_path,
        'qc_slice_dir': qc_slice_dir,
    }
    return pd.DataFrame([out])


def export_results_excel(workbook_path, master_df, run_summary_df, self_check_df, mrw_df, mra_df, lcd_lcci_df, sector_df):
    sc = self_check_df.rename(columns={'Check': 'item', 'Status': 'status', 'Detail': 'detail'}).copy()
    for col in ['item', 'status', 'detail']:
        if col not in sc.columns:
            sc[col] = np.nan
    sc = sc[['item', 'status', 'detail']]

    with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
        master_df.to_excel(writer, sheet_name='Master_Table', index=False)
        run_summary_df.to_excel(writer, sheet_name='Run_Summary', index=False)
        sc.to_excel(writer, sheet_name='Self_Check', index=False)
        mrw_df.to_excel(writer, sheet_name='MRW_detail', index=False)
        mra_df.to_excel(writer, sheet_name='MRA_detail', index=False)
        lcd_lcci_df.to_excel(writer, sheet_name='LCD_LCCI_detail', index=False)
        sector_df.to_excel(writer, sheet_name='Sector_Summary', index=False)
