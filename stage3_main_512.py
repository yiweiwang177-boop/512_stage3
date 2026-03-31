import argparse
from typing import Any, Dict, Optional

import pandas as pd

from stage3_onh3d_contract import ONH3DCase, validate_onh3d_case
from stage3_onh3d_metrics import ONH3DMetricConfig, compute_onh3d_metrics
from stage3_onh3d_report_adapter import adapt_onh3d_metrics_to_stage3_tables
from stage3_reporting import build_master_table, build_run_summary_df, export_results_excel


def parse_args_512(argv=None):
    parser = argparse.ArgumentParser(description="Stage3 512 ONH3D pipeline entry")
    parser.add_argument("--onh3d-case")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def load_onh3d_case(case_path: Optional[str]) -> ONH3DCase:
    raise NotImplementedError("ONH3D case loading is not implemented yet.")


def run_onh3d_stage3(
    case: ONH3DCase,
    *,
    config: Optional[ONH3DMetricConfig] = None,
    baseline_row: Optional[Dict[str, Any]] = None,
    self_check_df: Optional[pd.DataFrame] = None,
    workbook_path: Optional[str] = None,
    qc_slice_dir: Optional[str] = None,
):
    metrics_result = compute_onh3d_metrics(case, config=config)
    adapted = adapt_onh3d_metrics_to_stage3_tables(case, metrics_result)
    baseline_row = baseline_row or {}
    self_check_df = self_check_df if self_check_df is not None else pd.DataFrame(columns=["Check", "Status", "Detail"])

    final_cloud = {"SLICE_META": {}}
    master_df = build_master_table(
        case_id=case.case_id,
        patient_id=case.patient_id,
        laterality=case.laterality,
        axial_length=case.axial_length,
        baseline_row=baseline_row,
        stage2_schema_version="onh3d_512",
        z_stabilization_status="not_applicable_onh3d_512",
        self_check_df=self_check_df,
        final_cloud=final_cloud,
        mrw_df=adapted["mrw_df"],
        global_mra_mm2=metrics_result.MRA_global_sum_mm2,
        lcd_lcci_df=pd.DataFrame(),
        sector_df=adapted["master_sector_df"],
    )
    run_summary_df = build_run_summary_df(
        master_df,
        pd.DataFrame(),
        workbook_path=workbook_path or "",
        qc_slice_dir=qc_slice_dir or "",
    )
    if workbook_path:
        export_results_excel(
            workbook_path,
            master_df,
            run_summary_df,
            self_check_df,
            adapted["mrw_df"],
            adapted["mra_df"],
            pd.DataFrame(),
            adapted["sector_df"],
        )
    return {
        "master_df": master_df,
        "run_summary_df": run_summary_df,
        "report_meta": adapted["report_meta"],
    }


def main_512(argv=None):
    args = parse_args_512(argv)
    case = load_onh3d_case(args.onh3d_case)
    validate_onh3d_case(case)
    return run_onh3d_stage3(case, workbook_path=None, qc_slice_dir=args.output_dir)


if __name__ == "__main__":
    main_512()
