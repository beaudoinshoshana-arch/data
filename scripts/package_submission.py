from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "submission"
ZIP_PATH = OUT / "wwtp_safe_marl_submission.zip"
MANIFEST_PATH = OUT / "submission_manifest.json"

INCLUDE_FILES = [
    "README.md",
    "README_stage1_data.md",
    "README_stage2_model.md",
    "README_paper_repro.md",
    ".codex/rules/wwtp-safe-marl-workflow.md",
    ".codex/skills/wwtp-safe-marl/SKILL.md",
    "configs/external_sources.default.json",
    "configs/safe_marl.default.json",
    "configs/stage2_constraints.default.json",
    "dashboard/backend/__init__.py",
    "dashboard/backend/main.py",
    "dashboard/backend/requirements.txt",
    "dashboard/backend/schemas.py",
    "dashboard/backend/services.py",
    "dashboard/frontend/index.html",
    "dashboard/frontend/package.json",
    "dashboard/frontend/package-lock.json",
    "dashboard/frontend/tsconfig.json",
    "dashboard/frontend/vite.config.ts",
    "dashboard/frontend/src/main.tsx",
    "dashboard/frontend/src/styles.css",
    "docs/competition_report.md",
    "docs/competition_report.docx",
    "docs/completion_audit.md",
    "docs/data_sources_and_external_downloads.md",
    "docs/demo_script.md",
    "docs/deployment_guide.md",
    "docs/final_scoring_matrix.md",
    "docs/literature_basis_table.md",
    "docs/system_integration_manual.md",
    "docs/ui_dashboard_reference_resources.md",
    "docs/reflection_log.md",
    "outputs/dashboard_verified.png",
    "outputs/decision_benefit/decision_benefit_report.md",
    "outputs/decision_benefit/decision_benefit_summary.json",
    "outputs/fusion_data/scenario_library.csv",
    "outputs/fusion_data/source_registry.json",
    "outputs/paper_repro_integrated_control/report.md",
    "outputs/paper_repro_integrated_control/summary.json",
    "outputs/paper_repro_integrated_control/steady_state_pareto.csv",
    "outputs/paper_repro_integrated_control/online_dynamic_replay.csv",
    "outputs/safe_marl/rl_recommendations_test.csv",
    "outputs/safe_marl/summary.json",
    "outputs/safe_marl/training_curve.csv",
    "outputs/safe_marl/model_card.md",
    "outputs/model_efficiency/model_efficiency_summary.json",
    "outputs/model_efficiency/model_efficiency_report.md",
    "outputs/model_efficiency/compact_surrogate_candidates.csv",
    "outputs/model_efficiency/compact_surrogate_features.json",
    "outputs/stage1_data/reports/stage1_summary.json",
    "outputs/stage2_model/feature_columns.json",
    "outputs/stage2_model/leaderboard.json",
    "outputs/stage2_model/model_card.md",
    "outputs/stage2_model/scenario_recommendations_test.csv",
    "outputs/stage2_model/summary.json",
    "outputs/stage2_model/test_predictions.csv",
    "pyproject.toml",
    "scripts/build_external_fusion_dataset.py",
    "scripts/build_stage1_datasets.py",
    "scripts/evaluate_decision_benefits.py",
    "scripts/evaluate_model_energy_efficiency.py",
    "scripts/generate_competition_report.cjs",
    "scripts/package_submission.py",
    "scripts/reproduce_integrated_realtime_control.py",
    "scripts/run_dashboard.ps1",
    "scripts/run_pipeline.ps1",
    "scripts/train_safe_marl.py",
    "scripts/train_stage2_model.py",
    "scripts/validate_stage1_outputs.py",
    "tests/test_api_contract.py",
    "tests/test_fusion_outputs.py",
    "tests/test_safe_marl.py",
    "wwtp_decision/__init__.py",
    "wwtp_decision/safe_marl.py",
    "文献/PDF文献整理与项目公式汇总.md",
    "文献/PDF逐篇全量公式手册.md",
    "文献/文献索引与建模公式笔记.md",
    "文献/文献链接清单.csv",
    "文献/结构性研究空白_GAPS.md",
    "本项目数据建模要求-参考Integrated实时智能控制论文.md",
    "污水厂曝气加药智能决策项目实施计划.md",
]


def add_tree(zip_file: zipfile.ZipFile, root: Path, prefix: str) -> list[str]:
    added: list[str] = []
    if not root.exists():
        return added
    for file in root.rglob("*"):
        if file.is_file():
            arcname = str(Path(prefix) / file.relative_to(root)).replace("\\", "/")
            zip_file.write(file, arcname)
            added.append(arcname)
    return added


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    included: list[str] = []
    missing: list[str] = []
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in INCLUDE_FILES:
            path = ROOT / rel
            if path.exists() and path.is_file():
                zf.write(path, rel.replace("\\", "/"))
                included.append(rel.replace("\\", "/"))
            else:
                missing.append(rel.replace("\\", "/"))
        included.extend(add_tree(zf, ROOT / "dashboard" / "frontend" / "dist", "dashboard/frontend/dist"))
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "zip_path": str(ZIP_PATH),
        "included_count": len(included),
        "missing_count": len(missing),
        "included": included,
        "missing": missing,
        "notes": [
            "Raw data, PDFs, Office source files, model binaries, virtual environments, and node_modules are intentionally excluded.",
            "The generated DOCX report and frontend dist are included for competition review convenience.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"zip": str(ZIP_PATH), "included": len(included), "missing": len(missing)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
