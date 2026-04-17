import asyncio
import time
from pathlib import Path

import pandas as pd

status_map = {
    "planned_installation": "planned_installation",
    "planned": "planned_installation",
    "ok": "operational",
    "op": "operational",
    "operational": "operational",
    "faulty": "faulty",
    "broken": "faulty",
    "maintenance_scheduled": "maintenance_scheduled",
    "maintenance": "maintenance_scheduled",
}


def discover_input_files(base_dir: Path) -> list[Path]:
    """Найти входные xlsx файлы в проекте 5."""
    files = sorted(base_dir.glob("medical_diagnostic_devices_*.xlsx"))
    if files:
        return files

    files = sorted(base_dir.glob("*.xlsx"))
    if files:
        return files

    fallback = sorted((base_dir.parent / "3").glob("*.xlsx"))
    return fallback


def load_data(file_path: Path) -> pd.DataFrame:
    """Загрузить и подготовить данные."""
    df = pd.read_excel(file_path)

    date_cols = (
        "install_date",
        "warranty_until",
        "last_calibration_date",
        "last_service_date",
    )
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    numeric_cols = ("issues_reported_12mo", "failure_count_12mo", "uptime_pct")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    normalized = df["status"].astype(str).str.strip().str.lower()
    df["status_normalized"] = normalized.map(status_map).fillna("unknown")

    invalid_calibration = (
        df["last_calibration_date"].notna()
        & df["install_date"].notna()
        & (df["last_calibration_date"] < df["install_date"])
    )
    df.loc[invalid_calibration, "last_calibration_date"] = pd.NaT
    return df


def _warranty_parts(
    df: pd.DataFrame, today: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Собрать таблицы по гарантии."""
    expired = df[df["warranty_until"] < today].copy()
    active = df[df["warranty_until"] >= today].copy()
    unknown = df[df["warranty_until"].isna()].copy()

    for part in (expired, active, unknown):
        part.sort_values(["clinic_name", "warranty_until", "device_id"], inplace=True)

    return expired, active, unknown


def _clinics_problems(df: pd.DataFrame) -> pd.DataFrame:
    """Собрать таблицу клиник с наибольшим количеством проблем."""
    clinics = (
        df.groupby(["clinic_id", "clinic_name", "city"], dropna=False)
        .agg(
            devices_count=("device_id", "nunique"),
            issues_total_12mo=("issues_reported_12mo", "sum"),
            failures_total_12mo=("failure_count_12mo", "sum"),
            avg_uptime_pct=("uptime_pct", "mean"),
        )
        .reset_index()
    )

    cols = ["issues_total_12mo", "failures_total_12mo", "avg_uptime_pct"]
    clinics[cols] = clinics[cols].fillna(0)
    clinics = clinics.sort_values(
        ["issues_total_12mo", "failures_total_12mo"],
        ascending=False,
    )
    return clinics


def _calibration_report(df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Собрать отчет по срокам калибровки."""
    base_date = df["last_calibration_date"].fillna(df["install_date"])
    calibration = df[
        [
            "device_id",
            "clinic_id",
            "clinic_name",
            "city",
            "model",
            "install_date",
            "last_calibration_date",
            "status_normalized",
        ]
    ].copy()

    calibration["next_calibration_due"] = base_date + pd.DateOffset(months=12)
    calibration["days_to_due"] = (calibration["next_calibration_due"] - today).dt.days
    calibration["calibration_status"] = "ok"
    calibration.loc[
        calibration["next_calibration_due"].notna()
        & (calibration["days_to_due"] < 0),
        "calibration_status",
    ] = "overdue"
    calibration.loc[
        calibration["next_calibration_due"].notna()
        & calibration["days_to_due"].between(0, 30),
        "calibration_status",
    ] = "due_30_days"
    calibration = calibration.sort_values(
        ["calibration_status", "days_to_due", "clinic_name"]
    )
    return calibration


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Собрать сводную таблицу по клиникам и моделям."""
    summary = (
        df.groupby(["clinic_id", "clinic_name", "city", "model"], dropna=False)
        .agg(
            devices_count=("device_id", "nunique"),
            avg_uptime_pct=("uptime_pct", "mean"),
            issues_total_12mo=("issues_reported_12mo", "sum"),
            failures_total_12mo=("failure_count_12mo", "sum"),
        )
        .reset_index()
        .sort_values(["clinic_name", "model"])
    )
    return summary


def _reports_dict(
    expired: pd.DataFrame,
    active: pd.DataFrame,
    unknown: pd.DataFrame,
    clinics: pd.DataFrame,
    calibration: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Собрать словарь отчетов и имен выходных файлов."""
    return {
        "warranty_expired.xlsx": expired,
        "warranty_active.xlsx": active,
        "warranty_unknown.xlsx": unknown,
        "clinics_problems.xlsx": clinics,
        "calibration.xlsx": calibration,
        "clinic_equipment_summary.xlsx": summary,
    }


def build_reports(
    df: pd.DataFrame, today: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Собрать все таблицы отчета."""
    expired, active, unknown = _warranty_parts(df, today)
    clinics = _clinics_problems(df)
    calibration = _calibration_report(df, today)
    summary = _summary_table(df)
    return _reports_dict(expired, active, unknown, clinics, calibration, summary)


def _prepare_output_dir(reports_dir: Path) -> None:
    """Подготовить папку отчетов и очистить старые xlsx."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    for old_file in reports_dir.glob("*.xlsx"):
        old_file.unlink()


def save_reports(reports_dir: Path, reports: dict[str, pd.DataFrame]) -> None:
    """Сохранить все таблицы в отдельные excel файлы."""
    _prepare_output_dir(reports_dir)

    for file_name, frame in reports.items():
        frame.to_excel(reports_dir / file_name, index=False)


def load_all_sync(input_files: list[Path]) -> pd.DataFrame:
    """Синхронно прочитать и объединить все входные файлы."""
    frames = [load_data(file_path) for file_path in input_files]
    return pd.concat(frames, ignore_index=True)


async def load_all_async(input_files: list[Path]) -> pd.DataFrame:
    """Асинхронно прочитать и объединить все входные файлы."""
    tasks = [asyncio.to_thread(load_data, file_path) for file_path in input_files]
    frames = await asyncio.gather(*tasks)
    return pd.concat(frames, ignore_index=True)


async def build_reports_async(
    df: pd.DataFrame, today: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Асинхронно собрать все таблицы отчета."""
    warranty_task = asyncio.to_thread(_warranty_parts, df, today)
    clinics_task = asyncio.to_thread(_clinics_problems, df)
    calibration_task = asyncio.to_thread(_calibration_report, df, today)
    summary_task = asyncio.to_thread(_summary_table, df)

    warranty_parts, clinics, calibration, summary = await asyncio.gather(
        warranty_task,
        clinics_task,
        calibration_task,
        summary_task,
    )
    expired, active, unknown = warranty_parts
    return _reports_dict(expired, active, unknown, clinics, calibration, summary)


async def save_reports_async(
    reports_dir: Path, reports: dict[str, pd.DataFrame]
) -> None:
    """Асинхронно сохранить все таблицы отчета."""
    _prepare_output_dir(reports_dir)

    tasks = [
        asyncio.to_thread(frame.to_excel, reports_dir / file_name, index=False)
        for file_name, frame in reports.items()
    ]
    await asyncio.gather(*tasks)


def run_sync_pipeline(input_files: list[Path], output_dir: Path) -> float:
    """Запустить синхронный пайплайн и вернуть время выполнения."""
    start = time.perf_counter()

    df = load_all_sync(input_files)
    today = pd.Timestamp.today().normalize()
    reports = build_reports(df, today)
    save_reports(output_dir, reports)

    return time.perf_counter() - start


async def run_async_pipeline(input_files: list[Path], output_dir: Path) -> float:
    """Запустить асинхронный пайплайн и вернуть время выполнения."""
    start = time.perf_counter()

    df = await load_all_async(input_files)
    today = pd.Timestamp.today().normalize()
    reports = await build_reports_async(df, today)
    await save_reports_async(output_dir, reports)

    return time.perf_counter() - start


def save_timing_report(reports_dir: Path, sync_time: float, async_time: float) -> None:
    """Сохранить сравнение времени в csv и txt."""
    comparison = pd.DataFrame(
        {
            "mode": ["sync", "async"],
            "seconds": [round(sync_time, 4), round(async_time, 4)],
        }
    )
    comparison.to_csv(reports_dir / "execution_time_comparison.csv", index=False)

    speedup = sync_time / async_time if async_time > 0 else 0.0
    lines = [
        "Сравнение времени выполнения",
        f"sync: {sync_time:.4f} сек",
        f"async: {async_time:.4f} сек",
        f"ускорение sync/async: {speedup:.2f}x",
    ]
    with open(
        reports_dir / "execution_time_comparison.txt", "w", encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))


def main() -> None:
    """Точка входа."""
    base_dir = Path(__file__).resolve().parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(base_dir)
    if not input_files:
        raise FileNotFoundError("xlsx файлы не найдены в папках 2 сем/5 и 2 сем/3")

    print("найдены входные файлы:")
    for file_path in input_files:
        print(f"- {file_path.name}")

    sync_dir = reports_dir / "sync"
    async_dir = reports_dir / "async"

    sync_time = run_sync_pipeline(input_files, sync_dir)
    async_time = asyncio.run(run_async_pipeline(input_files, async_dir))
    save_timing_report(reports_dir, sync_time, async_time)

    speedup = sync_time / async_time if async_time > 0 else 0.0
    print("\n=== сравнение времени выполнения ===")
    print(f"sync : {sync_time:.4f} сек")
    print(f"async: {async_time:.4f} сек")
    print(f"sync/async: {speedup:.2f}x")

    print("\nотчеты сохранены:")
    print(f"- {sync_dir}")
    print(f"- {async_dir}")
    print(f"- {reports_dir / 'execution_time_comparison.csv'}")
    print(f"- {reports_dir / 'execution_time_comparison.txt'}")


if __name__ == "__main__":
    main()
