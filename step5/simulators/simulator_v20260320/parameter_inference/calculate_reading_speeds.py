import os
import re
import argparse
import pandas as pd


def extract_param_value(folder_name: str, param_name: str):
    """
    Extract parameter value from folder names such as:
    rho_0.1
    kappa_1.5
    reading_speed_rho_0.2
    """
    pattern = rf"{param_name}[_=]([-+]?\d*\.?\d+)"
    match = re.search(pattern, folder_name)

    if match is None:
        return None

    return float(match.group(1))


def summarize_reading_speed_grid(
    input_root: str,
    output_csv: str,
    param_name: str = "rho",
    reading_speed_col: str = "reading_speed",
    file_name: str = "trial_metrics.csv",
):
    """
    Traverse simulation folders and summarize reading speed.

    Output columns:
        param_name
        mean
        std
        sem
        n
    """

    rows = []

    param_folders = sorted(
        [
            f for f in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, f))
        ]
    )

    for folder in param_folders:
        folder_path = os.path.join(input_root, folder)

        param_value = extract_param_value(folder, param_name)

        if param_value is None:
            print(f"[SKIP] Cannot extract {param_name} from folder: {folder}")
            continue

        csv_path = os.path.join(folder_path, file_name)

        if not os.path.exists(csv_path):
            print(f"[SKIP] Missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        if reading_speed_col not in df.columns:
            print(f"[SKIP] Column '{reading_speed_col}' not found in: {csv_path}")
            print(f"       Available columns: {list(df.columns)}")
            continue

        values = df[reading_speed_col].dropna()

        if len(values) == 0:
            print(f"[SKIP] No valid reading speed values in: {csv_path}")
            continue

        mean = values.mean()
        std = values.std(ddof=1)
        sem = values.sem(ddof=1)
        n = len(values)

        rows.append({
            param_name: param_value,
            "mean": mean,
            "std": std,
            "sem": sem,
            "n": n,
            "source_folder": folder,
        })

    summary_df = pd.DataFrame(rows)

    if summary_df.empty:
        raise ValueError("No valid reading-speed data were found.")

    summary_df = summary_df.sort_values(by=param_name)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    print(f"Saved reading-speed summary to: {output_csv}")
    print(summary_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        required=True,
        help="Root folder containing parameter subfolders."
    )

    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to save the reading-speed summary CSV."
    )

    parser.add_argument(
        "--param_name",
        default="rho",
        help="Parameter name to extract from folder names, e.g., rho, kappa, lambda."
    )

    parser.add_argument(
        "--reading_speed_col",
        default="reading_speed",
        help="Column name for reading speed in each trial_metrics.csv file."
    )

    parser.add_argument(
        "--file_name",
        default="trial_metrics.csv",
        help="Name of the CSV file inside each parameter folder."
    )

    args = parser.parse_args()

    summarize_reading_speed_grid(
        input_root=args.input_root,
        output_csv=args.output_csv,
        param_name=args.param_name,
        reading_speed_col=args.reading_speed_col,
        file_name=args.file_name,
    )