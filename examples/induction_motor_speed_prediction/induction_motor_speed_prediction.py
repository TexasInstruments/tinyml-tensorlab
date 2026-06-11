"""Induction Motor Speed Dataset Generator

This module generates a synthetic dataset modeling the relationship between
electrical parameters and induction motor speed. It provides small helper
functions to create randomized electrical/motor/operating parameters,
compute derived quantities (active/apparent power, synchronous speed,
slip, and motor speed), and to persist the dataset to CSV.

The code preserves the original behavior while improving readability,
adding type hints and docstrings, and using safe file/Path handling where
appropriate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


def set_random_seed(seed: int = 42) -> None:
    """Set the random seed for NumPy's global RNG.

    Args:
        seed: Integer seed for reproducibility.
    """

    np.random.seed(seed)


def generate_supply_parameters(n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate randomized electrical supply parameters.

    Returns voltage (V), current (A), frequency (Hz) and power factor arrays.
    """

    voltage = np.random.uniform(200, 460, n_samples)
    base_current = voltage / 100.0
    current = base_current * (1 + np.random.uniform(0.5, 1.5, n_samples))
    frequency = np.random.choice([50, 60], n_samples)
    power_factor = np.random.uniform(0.65, 0.95, n_samples)

    return voltage, current, frequency, power_factor


def generate_motor_parameters(n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate motor-specific randomized parameters.

    Returns poles, stator resistance (Ohm), and efficiency (percent).
    """

    poles = np.random.choice([2, 4, 6, 8], n_samples)
    stator_resistance = np.random.uniform(0.5, 5.0, n_samples)
    efficiency = np.random.uniform(75.0, 95.0, n_samples)

    return poles, stator_resistance, efficiency


def generate_operating_conditions(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate operating conditions: load torque and ambient temperature."""

    load_torque = np.random.uniform(10.0, 100.0, n_samples)
    ambient_temp = np.random.uniform(15.0, 45.0, n_samples)
    return load_torque, ambient_temp


def calculate_power_parameters(voltage: np.ndarray, current: np.ndarray, power_factor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute active and apparent power from electrical parameters."""

    active_power = voltage * current * power_factor
    apparent_power = voltage * current
    return active_power, apparent_power


def calculate_motor_speed(
    frequency: np.ndarray,
    poles: np.ndarray,
    load_torque: np.ndarray,
    voltage: np.ndarray,
    stator_resistance: np.ndarray,
    ambient_temp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate synchronous speed, slip and actual motor speed (rpm).

    The function uses a simple, physically-inspired model where slip
    increases with load and stator resistance and decreases with higher
    supply voltage. Slip is clipped to realistic bounds.
    """

    sync_speed = 120.0 * frequency / poles

    base_slip = 0.01 + 0.05 * (load_torque / 100.0) - 0.01 * (voltage / 460.0)
    slip = base_slip * (1.0 + 0.1 * stator_resistance) * (1.0 + 0.005 * (ambient_temp - 25.0))
    slip = np.clip(slip, 0.01, 0.2)

    motor_speed = sync_speed * (1.0 - slip)
    return sync_speed, slip, motor_speed


def create_motor_dataframe(
    voltage: np.ndarray,
    current: np.ndarray,
    frequency: np.ndarray,
    power_factor: np.ndarray,
    poles: np.ndarray,
    stator_resistance: np.ndarray,
    efficiency: np.ndarray,
    load_torque: np.ndarray,
    ambient_temp: np.ndarray,
    active_power: np.ndarray,
    apparent_power: np.ndarray,
    motor_speed: np.ndarray,
) -> pd.DataFrame:
    """Build a Pandas DataFrame containing all generated parameters.

    The function applies reasonable rounding for readability; this is a
    presentation choice and does not change the underlying numeric values
    used for model training if higher precision is required later.
    """

    df = pd.DataFrame(
        {
            "voltage_V": np.round(voltage, 1),
            "current_A": np.round(current, 2),
            "frequency_Hz": frequency,
            "power_factor": np.round(power_factor, 3),
            "stator_resistance_ohm": np.round(stator_resistance, 3),
            "load_torque_Nm": np.round(load_torque, 1),
            "ambient_temp_C": np.round(ambient_temp, 1),
            "poles": poles,
            "active_power_W": np.round(active_power),
            "apparent_power_VA": np.round(apparent_power),
            "efficiency_percent": np.round(efficiency, 1),
            "motor_speed_rpm": np.round(motor_speed, 1),
        }
    )
    return df


def save_dataframe(df: pd.DataFrame, filename: str = "induction_motor_speed_prediction.csv") -> None:
    """Persist the DataFrame to CSV.

    Preserves the original behavior (no index written).
    """

    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    print(f"Generated {len(df)} samples")


# ---------------------------------------------------------------------------
# High-level dataset pipeline and small utilities for splitting and zipping
# ---------------------------------------------------------------------------

DATASET_NAME = "induction_motor_speed_prediction"
DATASET_FILE_NAME = "induction_motor_speed_prediction.csv"
independent_variables = (
    "voltage_V,current_A,frequency_Hz,power_factor,stator_resistance_ohm,"
    "load_torque_Nm,ambient_temp_C,poles,active_power_W,apparent_power_VA,efficiency_percent"
).split(",")
dependent_variables = ["motor_speed_rpm"]

NUM_SPLITS = 10
rename_target: Dict[str, str] = {dependent_variables[0]: "Target"}
replace_target_values: Dict[Any, Any] = {}


def generate_motor_dataset(n_samples: int = 15000, seed: int = 42, filename: str = DATASET_FILE_NAME) -> pd.DataFrame:
    """Generate and save a full motor dataset.

    This is a convenience wrapper that runs the full generation pipeline and
    writes the resulting DataFrame to disk.
    """

    set_random_seed(seed)

    voltage, current, frequency, power_factor = generate_supply_parameters(n_samples)
    poles, stator_resistance, efficiency = generate_motor_parameters(n_samples)
    load_torque, ambient_temp = generate_operating_conditions(n_samples)

    active_power, apparent_power = calculate_power_parameters(voltage, current, power_factor)
    _, _, motor_speed = calculate_motor_speed(
        frequency, poles, load_torque, voltage, stator_resistance, ambient_temp
    )

    motor_df = create_motor_dataframe(
        voltage,
        current,
        frequency,
        power_factor,
        poles,
        stator_resistance,
        efficiency,
        load_torque,
        ambient_temp,
        active_power,
        apparent_power,
        motor_speed,
    )

    save_dataframe(motor_df, filename)
    return motor_df


def load_data() -> pd.DataFrame:
    """Load the generated CSV and apply basic column selection/renaming."""

    data = pd.read_csv(DATASET_FILE_NAME, index_col=False)
    columns = independent_variables + dependent_variables
    data = data[columns]
    data.rename(columns=rename_target, inplace=True)
    data["Target"] = data["Target"].replace(replace_target_values)
    return data


def store_datafiles(df: pd.DataFrame) -> None:
    """Split the DataFrame into `NUM_SPLITS` chunk files and append annotation lists."""

    dir_path = Path("files")
    annotations_path = Path("annotations")
    dir_path.mkdir(exist_ok=True)
    annotations_path.mkdir(exist_ok=True)

    chunk_size = len(df) // NUM_SPLITS
    chunks: List[pd.DataFrame] = [df[i * chunk_size : (i + 1) * chunk_size] for i in range(NUM_SPLITS)]

    for idx, chunk in enumerate(chunks):
        rand_number = np.random.rand()
        file_type = "test" if (rand_number < 0.15) else "val" if (rand_number < 0.3) else "train"

        file_list = annotations_path / "file_list.txt"
        with file_list.open("a") as f:
            f.write(f"class_{idx}_{file_type}.csv\n")

        instances_list = annotations_path / f"instances_{file_type}_list.txt"
        with instances_list.open("a") as f:
            f.write(f"class_{idx}_{file_type}.csv\n")

        out_file = dir_path / f"class_{idx}_{file_type}.csv"
        chunk.to_csv(out_file, index=False)
        print(f"Created {out_file}")


def cleanup() -> None:
    """Remove temporary files/directories created by this pipeline (shell-backed)."""

    os.system(f"rm -rf {DATASET_NAME} > /dev/null 2>&1")
    os.system(f"rm {DATASET_NAME}.zip > /dev/null 2>&1")


def zip_files() -> None:
    """Archive `files` and `annotations` into a zip and remove the originals."""

    print(f"Zipping the classes into {DATASET_NAME}_dataset.zip")
    os.system(f"zip -r {DATASET_NAME}_dataset.zip files annotations > /dev/null 2>&1")
    os.system("rm -rf files annotations > /dev/null 2>&1")


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a small human-readable summary of the generated dataset."""

    print("\nInduction Motor Dataset Summary:")
    print("-" * 40)
    print(f"Number of samples: {len(df)}")
    print(f"Speed range: {df['motor_speed_rpm'].min():.1f} to {df['motor_speed_rpm'].max():.1f} RPM")
    print(f"Voltage range: {df['voltage_V'].min():.1f} to {df['voltage_V'].max():.1f} V")
    print(f"Current range: {df['current_A'].min():.2f} to {df['current_A'].max():.2f} A")
    print(f"Frequency options: {', '.join(map(str, sorted(df['frequency_Hz'].unique())))} Hz")
    print(f"Pole options: {', '.join(map(str, sorted(df['poles'].unique())))}")
    print("-" * 40)
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    motor_dataset = generate_motor_dataset(n_samples=15000)
    print_dataset_summary(motor_dataset)

    df = load_data()
    store_datafiles(df)
    zip_files()
    cleanup()