import os
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter


def filter_force_signal(signal_data, dt, median_window_sec=0.04, cutoff_hz=6.0):
    """
    Apply same filters.
    """
    arr = np.asarray(signal_data, dtype=float)
    if len(arr) < 15 or dt <= 0:
        return arr

    # Median filter
    win_size = max(3, int(round(median_window_sec / dt)))
    if win_size % 2 == 0:
        win_size += 1
    arr_med = median_filter(arr, size=win_size, mode="nearest")

    # Butterworth filter
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if cutoff_hz < nyq:
        b, a = butter(N=3, Wn=cutoff_hz / nyq, btype="low")
        padlen = min(len(arr_med) - 1, 3 * max(len(a), len(b)))
        arr_smooth = filtfilt(b, a, arr_med, padlen=padlen)
        return np.maximum(0.0, arr_smooth)

    return arr_med


def get_steady_state_mask(t, f_target, settling_window_sec=1.0):
    """
    Build a mask that excludes transient windows.
    """
    step_indices = np.where(np.abs(np.diff(f_target, prepend=f_target[0])) > 1e-3)[0]
    step_times = np.concatenate(([t[0]], t[step_indices]))

    ss_mask = np.ones_like(t, dtype=bool)
    for t_step in step_times:
        ss_mask &= ~((t >= t_step) & (t < t_step + settling_window_sec))

    return ss_mask


def analyze_force_csv(filepath, case_label, settling_window_sec=1.0):
    """
    Load a simulation CSV file and compute tracking and sensor estimation errors.
    """
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: '{filepath}'")
        return

    # Load columns
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    t = data[:, 0]
    f_target = data[:, 1]
    f_calc_raw = data[:, 2]
    f_direct_raw = data[:, 3]

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.002

    # Filters
    f_calc = filter_force_signal(f_calc_raw, dt)
    f_direct = filter_force_signal(f_direct_raw, dt)

    # Global tracking error (entire contact duration)
    err_global = f_target - f_calc
    mae_global = np.mean(np.abs(err_global))
    rmse_global = np.sqrt(np.mean(err_global**2))
    max_err_global = np.max(np.abs(err_global))

    # Steady-state tracking error (excluding step transients)
    ss_mask = get_steady_state_mask(t, f_target, settling_window_sec)
    if np.any(ss_mask):
        err_ss = f_target[ss_mask] - f_calc[ss_mask]
        mae_ss = np.mean(np.abs(err_ss))
        rmse_ss = np.sqrt(np.mean(err_ss**2))
        bias_ss = np.mean(err_ss)
        valid_rel = f_target[ss_mask] > 0.1
        rel_err_ss = (
            np.mean(np.abs(err_ss[valid_rel]) / f_target[ss_mask][valid_rel]) * 100.0
            if np.any(valid_rel)
            else 0.0
        )
    else:
        mae_ss = rmse_ss = bias_ss = rel_err_ss = np.nan

    # Wrist sensor estimation vs. ground truth tip force
    err_sensor = f_direct - f_calc
    mae_sensor = np.mean(np.abs(err_sensor))
    rmse_sensor = np.sqrt(np.mean(err_sensor**2))

    # Print formatted summary
    print("=" * 68)
    print(f"{case_label} ({os.path.basename(filepath)})")
    print(f"Duration: {t[-1] - t[0]:.2f} s | Samples: {len(t)} | dt: {dt*1000:.2f} ms")
    print("-" * 68)
    print("1. Global Tracking Error (Desired vs. Wrist Sensor):")
    print(f"   - MAE:  {mae_global:.4f} N")
    print(f"   - RMSE: {rmse_global:.4f} N")
    print(f"   - Max Abs Error (includes step jumps): {max_err_global:.4f} N")
    print(f"2. Steady-State Error (excluding {settling_window_sec:.1f} s after each step):")
    print(f"   - SS MAE:      {mae_ss:.4f} N")
    print(f"   - SS RMSE:     {rmse_ss:.4f} N")
    print(f"   - SS Mean Bias (Fd - Fcalc): {bias_ss:+.4f} N")
    print(f"   - SS Relative Error:         {rel_err_ss:.2f} %")
    print("3. Wrist Sensor vs. Ground Truth Tip Force:")
    print(f"   - Sensor MAE:  {mae_sensor:.4f} N")
    print(f"   - Sensor RMSE: {rmse_sensor:.4f} N")

    # Per-segment breakdown
    change_idx = np.where(np.abs(np.diff(f_target)) > 1e-3)[0] + 1
    segments = np.split(np.arange(len(t)), change_idx)
    if len(segments) > 1:
        print("4. Steady-State Breakdown per Force Step:")
        for idx_seg, seg in enumerate(segments, start=1):
            t_seg = t[seg]
            fd_val = f_target[seg[0]]
            seg_ss = seg[t_seg >= (t_seg[0] + settling_window_sec)]
            if len(seg_ss) > 0:
                f_mean = np.mean(f_calc[seg_ss])
                e_seg = fd_val - f_mean
                print(
                    f"   - Step {idx_seg}: Fd = {fd_val:5.2f} N | "
                    f"Mean Fcalc = {f_mean:5.2f} N | SS Error = {e_seg:+.4f} N"
                )
    print("=" * 68 + "\n")


if __name__ == "__main__":
    # Resolve OPENARM_WS root directory (3 levels up from src/openarm_control/openarm_control/)
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    cases = [
        ("CASE 1: Fixed Position - Force Steps", os.path.join(ws_root, "force_data_raw.csv")),
        ("CASE 2: Tangential Sweep - Constant Force", os.path.join(ws_root, "force_data_raw_2.csv")),
        ("CASE 3: Combined Clinical Maneuver", os.path.join(ws_root, "force_data_raw_3.csv")),
    ]

    for label, csv_path in cases:
        analyze_force_csv(csv_path, label, settling_window_sec=1.0)