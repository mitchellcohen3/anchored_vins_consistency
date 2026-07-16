"""Table generation utilities for VINS evaluation."""

import logging
import os
import typing

import numpy as np
import pandas as pd


def create_rpe_table(
    rpe_results_by_segment: typing.Dict[typing.Any, typing.Dict[str, typing.Dict[str, float]]],
    ori_key: str = "mean_ori",
    pos_key: str = "mean_pos",
    bold_best: bool = False,
    algorithm_order: typing.Optional[typing.List[str]] = None,
    precision: int = 3,
) -> str:
    """Creates a LaTeX RPE table with algorithms as rows and segment lengths as columns.

    Each cell contains the mean orientation / mean position RPE separated by a slash.

    Parameters
    ----------
    rpe_results_by_segment : Dict[segment_length, Dict[algorithm_name, Dict[metric, value]]]
        RPE results keyed first by segment length, then by algorithm name.
    ori_key : str
        Key for mean orientation RPE in the inner dict. Default is "mean_ori".
    pos_key : str
        Key for mean position RPE in the inner dict. Default is "mean_pos".
    bold_best : bool
        If True, bold the best (lowest) value per segment per metric.
    algorithm_order : list of str, optional
        Explicit ordering for algorithm rows. Algorithms not listed are appended sorted.
    precision : int
        Number of decimal places for numeric values.

    Returns
    -------
    str
        LaTeX tabular string.
    """
    all_algorithms = {
        algo
        for seg_results in rpe_results_by_segment.values()
        for algo in seg_results
    }
    if algorithm_order is not None:
        ordered = [a for a in algorithm_order if a in all_algorithms]
        remaining = sorted(all_algorithms - set(ordered))
        algorithms = ordered + remaining
    else:
        algorithms = sorted(all_algorithms)

    segments = sorted(rpe_results_by_segment.keys())

    # Collect (ori, pos) pairs; None if algorithm missing for a segment
    data = {
        seg: {
            algo: (
                (rpe_results_by_segment[seg][algo][ori_key],
                 rpe_results_by_segment[seg][algo][pos_key])
                if algo in rpe_results_by_segment[seg] else None
            )
            for algo in algorithms
        }
        for seg in segments
    }

    # Find per-segment best values for bolding
    best: typing.Dict[typing.Any, typing.Tuple] = {}
    if bold_best:
        for seg in segments:
            vals = [v for v in data[seg].values() if v is not None]
            best[seg] = (
                min(v[0] for v in vals) if vals else None,
                min(v[1] for v in vals) if vals else None,
            )

    def fmt_pair(ori: float, pos: float, best_ori, best_pos) -> str:
        if bold_best:
            ori_str = (f"\\textbf{{{ori:.{precision}f}}}"
                       if best_ori is not None and abs(ori - best_ori) < 1e-9
                       else f"{ori:.{precision}f}")
            pos_str = (f"\\textbf{{{pos:.{precision}f}}}"
                       if best_pos is not None and abs(pos - best_pos) < 1e-9
                       else f"{pos:.{precision}f}")
        else:
            ori_str, pos_str = f"{ori:.{precision}f}", f"{pos:.{precision}f}"
        return f"{ori_str} / {pos_str}"

    def fmt_cell(seg, algo: str) -> str:
        val = data[seg][algo]
        if val is None:
            return "-"
        best_ori, best_pos = best.get(seg, (None, None))
        return fmt_pair(val[0], val[1], best_ori, best_pos)

    def esc(s: str) -> str:
        return str(s).replace("_", r"\_")

    col_fmt = "l" + "c" * len(segments)
    header_segments = " & ".join(f"{seg}m" for seg in segments)
    lines = [
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        f"Algorithm & {header_segments}" + r" \\",
        "\\midrule",
    ]
    for algo in algorithms:
        cells = [esc(algo)] + [fmt_cell(seg, algo) for seg in segments]
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    return "\n".join(lines)

def create_ate_table(
    mc_results_by_dataset: typing.Dict[str, typing.Dict[str, typing.Any]],
    pos_key: str = "ate_pos",
    att_key: str = "ate_ori",
    bold_best: bool = False,
    include_average: bool = False,
    algorithm_order: typing.Optional[typing.List[str]] = None,
) -> str:
    """Creates a LaTeX ATE table with algorithms as rows and datasets as columns.

    Each cell contains the rotational / positional ATE separated by a slash.

    Parameters
    ----------
    mc_results_by_dataset : Dict[dataset_name, Dict[algorithm_name, Dict[metric, value]]]
        Results of multiple algorithms on multiple datasets.
    pos_key : str
        Key for position ATE in the results dict. Default is "ate_pos".
    att_key : str
        Key for attitude ATE in the results dict. Default is "ate_ori".
    bold_best : bool
        If True, bold the best (lowest) value per dataset per metric.
    include_average : bool
        If True, append an "Average" column with the mean ATE across all datasets
        for each algorithm (missing datasets are excluded from the mean).
    algorithm_order : list of str, optional
        Explicit ordering for algorithm rows. If provided, rows appear in this order;
        any algorithms not listed are appended sorted at the end.

    Returns
    -------
    str
        LaTeX tabular string.
    """
    all_algorithms = {
        algo
        for dataset_results in mc_results_by_dataset.values()
        for algo in dataset_results
    }
    if algorithm_order is not None:
        ordered = [a for a in algorithm_order if a in all_algorithms]
        remaining = sorted(all_algorithms - set(ordered))
        algorithms = ordered + remaining
    else:
        algorithms = sorted(all_algorithms)
    dataset_names = sorted(mc_results_by_dataset.keys())

    # Collect (att, pos) pairs; None if algorithm missing for a dataset
    data = {
        dataset: {
            algo: (
                (mc_results_by_dataset[dataset][algo][att_key],
                 mc_results_by_dataset[dataset][algo][pos_key])
                if algo in mc_results_by_dataset[dataset] else None
            )
            for algo in algorithms
        }
        for dataset in dataset_names
    }

    # Compute per-algorithm averages across datasets
    averages: typing.Dict[str, typing.Optional[typing.Tuple[float, float]]] = {}
    if include_average:
        for algo in algorithms:
            vals = [data[d][algo] for d in dataset_names if data[d][algo] is not None]
            averages[algo] = (
                (float(np.mean([v[0] for v in vals])),
                 float(np.mean([v[1] for v in vals])))
                if vals else None
            )

    # Find per-dataset best values for bolding
    best = {}
    if bold_best:
        for dataset in dataset_names:
            vals = [v for v in data[dataset].values() if v is not None]
            best[dataset] = (
                min(v[0] for v in vals) if vals else None,
                min(v[1] for v in vals) if vals else None,
            )

    # Find best average values for bolding
    best_avg: typing.Optional[typing.Tuple[float, float]] = None
    if bold_best and include_average:
        avg_vals = [v for v in averages.values() if v is not None]
        best_avg = (
            min(v[0] for v in avg_vals) if avg_vals else None,
            min(v[1] for v in avg_vals) if avg_vals else None,
        )

    def fmt_pair(att: float, pos: float, best_att: float, best_pos: float) -> str:
        if bold_best:
            att_str = (f"\\textbf{{{att:.3f}}}"
                       if best_att is not None and abs(att - best_att) < 1e-9
                       else f"{att:.3f}")
            pos_str = (f"\\textbf{{{pos:.3f}}}"
                       if best_pos is not None and abs(pos - best_pos) < 1e-9
                       else f"{pos:.3f}")
        else:
            att_str, pos_str = f"{att:.3f}", f"{pos:.3f}"
        return f"{att_str} / {pos_str}"

    def fmt_cell(dataset: str, algo: str) -> str:
        val = data[dataset][algo]
        if val is None:
            return "-"
        att, pos = val
        best_att, best_pos = best.get(dataset, (None, None))
        return fmt_pair(att, pos, best_att, best_pos)

    def fmt_avg_cell(algo: str) -> str:
        val = averages.get(algo)
        if val is None:
            return "-"
        att, pos = val
        best_att = best_avg[0] if best_avg else None
        best_pos = best_avg[1] if best_avg else None
        return fmt_pair(att, pos, best_att, best_pos)

    def esc(s: str) -> str:
        return s.replace("_", r"\_")

    extra_cols = 1 if include_average else 0
    col_fmt = "l" + "c" * (len(dataset_names) + extra_cols)
    header_datasets = " & ".join(esc(d) for d in dataset_names)
    header_avg = " & Average" if include_average else ""
    lines = [
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        f"Algorithm & {header_datasets}{header_avg}" + r" \\",
        "\\midrule",
    ]
    for algo in algorithms:
        cells = [esc(algo)] + [fmt_cell(d, algo) for d in dataset_names]
        if include_average:
            cells.append(fmt_avg_cell(algo))
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    return "\n".join(lines)


def create_combined_ate_table(
    mono_results: typing.Dict[str, typing.Dict[str, typing.Any]],
    stereo_results: typing.Dict[str, typing.Dict[str, typing.Any]],
    pos_key: str = "ate_pos",
    att_key: str = "ate_ori",
    bold_best: bool = False,
    include_average: bool = False,
    algorithm_order: typing.Optional[typing.List[str]] = None,
) -> str:
    """Creates a single LaTeX ATE table with monocular and stereo results as two sections.

    The table has the same column structure as :func:`create_ate_table` but contains
    two groups of rows separated by a labelled divider: one group for monocular results
    and one for stereo results. Bolding (when enabled) is applied independently within
    each section so the best mono algorithm and best stereo algorithm are both visible.

    Parameters
    ----------
    mono_results, stereo_results : Dict[dataset_name, Dict[algorithm_name, Dict[metric, value]]]
        Results for the monocular and stereo configurations respectively, in the same
        format accepted by :func:`create_ate_table`.
    pos_key, att_key : str
        Keys for position and attitude ATE in the inner result dicts.
    bold_best : bool
        If True, bold the best (lowest) value per dataset within each section.
    include_average : bool
        If True, append an "Average" column for each row.
    algorithm_order : list of str, optional
        Explicit algorithm row ordering (same order applied in both sections).

    Returns
    -------
    str
        LaTeX tabular string.
    """
    # Determine unified algorithm and dataset lists
    all_algorithms = {
        algo
        for results in (mono_results, stereo_results)
        for ds_results in results.values()
        for algo in ds_results
    }
    if algorithm_order is not None:
        ordered = [a for a in algorithm_order if a in all_algorithms]
        remaining = sorted(all_algorithms - set(ordered))
        algorithms = ordered + remaining
    else:
        algorithms = sorted(all_algorithms)

    all_datasets = sorted(
        set(mono_results.keys()) | set(stereo_results.keys())
    )

    def _build_data(results):
        return {
            ds: {
                algo: (
                    (results[ds][algo][att_key], results[ds][algo][pos_key])
                    if ds in results and algo in results[ds]
                    else None
                )
                for algo in algorithms
            }
            for ds in all_datasets
        }

    def _compute_averages(data):
        avgs = {}
        for algo in algorithms:
            vals = [data[d][algo] for d in all_datasets if data[d][algo] is not None]
            avgs[algo] = (
                (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
                if vals else None
            )
        return avgs

    def _best_per_dataset(data):
        best = {}
        for ds in all_datasets:
            vals = [v for v in data[ds].values() if v is not None]
            best[ds] = (
                min(v[0] for v in vals) if vals else None,
                min(v[1] for v in vals) if vals else None,
            )
        return best

    mono_data = _build_data(mono_results)
    stereo_data = _build_data(stereo_results)
    mono_avgs = _compute_averages(mono_data) if include_average else {}
    stereo_avgs = _compute_averages(stereo_data) if include_average else {}
    mono_best = _best_per_dataset(mono_data) if bold_best else {}
    stereo_best = _best_per_dataset(stereo_data) if bold_best else {}

    def _best_avg(avgs):
        vals = [v for v in avgs.values() if v is not None]
        return (
            (min(v[0] for v in vals), min(v[1] for v in vals)) if vals else (None, None)
        )

    mono_best_avg = _best_avg(mono_avgs) if (bold_best and include_average) else (None, None)
    stereo_best_avg = _best_avg(stereo_avgs) if (bold_best and include_average) else (None, None)

    def fmt_pair(att, pos, best_att, best_pos):
        def fv(v, bv):
            s = f"{v:.3f}"
            return f"\\textbf{{{s}}}" if bold_best and bv is not None and abs(v - bv) < 1e-9 else s
        return f"{fv(att, best_att)} / {fv(pos, best_pos)}"

    def fmt_cell(data, best, ds, algo):
        val = data[ds][algo]
        if val is None:
            return "-"
        b_att, b_pos = best.get(ds, (None, None))
        return fmt_pair(val[0], val[1], b_att, b_pos)

    def fmt_avg_cell(avgs, best_avg, algo):
        val = avgs.get(algo)
        if val is None:
            return "-"
        return fmt_pair(val[0], val[1], best_avg[0], best_avg[1])

    def esc(s):
        return s.replace("_", r"\_")

    def build_rows(data, avgs, best, best_avg):
        rows = []
        for algo in algorithms:
            cells = [esc(algo)] + [fmt_cell(data, best, ds, algo) for ds in all_datasets]
            if include_average:
                cells.append(fmt_avg_cell(avgs, best_avg, algo))
            rows.append(" & ".join(cells) + r" \\")
        return rows

    n_cols = 1 + len(all_datasets) + (1 if include_average else 0)
    col_fmt = "l" + "c" * (n_cols - 1)
    header_datasets = " & ".join(esc(d) for d in all_datasets)
    header_avg = " & Average" if include_average else ""

    lines = [
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        f"Algorithm & {header_datasets}{header_avg}" + r" \\",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Monocular}}}}" + r" \\",
        "\\midrule",
    ]
    lines += build_rows(mono_data, mono_avgs, mono_best, mono_best_avg)
    lines += [
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Stereo}}}}" + r" \\",
        "\\midrule",
    ]
    lines += build_rows(stereo_data, stereo_avgs, stereo_best, stereo_best_avg)
    lines += ["\\bottomrule", "\\end{tabular}"]

    return "\n".join(lines)


def create_alg_comparison_table(
    algorithm_names: typing.List[str],
    attitude_ates: typing.List[float],
    position_ates: typing.List[float],
    save_dir: str = None,
) -> pd.DataFrame:
    """Generate a comparison table of the attitude and position ATE for multiple algorithms.

    Parameters
    ----------
    algorithm_names : list of str
        Names of the algorithms being compared.
    attitude_ates : list of float
        Attitude ATE values in radians.
    position_ates : list of float
        Position ATE values in meters.
    save_dir : str, optional
        If provided, save the table as CSV to this directory.

    Returns
    -------
    pd.DataFrame
        Comparison table with algorithms and their ATEs.
    """
    if not (len(algorithm_names) == len(position_ates) == len(attitude_ates)):
        raise ValueError("All input lists must have the same length")

    # Convert attitude errors from radians to degrees
    attitude_ates_deg = [np.rad2deg(att) for att in attitude_ates]

    data = {
        "Algorithm": algorithm_names,
        "Mean Attitude ATE (deg)": attitude_ates_deg,
        "Mean Position ATE (m)": position_ates,
    }

    df = pd.DataFrame(data)

    # Optionally save to file
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "summary_table.csv")
        df.to_csv(save_path, index=False)
        print(f"Table saved to: {save_path}")

    return df


def generate_rmse_nees_comparison_table(
    results: typing.Dict[str, typing.Dict[str, float]],
    att_rmse_key: str = "att_ate",
    pos_rmse_key: str = "pos_ate",
    att_nees_key: str = "att_nees",
    pos_nees_key: str = "pos_nees",
    precision: int = 3,
    save_path: str = None,
) -> str:
    """Generate a LaTeX comparison table of average RMSE and NEES across algorithms.

    Each row corresponds to one algorithm. The best (lowest) RMSE values are bolded.

    Parameters
    ----------
    results : dict mapping algorithm name -> dict of metric values
        Keys of the inner dict should include att_rmse, pos_rmse, att_nees, pos_nees
        (or whatever keys are specified via the *_key parameters). Orientation RMSE
        should be in degrees; position RMSE in meters.
    att_rmse_key : str
        Key for average orientation RMSE in the inner dict.
    pos_rmse_key : str
        Key for average position RMSE in the inner dict.
    att_nees_key : str
        Key for average orientation NEES in the inner dict.
    pos_nees_key : str
        Key for average position NEES in the inner dict.
    precision : int
        Number of decimal places for numeric values.
    save_path : str, optional
        If provided, the LaTeX table string is saved to this file.

    Returns
    -------
    str
        LaTeX table as a string.
    """
    # Collect values
    data = []
    for alg_name, metrics in results.items():
        data.append(
            {
                "label": alg_name,
                "att_rmse": metrics.get(att_rmse_key, float("inf")),
                "pos_rmse": metrics.get(pos_rmse_key, float("inf")),
                "att_nees": metrics.get(att_nees_key, float("nan")),
                "pos_nees": metrics.get(pos_nees_key, float("nan")),
            }
        )

    # Find best (lowest) RMSE values for bolding
    finite_att = [d["att_rmse"] for d in data if np.isfinite(d["att_rmse"])]
    finite_pos = [d["pos_rmse"] for d in data if np.isfinite(d["pos_rmse"])]
    best_att_rmse = min(finite_att) if finite_att else None
    best_pos_rmse = min(finite_pos) if finite_pos else None

    def fmt_rmse(val, best):
        if not np.isfinite(val):
            return "-"
        formatted = f"{val:.{precision}f}"
        if best is not None and abs(val - best) < 1e-9:
            formatted = f"\\textbf{{{formatted}}}"
        return formatted

    def fmt_nees(val):
        if not np.isfinite(val):
            return "-"
        return f"{val:.{precision}f}"

    col_headers = ["Algorithm", "Mean ATE (deg / m)", "Mean NEES (att / pos)"]
    col_fmt = "c" + "c" * (len(col_headers) - 1)

    lines = []
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(col_headers) + " \\\\")
    lines.append("\\midrule")
    for d in data:
        ate_cell = f"{fmt_rmse(d['att_rmse'], best_att_rmse)} / {fmt_rmse(d['pos_rmse'], best_pos_rmse)}"
        nees_cell = f"{fmt_nees(d['att_nees'])} / {fmt_nees(d['pos_nees'])}"
        lines.append(f"{d['label']} & {ate_cell} & {nees_cell} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table_str = "\n".join(lines)

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(table_str)
        logging.info(f"Saved RMSE/NEES comparison table to {save_path}")

    return table_str


def generate_ate_comparison_table(
    ate_df: pd.DataFrame,
    position_col: str = "position_ate",
    attitude_col: str = "attitude_ate",
    algorithm_col: str = "algorithm",
    output_format: str = "markdown",
    precision: int = 3,
    position_unit: str = "m",
    attitude_unit: str = "deg",
    highlight_best: bool = True,
    save_path: str = None,
) -> str:
    """Generate a formatted comparison table of position and attitude ATE across algorithms.

    Parameters
    ----------
    ate_df : pd.DataFrame
        DataFrame containing ATE values for different algorithms. Must contain
        columns for algorithm names, position ATE, and attitude ATE.
    position_col : str
        Column name for position ATE values.
    attitude_col : str
        Column name for attitude ATE values.
    algorithm_col : str
        Column name for algorithm identifiers.
    output_format : str
        Output format: "markdown", "latex", "csv", or "console".
    precision : int
        Number of decimal places for ATE values.
    position_unit : str
        Unit label for position ATE (e.g., "m", "cm").
    attitude_unit : str
        Unit label for attitude ATE (e.g., "deg", "rad").
    highlight_best : bool
        If True, highlight the best (lowest) values in the table.
        For LaTeX, uses bold. For markdown, uses bold (**).
    save_path : str
        If provided, save the table to this file path.

    Returns
    -------
    str
        Formatted table as a string.
    """
    if algorithm_col not in ate_df.columns:
        raise ValueError(f"Algorithm column '{algorithm_col}' not found in DataFrame")
    if position_col not in ate_df.columns:
        raise ValueError(f"Position ATE column '{position_col}' not found in DataFrame")
    if attitude_col not in ate_df.columns:
        raise ValueError(f"Attitude ATE column '{attitude_col}' not found in DataFrame")

    df = ate_df.copy()

    # Find best values for highlighting
    best_pos_idx = df[position_col].idxmin() if highlight_best else None
    best_att_idx = df[attitude_col].idxmin() if highlight_best else None

    # Format values with optional std
    def format_value(val, is_best=False):
        formatted = f"{val:.{precision}f}"

        if is_best and highlight_best:
            if output_format == "latex":
                formatted = f"\\textbf{{{formatted}}}"
            elif output_format == "markdown":
                formatted = f"**{formatted}**"
        return formatted

    # Build the output table
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "Algorithm": row[algorithm_col],
                f"Position ATE ({position_unit})": format_value(
                    row[position_col], idx == best_pos_idx
                ),
                f"Attitude ATE ({attitude_unit})": format_value(
                    row[attitude_col], idx == best_att_idx
                ),
            }
        )

    result_df = pd.DataFrame(rows)

    # Generate output in the requested format
    if output_format == "latex":
        table_str = result_df.to_latex(index=False, escape=False)
    elif output_format == "markdown":
        table_str = result_df.to_markdown(index=False)
    elif output_format == "csv":
        table_str = result_df.to_csv(index=False)
    elif output_format == "console":
        table_str = result_df.to_string(index=False)
    else:
        raise ValueError(f"Unknown output format: {output_format}")

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(table_str)
        logging.info(f"Saved ATE comparison table to {save_path}")

    return table_str
