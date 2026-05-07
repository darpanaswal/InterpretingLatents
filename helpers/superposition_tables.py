# superposition_analyse.py
import json
import argparse
import math
from pathlib import Path
from src.config import CONTROL_EXPT

_MODES_ORDER = ["base", "cot", "pause", "coconut", "coconut_u", "codi"]
_LABELS_SHORT = {
    "base": "B", "cot": "CoT", "pause": "PaT", "coconut": "C",
    "coconut_u": "C$_u$", "codi": "CODI",
}
_LABELS_LONG = {
    "base": "Base", "cot": "CoT", "pause": "Pause", "coconut": "Coconut",
    "coconut_u": "Coconut$_{u}$", "codi": "CODI",
}

def _load_all_results(task):
    results_dir = CONTROL_EXPT
    results = {}
    if not results_dir.exists():
        return results, results_dir
    for f in sorted(results_dir.glob("results_*.json")):
        mode = f.stem.replace("results_", "")
        with open(f) as fh:
            results[mode] = json.load(fh)
    return results, results_dir

def fmt_float(val, decimals=2):
    if math.isnan(val): return "-"
    # Mathematically rounds to specified decimal places
    s = f"{val:.{decimals}f}"
    # Remove leading zero for values < 1 and > -1
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s

def fmt_pct(val):
    if math.isnan(val): return "-"
    # Rounds to the nearest integer and removes decimals
    return f"{round(val * 100)}"

def generate_main_table(results, modes, all_k):
    num_m = len(modes)
    cols = "l " + " ".join(["c" * num_m] * 4)
    
    latex = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\tiny",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{$P(\\text{correct})$ is the raw (unnormalized) joint probability summed over correct candidates. Candidate mass is the total raw probability assigned to all candidates at the depth frontier. Values $<0.01$ for PaT and C at $k{=}0{-}2$ reflect that these models were trained to reason through latent recurrence and assign near-zero probability to any concept before the recurrence has begun.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "& \\multicolumn{" + str(num_m) + "}{c}{$H/\\log_2 N$} & \\multicolumn{" + str(num_m) + "}{c}{Top-1 correct (\\%)} & \\multicolumn{" + str(num_m) + "}{c}{$P(\\text{correct})$} & \\multicolumn{" + str(num_m) + "}{c}{Cand.\\ mass} \\\\"
    ]
    
    cmidrules = []
    start = 2
    for _ in range(4):
        end = start + num_m - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    latex.append(" ".join(cmidrules))
    
    headers = ["$k$"] + [_LABELS_SHORT[m] for m in modes] * 4
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule")
    
    for k in all_k:
        row = [str(k)]
        
        for m in modes:
            v = results[m]["summary"].get(str(k), {}).get("mean_normalized_entropy", float("nan"))
            row.append(fmt_float(v, 2))
            
        for m in modes:
            v = results[m]["summary"].get(str(k), {}).get("top1_correct_frac", float("nan"))
            row.append(fmt_pct(v))
            
        for m in modes:
            v = results[m]["summary"].get(str(k), {}).get("mean_value_correct", float("nan"))
            row.append(fmt_float(v, 2))
            
        for m in modes:
            v = results[m]["summary"].get(str(k), {}).get("mean_candidate_mass", float("nan"))
            row.append(fmt_float(v, 2))
            
        latex.append(" & ".join(row) + " \\\\")
        latex.append("")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:standard_probing}",
        "\\end{table*}",
        ""
    ])
    return "\n".join(latex)

def generate_table9(results, modes, all_k):
    num_m = len(modes)
    cols = "l " + " ".join(["ccc"] * num_m)
    
    latex = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\small",
        "\setlength{\tabcolsep}{4pt}",
        "\\caption{Cumulative top-$k$ raw probabilities. T1 = top-1, T3 = top-3, $\Delta$ = T3 $-$ T1. Coconut $u{=}0.0$ and Pause-as-thought show negligible gaps throughout, concentrating mass on a single candidate. Coconut $u{=}0.3$ shows substantial gaps at shallow $k$ (broad exploration) that narrow with depth.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
    ]
    
    h1 = [""] + [f"\\multicolumn{{3}}{{c}}{{{_LABELS_LONG[m]}}}" for m in modes]
    latex.append(" & ".join(h1) + " \\\\")
    
    cmidrules = []
    start = 2
    for _ in range(num_m):
        end = start + 2
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    latex.append(" ".join(cmidrules))
    
    h2 = ["$k$"] + ["T1 & T3 & $\\Delta$"] * num_m
    latex.append(" & ".join(h2) + " \\\\")
    latex.append("\\midrule")
    
    for k in all_k[:5]:
        row = [str(k)]
        for m in modes:
            s = results[m]["summary"].get(str(k), {})
            t1 = s.get("mean_top1_prob", float("nan"))
            t3 = s.get("mean_top3_cumul", float("nan"))
            if math.isnan(t1):
                row.append("- & - & -")
            else:
                delta = max(0, t3 - t1)
                row.append(f"{fmt_float(t1, 3)} & {fmt_float(t3, 3)} & {fmt_float(delta, 3)}")
        latex.append(" & ".join(row) + " \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:candidate_parallelism}",
        "\\end{table*}",
        ""
    ])
    return "\n".join(latex)

def generate_table10(results, modes):
    latex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\small",
        "\\caption{Convergence from $k{=}0$ to $k{=}4$. Pause-as-thought and Coconut $u{=}0.0$ show increasing P(correct) with decreasing entropy (exploration to convergence). Base, CoT, and Coconut $u{=}0.3$ show degrading or flat P(correct).}",
        "\\begin{tabular}{l cc cc}",
        "\\toprule",
        "& \\multicolumn{2}{c}{$P(\\text{correct})$} & \\multicolumn{2}{c}{$H/\\log_2 N$} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "Condition & $k{=}0$ & $k{=}4$ & $k{=}0$ & $k{=}4$ \\\\",
        "\\midrule"
    ]
    
    for m in modes:
        s0 = results[m]["summary"].get("0", {})
        s4 = results[m]["summary"].get("4", {})
        
        pc0 = s0.get("mean_value_correct", float("nan"))
        pc4 = s4.get("mean_value_correct", float("nan"))
        h0 = s0.get("mean_normalized_entropy", float("nan"))
        h4 = s4.get("mean_normalized_entropy", float("nan"))
        
        row = [
            _LABELS_LONG[m],
            fmt_float(pc0, 2),
            fmt_float(pc4, 2),
            fmt_float(h0, 2),
            fmt_float(h4, 2)
        ]
        latex.append(" & ".join(row) + " \\\\")
        
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:convergence_trajectories}",
        "\\end{table}",
        ""
    ])
    return "\n".join(latex)

def generate_table11(results, modes):
    num_m = len(modes)
    cols = "l " + "c"*num_m
    
    latex = [
        "\\begin{table*}[h!]",
        "\\centering",
        "\\small",
        "\\caption{Per-instance analysis. ``P(correct) increases'' counts instances where the raw probability summed over correct candidates is higher at the final $k$ evaluated for that instance than at $k{=}0$. The close match between Coconut $u{=}0.0$ (61.5\%) and Pause-as-thought (62.5\%) reinforces that the curriculum drives the convergence pattern.}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & " + " & ".join([_LABELS_LONG[m] for m in modes]) + " \\\\",
        "\\midrule"
    ]
    
    row_inc = ["P(correct) increases ($k{=}0 \\to k_{\\text{last}}$)"]
    row_top1 = ["Top-1 correct at final $k$"]
    row_ent = ["High entropy at $k{=}0$ ($H/\\log_2 N > 0.5$)"]
    row_n = ["$n$"]
    
    for m in modes:
        per_inst = results[m].get("per_instance", {})
        n_total = n_inc = n_top1 = n_ent = 0
        
        for si, k_data in per_inst.items():
            ks = sorted(int(k) for k in k_data)
            if len(ks) < 2: continue
            n_total += 1
            first = k_data[str(ks[0])]["metrics"]
            last = k_data[str(ks[-1])]["metrics"]
            
            if last.get("value_correct", 0) > first.get("value_correct", 0) + 0.01:
                n_inc += 1
            if last.get("top1_is_correct", False):
                n_top1 += 1
            if first.get("normalized_entropy", 0) > 0.5:
                n_ent += 1
                
        if n_total > 0:
            row_inc.append(f"{round((n_inc/n_total)*100)}\\%")
            row_top1.append(f"{round((n_top1/n_total)*100)}\\%")
            row_ent.append(f"{round((n_ent/n_total)*100)}\\%")
            row_n.append(str(n_total))
        else:
            row_inc.append("-")
            row_top1.append("-")
            row_ent.append("-")
            row_n.append("-")
            
    latex.append(" & ".join(row_inc) + " \\\\")
    latex.append(" & ".join(row_top1) + " \\\\")
    latex.append(" & ".join(row_ent) + " \\\\")
    latex.append("\\midrule")
    latex.append(" & ".join(row_n) + " \\\\")
    
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\label{tab:per_instance_statistics}",
        "\\end{table*}"
    ])
    return "\n".join(latex)

def run_analysis(task):
    all_results, results_dir = _load_all_results(task)
    if not all_results:
        print(f"No results found in {results_dir}")
        return
        
    available = [m for m in _MODES_ORDER if m in all_results]
    all_k = sorted({
        int(k) for mode in available for k in all_results[mode]["summary"]
    })
    
    # Save Main Table
    main_latex = generate_main_table(all_results, available, all_k)
    main_path = results_dir / "table_main.tex"
    with open(main_path, "w") as f:
        f.write(main_latex)
    print(f"Saved Main Table to: {main_path}")
    
    # Save Table 9
    t9_latex = generate_table9(all_results, available, all_k)
    t9_path = results_dir / "table_9.tex"
    with open(t9_path, "w") as f:
        f.write(t9_latex)
    print(f"Saved Table 9 to: {t9_path}")
    
    # Save Table 10
    t10_latex = generate_table10(all_results, available)
    t10_path = results_dir / "table_10.tex"
    with open(t10_path, "w") as f:
        f.write(t10_latex)
    print(f"Saved Table 10 to: {t10_path}")
    
    # Save Table 11
    t11_latex = generate_table11(all_results, available)
    t11_path = results_dir / "table_11.tex"
    with open(t11_path, "w") as f:
        f.write(t11_latex)
    print(f"Saved Table 11 to: {t11_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["prosqa", "gsm"], default="prosqa")
    args = parser.parse_args()
    run_analysis(args.task)