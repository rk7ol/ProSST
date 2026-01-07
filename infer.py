#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with ProSST.

This script reads one or more ProteinGym substitution CSVs and computes a
masked-marginal delta log-probability score for each variant:

  Δ = log P(mutant_aa | context, structure) - log P(wildtype_aa | context, structure)

ProSST requires a quantized structure sequence (SST) aligned to the residue
sequence length L. Provide it via ONE of:
  - `--structure_seq_path`: a FASTA whose sequence is "i0,i1,...,i(L-1)"
  - `--structure_dir`: directory containing `{protein}.fasta` (default protein=CSV stem)
  - a CSV column (default `structure_sequence`) containing the same comma-separated ints

Expected input columns:
  - `mutant`: e.g. "A123G" or multi-mutants like "A123G:D200N" (1-based positions)
  - `mutated_sequence`: full mutant protein sequence (same length across rows)
  - `DMS_score`: experimental score (used only for Spearman summary)

Outputs:
  - For each input CSV: a new CSV with a `prosst_delta_logp` column.
  - A `summary.csv` aggregating per-file Spearman statistics.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from importlib.metadata import distributions
except Exception:  # pragma: no cover
    def distributions():  # type: ignore[override]
        return []


REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def compute_spearman(pred_scores, true_scores) -> tuple[float | None, float | None]:
    rho, pval = spearmanr(pred_scores, true_scores, nan_policy="omit")
    rho_val = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
    pval_val = None if pval is None or (isinstance(pval, float) and math.isnan(pval)) else float(pval)
    return rho_val, pval_val


def _fmt_float(x: float | None, *, fmt: str) -> str:
    return "nan" if x is None else format(x, fmt)


def collect_installed_packages() -> list[str]:
    items: list[str] = []
    for dist in distributions():
        name = None
        try:
            name = dist.metadata.get("Name")
        except Exception:
            name = None
        if not name:
            continue
        items.append(f"{name}=={dist.version}")
    return sorted(set(items), key=str.lower)


def print_runtime_environment() -> None:
    print("========== Runtime ==========")
    print(f"Python:        {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable:    {sys.executable}")
    print(f"Platform:      {sys.platform}")
    print("Packages:")
    for item in collect_installed_packages():
        print(f"  - {item}")
    print("=============================\n")


_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_single_mutant(mut_str: str) -> tuple[str, int, str]:
    m = _MUT_RE.match(mut_str.strip())
    if not m:
        raise ValueError(f"Unsupported mutant format: {mut_str!r} (expected e.g. 'A123G')")
    wt_aa, pos1_s, mut_aa = m.group(1), m.group(2), m.group(3)
    pos1 = int(pos1_s)
    return wt_aa, pos1, mut_aa


def split_multi_mutant(mutant: str) -> list[str]:
    mutant = str(mutant).strip()
    if not mutant:
        return []
    return [m.strip() for m in mutant.split(":") if m.strip()]


def recover_wt_sequence(mutated_sequence: str, mutant: str) -> str:
    chars = list(mutated_sequence)
    for sub in split_multi_mutant(mutant):
        wt_aa, pos1, _mut_aa = parse_single_mutant(sub)
        idx0 = pos1 - 1
        if idx0 < 0 or idx0 >= len(chars):
            raise ValueError(f"Mutation position out of range: {sub!r} for sequence length {len(chars)}")
        chars[idx0] = wt_aa
    return "".join(chars)


def resolve_csv_paths(*, data_dir: Path, csv: str | None) -> list[Path]:
    if csv:
        p = Path(csv)
        if p.exists():
            return [p]
        cand = data_dir / csv
        if cand.exists():
            return [cand]
        raise FileNotFoundError(f"--input_csv not found: {csv!r} (looked in {data_dir})")

    return sorted(data_dir.glob("*.csv"))


def read_fasta_sequence(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(str(path))
    seq_lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(">"):
                continue
            seq_lines.append(s)
    if not seq_lines:
        raise ValueError(f"Empty FASTA sequence: {path}")
    return "".join(seq_lines).strip()


def parse_structure_sequence(raw: str) -> list[int]:
    s = str(raw).strip()
    if not s:
        return []
    s = s.strip("[](){}")
    parts = [p.strip() for p in s.split(",")]
    items: list[int] = []
    for p in parts:
        if not p:
            continue
        items.append(int(p))
    return items


def tokenize_structure_sequence(structure_sequence: list[int], *, device: str) -> torch.Tensor:
    shift = [i + 3 for i in structure_sequence]
    shift = [1, *shift, 2]
    return torch.tensor([shift], dtype=torch.long, device=device)


def _resolve_structure_sequence(
    *,
    df: pd.DataFrame,
    csv_path: Path,
    structure_seq_path: str | None,
    structure_dir: str | None,
    structure_fasta: str | None,
    structure_col: str,
) -> list[int]:
    if structure_seq_path:
        return parse_structure_sequence(read_fasta_sequence(Path(structure_seq_path)))

    if structure_dir:
        base = Path(structure_dir)
        fname = structure_fasta or f"{csv_path.stem}.fasta"
        return parse_structure_sequence(read_fasta_sequence(base / fname))

    if structure_col in df.columns:
        val = df[structure_col].iloc[0]
        return parse_structure_sequence(val)

    raise ValueError(
        "Missing structure sequence. Provide one of: "
        "--structure_seq_path, --structure_dir, or a CSV column via --structure_col."
    )


@torch.no_grad()
def run_one_csv(
    *,
    csv_path: Path,
    output_dir: Path,
    output_suffix: str,
    model,
    tokenizer,
    device: str,
    progress_every: int,
    structure_seq_path: str | None,
    structure_dir: str | None,
    structure_fasta: str | None,
    structure_col: str,
    score_col: str,
) -> dict:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns: {sorted(missing)}")

    first = df.iloc[0]
    wt_seq = recover_wt_sequence(str(first["mutated_sequence"]), str(first["mutant"]))
    if not wt_seq:
        raise ValueError(f"{csv_path.name}: failed to recover wildtype sequence")

    wt_seqs = (
        df.apply(lambda r: recover_wt_sequence(str(r["mutated_sequence"]), str(r["mutant"])), axis=1)
        .dropna()
        .unique()
        .tolist()
    )
    if len(wt_seqs) != 1:
        raise ValueError(
            f"{csv_path.name}: wildtype recovery inconsistent across rows "
            f"(found {len(wt_seqs)} distinct WT sequences)"
        )

    structure_sequence = _resolve_structure_sequence(
        df=df,
        csv_path=csv_path,
        structure_seq_path=structure_seq_path,
        structure_dir=structure_dir,
        structure_fasta=structure_fasta,
        structure_col=structure_col,
    )
    if not structure_sequence:
        raise ValueError(f"{csv_path.name}: empty structure sequence")
    if len(structure_sequence) != len(wt_seq):
        raise ValueError(
            f"{csv_path.name}: structure length {len(structure_sequence)} != sequence length {len(wt_seq)}"
        )

    ss_input_ids = tokenize_structure_sequence(structure_sequence, device=device)
    enc = tokenizer([wt_seq], return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )

    log_probs = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)[0]
    vocab = tokenizer.get_vocab()

    scores: list[float] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        mutant = getattr(row, "mutant")
        mutated_sequence = getattr(row, "mutated_sequence")

        pred_score = 0.0
        for sub in split_multi_mutant(mutant):
            wt_aa, pos1, mut_aa = parse_single_mutant(sub)
            idx0 = pos1 - 1

            if not (0 <= idx0 < len(mutated_sequence)):
                raise ValueError(f"{csv_path.name}: {sub!r} out of range for length {len(mutated_sequence)}")
            if mutated_sequence[idx0] != mut_aa:
                raise ValueError(
                    f"{csv_path.name}: {sub!r} inconsistent with mutated_sequence at pos {pos1} "
                    f"(expected {mut_aa!r}, got {mutated_sequence[idx0]!r})"
                )

            wt_id = vocab.get(wt_aa)
            mut_id = vocab.get(mut_aa)
            if wt_id is None or mut_id is None:
                raise ValueError(f"{csv_path.name}: tokenizer vocab missing AA token(s): wt={wt_aa!r} mut={mut_aa!r}")

            pred_score += (log_probs[idx0, mut_id] - log_probs[idx0, wt_id]).item()

        scores.append(float(pred_score))

        if progress_every > 0 and i % progress_every == 0:
            print(f"[{csv_path.name}] scored {i}/{len(df)} variants")

    df[score_col] = scores
    rho, pval = compute_spearman(scores, df["DMS_score"].tolist())

    out_path = output_dir / f"{csv_path.stem}{output_suffix}"
    df.to_csv(out_path, index=False)

    print("\n========== ProteinGym zero-shot ==========")
    print("Model:        ProSST")
    print(f"CSV:          {csv_path.name}")
    print(f"Variants:     {len(df)}")
    print(f"Spearman ρ:   {_fmt_float(rho, fmt='.4f')}")
    print(f"P-value:      {_fmt_float(pval, fmt='.2e')}")
    print(f"Saved to:     {out_path}")
    print("==========================================\n")

    return {
        "model": "prosst",
        "csv": csv_path.name,
        "variants": int(len(df)),
        "spearman_rho": rho,
        "p_value": pval,
        "output_csv": out_path.name,
        "score_column": score_col,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="AI4Protein/ProSST-2048", help="HF repo id or local path.")
    p.add_argument("--input_csv", default=None, help="Only process this CSV (basename under data_dir, or an absolute path).")
    p.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    p.add_argument("--output_suffix", default="_prosst_zeroshot.csv")
    p.add_argument("--progress_every", type=int, default=100, help="Print progress every N variants (0 disables).")

    p.add_argument("--structure_seq_path", default=None, help="FASTA path containing SST sequence 'i0,i1,...'.")
    p.add_argument("--structure_dir", default=None, help="Directory containing `{protein}.fasta` SST sequences.")
    p.add_argument("--structure_fasta", default=None, help="Override SST fasta filename within --structure_dir.")
    p.add_argument("--structure_col", default="structure_sequence", help="CSV column name for SST sequence.")

    p.add_argument("--device", default=None, help="Override device: cuda/cpu (default: auto).")
    p.add_argument("--score_col", default="prosst_delta_logp")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_runtime_environment()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model: {args.model_path}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    summaries: list[dict] = []
    for csv_path in csv_paths:
        summaries.append(
            run_one_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                output_suffix=args.output_suffix,
                model=model,
                tokenizer=tokenizer,
                device=device,
                progress_every=max(0, int(args.progress_every)),
                structure_seq_path=args.structure_seq_path,
                structure_dir=args.structure_dir,
                structure_fasta=args.structure_fasta,
                structure_col=str(args.structure_col),
                score_col=str(args.score_col),
            )
        )

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

