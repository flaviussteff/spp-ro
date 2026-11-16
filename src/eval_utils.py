"""
Shared evaluation utilities for SPP-Ro.

1. Correct option scoring: `prompt` and `option` are tokenized JOINTLY and the
   option span is recovered from character offsets. Tokenizing the prompt
   separately is wrong for BPE tokenizers: a prompt ending in a space
   ("Răspuns: ") yields a standalone "Ġ" token on its own, but when followed by
   the option the space merges into the option's first token ("ĠTrat"). The old
   code therefore skipped the first option token and miscounted the length.
2. Statistics, so that small benchmarks (N=20..600) are reported with uncertainty instead of bare
   percentages: bootstrap confidence intervals; for two models scored on the SAME items, the exact
   McNemar test (binomial test on the discordant items) and an exact paired sign-flip test for integer
   per-item scores; the exact sign test for continuous paired differences; Holm's step-down correction
   for a family of comparisons. Fisher's exact test is kept only for unpaired samples (it ignores the
   pairing and is too conservative for two models on the same items).
3. `tex_escape` for every free-text string written into a LaTeX table (a bare % starts a comment).
"""
from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Option scoring
# ---------------------------------------------------------------------------
def option_token_span(tokenizer, prompt: str, option: str) -> Tuple[List[int], int]:
    """
    Tokenizes prompt+option jointly. Returns (input_ids, first_option_token_index).
    Trailing whitespace of the prompt is attributed to the option (it merges into the
    option's first BPE token), so the boundary is len(prompt.rstrip()).
    """
    full = prompt + option
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    boundary = len(prompt.rstrip())
    start = len(ids)
    for i, (s, e) in enumerate(offsets):
        if e > boundary:
            start = i
            break
    return ids, start


@torch.no_grad()
def score_option_ll(model, tokenizer, prompt: str, option: str, device: str = "cuda",
                    length_normalize: bool = True) -> float:
    """Conditional log-likelihood of `option` given `prompt` (per token if length_normalize)."""
    ids, start = option_token_span(tokenizer, prompt, option)
    if start >= len(ids):
        return -999.0
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=input_ids).logits[0]            # [T, V]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    targets = input_ids[0, start:]                            # tokens of the option
    pred_rows = log_probs[start - 1 : len(ids) - 1]           # row t-1 predicts token t
    tok_lp = pred_rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = tok_lp.sum().item()
    return total / len(targets) if length_normalize else total


@torch.no_grad()
def batch_score_options(model, tokenizer, prompts: Sequence[str], options: Sequence[str],
                        device: str = "cuda", batch_size: int = 64,
                        length_normalize: bool = True) -> List[float]:
    """Batched version of score_option_ll with right padding."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    scores: List[float] = []
    for b0 in range(0, len(prompts), batch_size):
        spans = [option_token_span(tokenizer, p, o) for p, o in zip(prompts[b0:b0 + batch_size], options[b0:b0 + batch_size])]
        max_len = max(len(ids) for ids, _ in spans)
        inp = torch.full((len(spans), max_len), pad_id, dtype=torch.long, device=device)
        att = torch.zeros((len(spans), max_len), dtype=torch.long, device=device)
        for i, (ids, _) in enumerate(spans):
            inp[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            att[i, : len(ids)] = 1
        log_probs = F.log_softmax(model(input_ids=inp, attention_mask=att).logits.float(), dim=-1)
        for i, (ids, start) in enumerate(spans):
            if start >= len(ids):
                scores.append(-999.0)
                continue
            targets = inp[i, start : len(ids)]
            rows = log_probs[i, start - 1 : len(ids) - 1]
            lp = rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum().item()
            scores.append(lp / len(targets) if length_normalize else lp)
    return scores


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci(flags: Sequence[bool], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI (in %) for the accuracy of a list of correct/incorrect flags."""
    n = len(flags)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    vals = [1.0 if f else 0.0 for f in flags]
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(100 * lo, 1), round(100 * hi, 1))


def _hypergeom_pmf(k: int, n1: int, n2: int, total_success: int) -> float:
    # P(X = k) where X = successes in group 1, given margins
    return math.comb(n1, k) * math.comb(n2, total_success - k) / math.comb(n1 + n2, total_success)


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test p-value for k1/n1 vs k2/n2 (sum of tables as or less probable)."""
    total_success = k1 + k2
    k_min = max(0, total_success - n2)
    k_max = min(n1, total_success)
    p_obs = _hypergeom_pmf(k1, n1, n2, total_success)
    p = 0.0
    for k in range(k_min, k_max + 1):
        pk = _hypergeom_pmf(k, n1, n2, total_success)
        if pk <= p_obs * (1 + 1e-9):
            p += pk
    return round(min(1.0, p), 4)


def paired_permutation_p(flags_a: Sequence[bool], flags_b: Sequence[bool], n_perm: int = 20000, seed: int = 0) -> float:
    """Paired sign-flip permutation test on per-item differences (same items scored by two models)."""
    assert len(flags_a) == len(flags_b)
    diffs = [int(a) - int(b) for a, b in zip(flags_a, flags_b)]
    obs = abs(sum(diffs))
    nz = [d for d in diffs if d != 0]
    if not nz:
        return 1.0
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        s = 0
        for d in nz:
            s += d if rng.random() < 0.5 else -d
        if abs(s) >= obs:
            count += 1
    return round(count / n_perm, 4)


def _binom_two_sided_half(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n trials against p = 0.5 (exact integer arithmetic)."""
    if n <= 0:
        return 1.0
    m = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(m + 1))
    return min(1.0, float(Fraction(2 * tail, 1 << n)))


def mcnemar_exact_p(flags_a: Sequence[bool], flags_b: Sequence[bool]) -> float:
    """Two-sided exact McNemar test for two models scored on the same items: binomial test on the discordant
    items (a right / b wrong vs a wrong / b right)."""
    assert len(flags_a) == len(flags_b)
    gained = sum(1 for a, b in zip(flags_a, flags_b) if a and not b)
    lost = sum(1 for a, b in zip(flags_a, flags_b) if b and not a)
    return _binom_two_sided_half(gained, gained + lost)


def sign_test_p(diffs: Sequence[float]) -> float:
    """Two-sided exact sign test on paired differences (zeros dropped)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return _binom_two_sided_half(pos, pos + neg)


def signflip_exact_p(diffs: Sequence[int]) -> float:
    """Exact two-sided paired sign-flip (randomisation) test for INTEGER per-item differences: the probability, over
    all 2^m sign assignments of the m non-zero differences, that |sum| is at least the observed |sum|. Computed exactly
    by dynamic programming over the attainable sums. For 0/1 outcomes (differences in {-1, 0, 1}) it equals the exact
    McNemar test."""
    vals = [abs(int(d)) for d in diffs if int(d) != 0]
    if not vals:
        return 1.0
    obs = abs(sum(int(d) for d in diffs))
    total = sum(vals)
    counts = [0] * (2 * total + 1)        # counts[s + total] = number of sign assignments with sum s
    counts[total] = 1
    reach = 0
    for v in vals:
        new = [0] * len(counts)
        for s in range(total - reach, total + reach + 1):
            c = counts[s]
            if c:
                new[s - v] += c
                new[s + v] += c
        counts = new
        reach += v
    extreme = sum(c for i, c in enumerate(counts) if c and abs(i - total) >= obs)
    return min(1.0, float(Fraction(extreme, 1 << len(vals))))


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """Holm's step-down adjusted p-values (family-wise error control), in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def round_p(p: float) -> float:
    """p-values are stored with 4 significant digits (rounding to 4 decimals would turn 1e-24 into 0)."""
    return float(f"{p:.4g}")


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: object) -> str:
    """Escapes LaTeX special characters in free text (model descriptions, labels): a bare % would comment out the
    rest of a table row, & would add a column, _ { } would break the line."""
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(s))


def summarize_accuracy(flags: Sequence[bool], base_flags: Sequence[bool] | None = None) -> Dict[str, object]:
    """Accuracy with a bootstrap CI and, if the reference model's flags on the same items are given, the exact
    McNemar test on the paired items (`p_vs_base`)."""
    n = len(flags)
    k = sum(1 for f in flags if f)
    out: Dict[str, object] = {
        "n": n,
        "correct": k,
        "accuracy": round(100.0 * k / n, 1) if n else 0.0,
        "ci95": bootstrap_ci(flags),
    }
    if base_flags is not None and len(base_flags) == n:
        out["baseline_correct"] = sum(1 for f in base_flags if f)
        out["items_gained"] = sum(1 for a, b in zip(flags, base_flags) if a and not b)
        out["items_lost"] = sum(1 for a, b in zip(flags, base_flags) if b and not a)
        out["p_vs_base"] = round_p(mcnemar_exact_p(flags, base_flags))
        out["p_method"] = "McNemar exact (paired items)"
    return out
