"""
Model-free reproduction of the suspected context-window truncation in BitMark.

No Infinity, no CUDA, no VAE. We import the *real, unmodified*
WatermarkLookupProcessor from the BitMark repo and drive it exactly the way
Infinity/infinity/models/infinity.py drives it during generation, replacing the
transformer's logits with a synthetic tensor.

The generation loop being mirrored is infinity.py:620-629:

    for n in range(token_size):
        t_n = torch.arange(0, num_tokens, device="cuda")*token_size + n
        if n < watermark.context_width:
            processed_logits = logits_BlV[:, bit_order[t_n], :]
        else:
            index_tensor = idx_Bld[:, bit_order[t_n-1]]          # <-- previous BIT VALUE
            processed_logits = watermark.logits_processor(index_tensor, logits_BlV[:, bit_order[t_n], :])
        bit_n_of_tokens = sample(...)
        idx_Bld[:, bit_order[t_n]] = bit_n_of_tokens

`index_tensor` holds the *value* of the previous bit (0 or 1), and is used
directly as a row index into a lookup table of size 2**n.

Usage:  python repro_context_truncation.py
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from extended_watermark_processor import WatermarkDetector, WatermarkLookupProcessor  # noqa: E402

TOKEN_SIZE = 32  # bits per token for Infinity-2B
DEVICE = "cpu"


def make_processor(green_list, delta):
    return WatermarkLookupProcessor(
        vocab=[0, 1], device=DEVICE, delta=delta, gamma=0.5, green_list=set(green_list)
    )


def simulate(green_list, delta=2.0, num_tokens=4096, seed=0, mode="asis", logit_sigma=2.0):
    """Run the real logit-bias processor over a synthetic stream of tokens.

    mode="asis"  -> index the lookup table the way infinity.py does (previous bit value)
    mode="fixed" -> index it with the last n-1 generated bits of the current token

    Returns statistics plus a record of which contexts were ever consulted.
    """
    proc = make_processor(green_list, delta)
    n = proc.context_width          # full green-list word length, e.g. 3
    prefix_width = n - 1            # what Eq. (1) says the context should be

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # Stand-in for the transformer's per-bit logits. Symmetric and iid, so any
    # asymmetry in the output is caused by the watermark alone.
    logits = torch.randn((num_tokens, TOKEN_SIZE, 2), generator=g, device=DEVICE) * logit_sigma

    bits = torch.zeros((num_tokens, TOKEN_SIZE), dtype=torch.long, device=DEVICE)

    rows_used = set()               # lookup-table row indices actually reached
    mismatches = 0                  # steps where the consulted prefix != the true prefix
    steps = 0
    watermarked_positions = []

    for j in range(TOKEN_SIZE):
        step_logits = logits[:, j, :].clone()

        if j < n:  # head bits, left unwatermarked by the original loop
            processed = step_logits
        else:
            if mode == "asis":
                # verbatim infinity.py:625 -- the value of the single previous bit
                index_tensor = bits[:, j - 1]
            elif mode == "fixed":
                # the patched path: hand the processor the last n-1 generated bits
                index_tensor = proc.context_index(bits[:, j - prefix_width : j])
            else:
                raise ValueError(mode)

            rows_used.update(index_tensor.unique().tolist())
            mask = proc.lookup_greenset[index_tensor]          # (num_tokens, 2)
            processed = proc._bias_greenlist_logits(
                scores=step_logits, greenlist_mask=mask, greenlist_bias=proc.delta
            )

            # The table reads the prefix off the low n-1 bits of the row index.
            consulted = index_tensor % (2 ** prefix_width)
            true_window = bits[:, j - prefix_width : j]
            weights = torch.tensor(
                [2 ** k for k in range(prefix_width)][::-1], dtype=torch.long, device=DEVICE
            )
            true_idx = (true_window * weights).sum(dim=1)
            mismatches += int((consulted != true_idx).sum())
            steps += index_tensor.numel()
            watermarked_positions.append(j)

        probs = torch.softmax(processed, dim=-1)
        bits[:, j] = torch.multinomial(probs, num_samples=1, generator=g)[:, 0]

    wm = bits[:, watermarked_positions]
    consulted_prefixes = sorted(
        {format(r % (2 ** prefix_width), f"0{prefix_width}b") for r in rows_used}
    )
    return {
        "n": n,
        "frac_ones_watermarked": wm.float().mean().item(),
        "frac_ones_all": bits.float().mean().item(),
        "rows_used": sorted(rows_used),
        "consulted_prefixes": consulted_prefixes,
        "wrong_prefix_rate": mismatches / max(steps, 1),
        "bits": bits,
        "table": proc.lookup_greenset,
    }


def rule_of(green_list, n=3):
    """Map a green list to f: {0,1}^(n-1) -> {0, 1, '-'} (idle when both/neither green)."""
    f = {}
    for p in ["".join(b) for b in itertools.product("01", repeat=n - 1)]:
        g0, g1 = (p + "0") in green_list, (p + "1") in green_list
        f[p] = "-" if g0 == g1 else ("0" if g0 else "1")
    return f


def all_3bit_lists_of_size_4():
    words = [f"{i:03b}" for i in range(8)]
    return [set(c) for c in itertools.combinations(words, 4)]


def main():
    torch.set_printoptions(linewidth=120)

    print("=" * 78)
    print("PART 1 -- which contexts does the embedder ever consult?")
    print("=" * 78)
    gl = {"000", "011", "101", "110"}  # fully active, f = 0 1 1 0
    for mode in ("asis", "fixed"):
        r = simulate(gl, mode=mode, num_tokens=2048)
        all_prefixes = sorted("".join(b) for b in itertools.product("01", repeat=r["n"] - 1))
        missing = sorted(set(all_prefixes) - set(r["consulted_prefixes"]))
        print(f"\ngreen list {sorted(gl)}   n={r['n']}   mode={mode}")
        print(f"  lookup-table rows reached : {r['rows_used']}  (table has {2**r['n']} rows)")
        print(f"  prefixes consulted        : {r['consulted_prefixes']}")
        print(f"  prefixes NEVER consulted  : {missing if missing else 'none'}")
        print(f"  steps using wrong prefix  : {r['wrong_prefix_rate']:.1%}")

    print("\n" + "=" * 78)
    print("PART 2 -- do f(10) and f(11) affect generation at all?")
    print("=" * 78)
    print("Four lists sharing f(00)=0, f(01)=1 but differing at 10/11:\n")
    print(f"{'green list':<28} {'f(00,01,10,11)':<16} {'asis 1s':>9} {'fixed 1s':>9}")
    for gl in [
        {"000", "011", "100", "110"},
        {"000", "011", "100", "111"},
        {"000", "011", "101", "110"},
        {"000", "011", "101", "111"},
    ]:
        f = rule_of(gl)
        rule = "".join(f[p] for p in ["00", "01", "10", "11"])
        a = simulate(gl, mode="asis", num_tokens=4096)["frac_ones_watermarked"]
        b = simulate(gl, mode="fixed", num_tokens=4096)["frac_ones_watermarked"]
        print(f"{','.join(sorted(gl)):<28} {rule:<16} {a:>9.4f} {b:>9.4f}")

    print("\n" + "=" * 78)
    print("PART 3 -- all 70 three-bit lists of size 4, grouped by (f(00), f(01))")
    print("=" * 78)
    groups = {}
    for gl in all_3bit_lists_of_size_4():
        f = rule_of(gl)
        key = (f["00"], f["01"])
        val = simulate(gl, mode="asis", num_tokens=2048)["frac_ones_watermarked"]
        groups.setdefault(key, []).append(val)

    print(f"\n{'f(00) f(01)':<14} {'rows':>5} {'min 1s':>9} {'max 1s':>9} {'spread':>9}")
    worst = 0.0
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        vals = groups[key]
        spread = max(vals) - min(vals)
        worst = max(worst, spread)
        print(f"{key[0]:<6} {key[1]:<7} {len(vals):>5} {min(vals):>9.4f} {max(vals):>9.4f} {spread:>9.4f}")
    print(
        f"\nlargest within-group spread over all 70 lists: {worst:.4f}"
        "  (Monte-Carlo noise is ~0.01 at this sample size)"
    )

    print("\n" + "=" * 78)
    print("PART 4 -- the six fully-active, balance-preserving 3-bit lists")
    print("=" * 78)
    print("A genuine 3-bit rule makes all six produce a 50/50 bit stream.\n")
    print(f"{'green list':<28} {'rule':<8} {'asis 1s':>9} {'fixed 1s':>9}")
    for gl in all_3bit_lists_of_size_4():
        f = rule_of(gl)
        if "-" in f.values():
            continue
        rule = "".join(f[p] for p in ["00", "01", "10", "11"])
        if rule.count("1") != 2:  # balance-preserving ones only
            continue
        a = simulate(gl, mode="asis", num_tokens=8192)["frac_ones_watermarked"]
        b = simulate(gl, mode="fixed", num_tokens=8192)["frac_ones_watermarked"]
        print(f"{','.join(sorted(gl)):<28} {rule:<8} {a:>9.4f} {b:>9.4f}")

    print("\n" + "=" * 78)
    print("PART 5 -- embed, then detect with the REAL WatermarkDetector")
    print("=" * 78)
    print("No VAE round-trip, so the detector sees exactly the bits that were")
    print("generated. Any z below the ceiling is pure embed/detect disagreement.\n")
    print(f"{'green list':<28} {'rule':<8} {'asis z':>10} {'fixed z':>10}")
    for gl in all_3bit_lists_of_size_4():
        f = rule_of(gl)
        if "-" in f.values() or "".join(f[p] for p in ["00", "01", "10", "11"]).count("1") != 2:
            continue
        rule = "".join(f[p] for p in ["00", "01", "10", "11"])
        row = []
        for mode in ("asis", "fixed"):
            bits = simulate(gl, mode=mode, num_tokens=2048)["bits"]
            det = WatermarkDetector(
                vocab=[0, 1], gamma=0.5, delta=2.0, device=DEVICE,
                z_threshold=4.0, ignore_repeated_ngrams=False,
                green_list=",".join(sorted(gl)),
            )
            row.append(det.detect(tokenized_text=bits.flatten())["z_score"])
        print(f"{','.join(sorted(gl)):<28} {rule:<8} {row[0]:>10.2f} {row[1]:>10.2f}")


if __name__ == "__main__":
    main()
