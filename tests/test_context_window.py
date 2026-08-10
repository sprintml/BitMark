"""Regression tests for the BitMark embedding context window.

These pin the property that the embedder consults the *full* (n-1)-bit prefix,
which is what Eq. (1) of the paper specifies. They fail against the original
`index_tensor = idx_Bld[:, bit_order[t_n-1]]` and pass against the patched
`WatermarkLookupProcessor.context_index` path.

Run:  python -m pytest test_context_window.py -q
"""

import itertools

import pytest
import torch

from repro_context_truncation import (
    DEVICE,
    all_3bit_lists_of_size_4,
    make_processor,
    rule_of,
    simulate,
)

FULLY_ACTIVE_3BIT = [
    set(p + f[i] for i, p in enumerate(["00", "01", "10", "11"]))
    for f in ["".join(c) for c in itertools.product("01", repeat=4)]
]
BALANCE_PRESERVING = [gl for gl in FULLY_ACTIVE_3BIT if sum(int(w[-1]) for w in gl) == 2]


def test_there_are_16_fully_active_and_6_balance_preserving_lists():
    """Sanity check on the fixtures themselves."""
    assert len(FULLY_ACTIVE_3BIT) == 16
    assert len(BALANCE_PRESERVING) == 6


@pytest.mark.parametrize("green_list", FULLY_ACTIVE_3BIT, ids=lambda g: ",".join(sorted(g)))
def test_every_prefix_is_consulted(green_list):
    """A fully-active list must exercise all 2^(n-1) prefixes over a long stream."""
    r = simulate(green_list, mode="fixed", num_tokens=2048)
    expected = sorted("".join(b) for b in itertools.product("01", repeat=r["n"] - 1))
    assert r["consulted_prefixes"] == expected
    assert r["wrong_prefix_rate"] == 0.0


@pytest.mark.parametrize("green_list", BALANCE_PRESERVING, ids=lambda g: ",".join(sorted(g)))
def test_balance_preserving_lists_generate_a_fair_stream(green_list):
    """f with two 1s and two 0s cannot bias the marginal bit distribution."""
    frac = simulate(green_list, mode="fixed", num_tokens=8192)["frac_ones_watermarked"]
    assert frac == pytest.approx(0.5, abs=0.02), f"1-fraction {frac:.4f} for {sorted(green_list)}"


def test_all_fully_active_lists_share_one_green_fraction():
    """By symmetry every fully-active list must be detected equally well."""
    fractions = []
    for green_list in FULLY_ACTIVE_3BIT:
        r = simulate(green_list, mode="fixed", num_tokens=4096)
        bits = r["bits"]
        n = r["n"]
        windows = bits.unfold(1, n, 1).reshape(-1, n)
        words = ["".join(str(b) for b in row) for row in windows.tolist()]
        fractions.append(sum(w in green_list for w in words) / len(words))
    spread = max(fractions) - min(fractions)
    assert spread < 0.02, f"green fractions ranged over {spread:.4f}: {fractions}"


def test_trailing_context_bits_change_generation():
    """f(10) and f(11) must affect the output; under the bug they are dead inputs."""
    base = {"000", "011"}  # fixes f(00)=0, f(01)=1
    variants = [base | {"100", "110"}, base | {"101", "111"}]
    fracs = [simulate(gl, mode="fixed", num_tokens=4096)["frac_ones_watermarked"] for gl in variants]
    assert abs(fracs[0] - fracs[1]) > 0.1, f"f(10)/f(11) had no effect: {fracs}"


def test_context_index_rejects_a_short_window():
    """The narrow window that caused the bug must not be silently accepted."""
    proc = make_processor({"000", "011", "101", "110"}, delta=2.0)
    single_bit = torch.ones((4, 1), dtype=torch.long, device=DEVICE)
    with pytest.raises(AssertionError):
        proc.context_index(single_bit)


def test_context_index_packs_msb_first():
    proc = make_processor({"000", "011", "101", "110"}, delta=2.0)
    window = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=DEVICE)
    assert proc.context_index(window).tolist() == [0, 1, 2, 3]


def test_two_bit_scheme_is_unchanged_by_the_fix():
    """n=2 was accidentally correct before; the patch must not perturb it."""
    for green_list in [{"00", "11"}, {"01", "10"}, {"00", "10"}]:
        a = simulate(green_list, mode="asis", num_tokens=4096)["frac_ones_watermarked"]
        b = simulate(green_list, mode="fixed", num_tokens=4096)["frac_ones_watermarked"]
        assert a == pytest.approx(b, abs=1e-9), f"{sorted(green_list)}: {a} vs {b}"


def test_generation_is_not_collapsible_to_the_first_two_prefixes():
    """The 70 size-4 lists must not group by (f(00), f(01)) alone once fixed."""
    groups = {}
    for green_list in all_3bit_lists_of_size_4():
        f = rule_of(green_list)
        val = simulate(green_list, mode="fixed", num_tokens=2048)["frac_ones_watermarked"]
        groups.setdefault((f["00"], f["01"]), []).append(val)
    worst = max(max(v) - min(v) for v in groups.values())
    assert worst > 0.1, f"largest within-group spread was only {worst:.4f}"
