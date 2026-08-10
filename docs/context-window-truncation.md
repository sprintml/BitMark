# Embedding used a truncated context window

The watermark embedder consulted only the single previous bit as context,
left-zero-padded to width `n-1`, while the detector used the full `n`-bit window.
For `n >= 3` this silently degenerated the scheme to the `n = 2` scheme. `n = 2`
was unaffected, which is why it went unnoticed.

This document records what was wrong, how it was verified, and what it implies for
previously published `n >= 3` numbers.

## The defect

### Embedding — `Infinity/infinity/models/infinity.py:625` (before)

```python
index_tensor = idx_Bld[:, bit_order[t_n-1]]
processed_logits = watermark.logits_processor(index_tensor, logits_BlV[:, bit_order[t_n], :])
```

`idx_Bld` holds sampled bits, so `index_tensor` is the **value of the single previous
bit** — an integer in `{0, 1}` — used directly as a row index.

### The table it indexes — `extended_watermark_processor.py:_init_lookup_set`

```python
for i in range(2**self.context_width):              # context_width = n
    binary = format(i, f'0{self.context_width}b')
    last_pred = binary[-(self.context_width-1):]    # prefix = LOW n-1 bits of the row index
```

The table is `2**n` rows and reads the prefix off the low `n-1` bits of the row index.
Correct in isolation — but the caller only ever supplies `0` or `1`, so at `n = 3`
only rows `000` and `001` are reachable, giving prefixes `00` and `01`.

**Effectively the prefix was `b_{j-1}` left-padded with zeros to width `n-1`.**
`f(10)` and `f(11)` were dead inputs.

### Detection — `extended_watermark_processor.py:_score_ngrams_in_passage`

```python
token_ngram_generator = ngrams(input_ids.cpu().tolist(), self.context_width+1)
```

The detector slides a genuine `n`-bit window. Hence the asymmetry.

Note that `context_width` means `n` in `WatermarkLookupProcessor` and `n-1` in
`WatermarkDetector`. The shared attribute name for two different quantities is part
of why this stayed invisible.

### Why `n = 2` was unaffected

Four rows, prefix = low 1 bit. Index `0` → row `00` → prefix `0`; index `1` → row `01`
→ prefix `1`. The single previous bit *is* the whole context. The generation branch is
named `case '2-bit_pattern':` — the code was written for a 2-bit context and never
generalised when `n` grew.

## The fix

`WatermarkLookupProcessor.context_index()` packs the last `n-1` generated bits
MSB-first into a row index, and **asserts** the window width so a narrow window can
never again be silently zero-padded. `infinity.py` builds that window from `idx_Bld`
and calls it.

The window is taken within the current token — `j >= n` is already enforced by the
existing `if n < watermark.context_width` guard — so no cross-token or cross-scale
bleed is introduced. `n = 2` output is bit-identical before and after.

## Verification

`tests/repro_context_truncation.py` imports the real, unmodified processor and
detector and drives them exactly as `infinity.py` does, substituting iid Gaussian
logits for the transformer. No Infinity, no VAE, no CUDA; it runs on CPU in seconds.

```bash
python tests/repro_context_truncation.py
python -m pytest tests/test_context_window.py -q
```

**Which contexts get consulted** (`{000,011,101,110}`, `delta=2`):

| version | table rows reached | prefixes consulted | never consulted | steps using wrong prefix |
|---|---|---|---|---|
| before | `[0, 1]` of 8 | `00`, `01` | `10`, `11` | 49.6% |
| after | `[0, 1, 2, 3]` | all four | none | 0.0% |

**`f(10)`/`f(11)` were dead inputs.** Four lists sharing `f(00)=0, f(01)=1`:

| green list | rule `f(00,01,10,11)` | before, 1s | after, 1s |
|---|---|---|---|
| `000,011,100,110` | 0100 | 0.4972 | 0.3531 |
| `000,011,100,111` | 0101 | 0.4972 | 0.4972 |
| `000,011,101,110` | 0110 | 0.4972 | 0.4996 |
| `000,011,101,111` | 0111 | 0.4972 | 0.6416 |

**All 70 size-4 green lists collapsed into 9 groups** keyed by `(f(00), f(01))` alone,
with a within-group spread of **0.0000** — with a fixed seed the generated bit streams
were byte-identical across every list in a group.

**The six balance-preserving fully-active lists:**

| green list | rule | before, 1s | after, 1s |
|---|---|---|---|
| `000,010,101,111` | 0011 | 0.2742 | 0.5011 |
| `000,011,100,111` | 0101 | 0.5022 | 0.5022 |
| `000,011,101,110` | 0110 | 0.5022 | 0.5006 |
| `001,010,100,111` | 1001 | 0.5002 | 0.5004 |
| `001,010,101,110` | 1010 | 0.5002 | 0.5002 |
| `001,011,100,110` | 1100 | 0.7271 | 0.5004 |

**Embed, then detect with the real `WatermarkDetector`.** No VAE round-trip, so every
deviation from the ceiling is pure embed/detect disagreement:

| green list | rule | before, z | after, z |
|---|---|---|---|
| `000,010,101,111` | 0011 | 42.72 | 103.93 |
| `000,011,100,111` | 0101 | 103.71 | 103.71 |
| `000,011,101,110` | 0110 | 0.93 | 105.50 |
| `001,010,100,111` | 1001 | -0.27 | 104.08 |
| `001,010,101,110` | 1010 | 105.16 | 105.16 |
| `001,011,100,110` | 1100 | -42.88 | 106.11 |

## Mechanism

The truncated embedder implemented `g(b_{j-1}) = f(0, b_{j-1})` while the detector
scored `f(b_{j-2}, b_{j-1})`. Therefore:

- `0101` and `1010` ignore `b_{j-2}` by construction → truncation was a no-op → full z.
- `0110` (XOR) and `1001` (XNOR) agreed only when `b_{j-2} = 0` → z ≈ 0.
- `0011` and `1100` collapsed to "always push 0" / "always push 1" → skewed stream.

## Implications for previously reported `n >= 3` results

The z ceiling does **not** move. `0101` and `1010` already sat on it, and for them the
fix changes nothing (103.71 → 103.71, bit-identical streams). What the fix does is
bring the other 14 fully-active lists *up to* the existing ceiling.

The best-reported `n=3` list, `000,011,100,111`, is `f = 0101` — one of exactly two
rules that survive the truncation intact. That number is real, but it was achieved by
a green list whose rule never needed the second context bit, which is why the `n=2`
best z rescaled for the extra idle head bits lands so close to it. Any `n >= 3`
ranking across green lists computed before this fix reflects the degenerate 2-bit
scheme and should be re-run.

## Regression tests

`tests/test_context_window.py` — 29 pass against the fix, 21 fail against the previous
behaviour. It asserts that every prefix in `{0,1}^(n-1)` is consulted for a fully
active list, that the six balance-preserving 3-bit lists produce a fair stream, that
all 16 fully-active lists share one green fraction, that `n = 2` is unperturbed, and
that `context_index` rejects a short window.

## Known remaining issues (not addressed here)

- The detector slides over the flattened stream, so n-grams straddling token
  boundaries and those covering the `n` unwatermarked head bits of each token are
  scored as if they were watermark decisions. At `n = 3` that is 3 of every 32
  positions (~9%), diluting z.
- `--watermark_context_width` (`architecture_wrapper.py:481`, default 4) is dead; the
  real width comes from `len(list(green_list)[0])`.
- `context_width` means `n` in the processor and `n-1` in the detector.
