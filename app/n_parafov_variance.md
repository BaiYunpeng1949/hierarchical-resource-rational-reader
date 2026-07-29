# Is n_parafov measurable, or is it noise?

Short answer: **measurable, by a wide margin.** At the corpus level the parafoveal
preview moves every quantity the sentence env reads by 34–190x the spread across
reruns of the same setting. Shipping the n_parafov control is justified.

## Why this had to be checked first

The generator's noisy parafoveal matching draws from `torch.rand`, and until
`--seed` was added it drew unseeded. A single n=1 file next to a single n=3 file
would have been two arbitrary draws, and any difference between them could have
been the preview or could have been the noise. This sweep separates the two.

## Method

15 runs of `process_my_sentences_with_word_features_add_obs.py`: n_parafov ∈
{1, 2, 3} x seed ∈ {0..4}, over all 9 stimuli (1174 words, excluding the last
word of each sentence, which predicts `[END]` rather than a preview). ~3m20s per
run, three in parallel, ~16 minutes wall clock.

The seed is applied per word, derived from `(seed, position in the dataset)`, so
one seed gives every n_parafov variant the same draws at the same word. Without
that, a variant that draws a different number of random values per word would
leave every later word offset in the stream, and the preview effect would arrive
mixed with a reshuffle of the noise.

Four measured quantities, all of them things `SentencesManager` derives from
`prediction_candidates` and hands to the env:

| Metric | What reads it |
| --- | --- |
| `predictability` | max candidate probability — the env's observed next-word predictability (`SentenceReadingEnv.py:470`) |
| `top_correct` | whether that top candidate is the actual next word |
| `n_candidates` | how many candidates survived the preview filter |
| `skip_integration` | the belief a skipped word is integrated with (`SentenceReadingEnv.py:335`) |

## Result

Mean across seeds, +/- SD across the 5 seeds:

| n_parafov | predictability | top_correct | n_candidates | skip_integration |
| --- | --- | --- | --- | --- |
| 1 | 0.5049 +/- 0.0059 | 0.3761 +/- 0.0051 | 4.586 +/- 0.010 | 0.4270 +/- 0.0019 |
| 2 | 0.6822 +/- 0.0055 | 0.5450 +/- 0.0053 | 2.663 +/- 0.019 | 0.3990 +/- 0.0004 |
| 3 | 0.7007 +/- 0.0059 | 0.6204 +/- 0.0042 | 1.753 +/- 0.015 | 0.3648 +/- 0.0025 |

Effect against noise — the range across n_parafov, over the mean seed SD:

| Metric | n-range | seed SD | ratio |
| --- | --- | --- | --- |
| predictability | 0.1959 | 0.0058 | **34x** |
| top_correct | 0.2443 | 0.0049 | **50x** |
| n_candidates | 2.8324 | 0.0149 | **190x** |
| skip_integration | 0.0622 | 0.0016 | **39x** |

The paired within-seed differences agree, with SDs across seeds an order of
magnitude below the effect (e.g. n=1 -> n=3 predictability +0.1959, SD 0.0089).

## Three things worth knowing before this goes in a figure

**The corpus mean is stable; individual words are not.** A rerun at the same
n_parafov changes 92% of words at n=1 (mean |Δ predictability| 0.11) and 53% at
n=2. It is only averaging over 1174 words that pulls the seed SD down to ~0.006.
Any per-word or per-sentence claim about n_parafov needs its own variance check —
this sweep does not license one.

**The effect is not linear.** Predictability jumps +0.177 from n=1 to n=2 and only
+0.019 from n=2 to n=3; `top_correct` keeps climbing (+0.169 then +0.076). n=2 is
close to saturation on predictability but not on accuracy.

**`skip_integration` moves the opposite way** — more preview gives a *lower*
integration belief for skipped words (0.427 -> 0.365), because the surviving
candidate set shrinks and its top member is ranked differently. This is a
consequence of how the ranked probability is assigned, not obviously an intended
model behaviour, and it is worth a look before it is interpreted.

## What this does not show

These are asset-level measurements. They establish that n_parafov changes what
the sentence agent observes; they do **not** show how far that propagates into
simulated eye-movement metrics, where the policy, the time budget and the other
two levels all intervene. That needs simulation runs under each variant, with
their own seed sweep.

They also say nothing about the shipped
`assets/processed_my_stimulus_with_observations.json`, which is an unseeded draw
and no seed reproduces. It is excluded from this comparison on purpose. A seeded
rebuild at its own n_parafov=2 disagrees with it on most words, so adopting
seeded assets shifts absolute results away from the published ones — a decision
for Bai, and a reproducibility issue in the paper independent of this app.

## Reproducing

Not reproducible from the repository, by choice. The sweep was one-off work to
decide whether the n_parafov control was worth shipping; the answer was yes, the
three seed-0 assets it produced are committed in
`sub_models/sentence_read_v0604/assets/variants/`, and the seeding and sweep
tooling was not kept — app users do not regenerate assets.

Redoing this measurement means reinstating that tooling: a `--seed` option on
`process_my_sentences_with_word_features_add_obs.py` seeding `torch` **per word**
from `(seed, position in the dataset)`, plus the 3x5 driver and the analysis. The
per-word detail is the part that is easy to get wrong — see Method above.
