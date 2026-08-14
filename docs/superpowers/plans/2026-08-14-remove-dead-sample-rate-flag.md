# Remove Dead --sample-rate Flag from audio_classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `--sample-rate` CLI flag from `audio_classification`'s `train.py` and `test_onnx.py` — confirmed dead code that silently does nothing while looking like the thing that controls audio sample rate, trapping anyone who reads or invokes these scripts directly.

**Architecture:** Two `parser.add_argument('--sample-rate', ...)` calls deleted (one per file, each script has its own independent `get_args_parser()`). No other code changes. `--sampling-rate` — a separate, shared flag defined once in `train_base.py`'s base parser, already `required=True`, already the only thing `GoogleSpeechCommandsDataset` actually reads, and already what `tinyml-modelmaker`'s production orchestration (`audio_base.py`) passes — continues unchanged as the sole real control for audio sample rate. Two regression tests confirm the dead flag is gone from each parser.

**Tech Stack:** Python 3.10, PyTorch 2.7.1, pytest

## Global Constraints

- Python `==3.10.*`
- No new dependencies
- `--sampling-rate` (the flag that actually works, defined in `train_base.py`) must not be touched, renamed, or moved
- Removing `--sample-rate` must not change the parsed value of any other CLI argument

## Context: Why This Is Needed

Discovered during the `compile-hardening-radar-image-audio` plan's Task 3 (audio_classification), while building a synthetic benchmark fixture. Passing `--sampling-rate 1.0` (a value meaningful for the *other* reference scripts' generic FFT preprocessing, meaningless for audio) crashed `GoogleSpeechCommandsDataset`'s MFCC extraction with `RuntimeError: stft(...) : expected 0 < n_fft < 1, but got n_fft=0`, exposing the mechanism:

- `train_base.py:113` defines the base parser's generic `--sampling-rate` (`type=float, required=True`, dest `sampling_rate`) — shared across all seven reference scripts, meant for FFT-domain preprocessing params like radar's/timeseries' frame sizing.
- `audio_classification/train.py:132` (and, confirmed separately, `test_onnx.py:69`) *also* defines its own `--sample-rate` (`default=16000, type=int`, dest `sample_rate`) — a second, differently-named, audio-specific-looking flag.
- `GoogleSpeechCommandsDataset.__init__` (`audio_dataset.py:143-146`) does `for key, value in kwargs.items(): setattr(self, key, ...)` over every parsed CLI arg, then reads `self.sampling_rate = int(getattr(self, "sampling_rate", 16000))` — this reads the *base parser's* `sampling_rate`, never `sample_rate`. Confirmed via repo-wide grep: no reference to `self.sample_rate` or `args.sample_rate` exists anywhere in `tinyml-tinyverse`, `tinyml-modelmaker`, or `tinyml-modelzoo`.
- `tinyml-modelmaker/tinyml_modelmaker/ai_modules/audio/training/tinyml_tinyverse/audio_base.py:329,377` (the production orchestration layer) only ever passes `--sampling-rate` (from `self.params.data_processing_feature_extraction.sampling_rate`, default 16000) — it never passes `--sample-rate` at all.

**Practical impact:** the modelmaker-driven production path is not currently broken by this — it happens to work because `--sampling-rate`'s default (16000) and `--sample-rate`'s default (16000) coincide, and modelmaker only ever sets the one that's actually read. The real risk is for anyone invoking these scripts directly (bypassing modelmaker, as this session's own benchmark driver did) or maintaining this code: `--sample-rate` reads as the obviously-correct flag to set for a non-default sample rate, silently does nothing, and the actually-effective flag (`--sampling-rate`) reads as generic/unrelated-to-audio. This is a maintainability trap, not an active data-corruption bug in the shipped product today.

**Why removal, not redirection:** the alternative (making `GoogleSpeechCommandsDataset` read `sample_rate` instead of, or in preference to, `sampling_rate`) would change behavior for the working, already-relied-upon path (`--sampling-rate`, which `audio_base.py` already sets correctly) for no benefit, and risks introducing yet another dual-flag ambiguity. Since `--sample-rate` is provably read by nothing, deleting it is a pure simplification with no behavior change on any currently-working path.

---

## File Map

| Action | Path | Responsibility |
|--------|------|-----------------|
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/train.py:132` | Remove dead `--sample-rate` argument |
| Modify | `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/test_onnx.py:69` | Remove dead `--sample-rate` argument |
| Create | `tinyml-tinyverse/tests/test_audio_sample_rate_dead_flag_removed.py` | Regression test pinning both flags' removal |

---

## Task 1: Remove `--sample-rate` from both audio_classification scripts, with regression test

**Files:**
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/train.py` (line 132)
- Modify: `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/test_onnx.py` (line 69)
- Create: `tinyml-tinyverse/tests/test_audio_sample_rate_dead_flag_removed.py`

**Interfaces:**
- Consumes: `tinyml_tinyverse.references.audio_classification.train.get_args_parser`, `tinyml_tinyverse.references.audio_classification.test_onnx.get_args_parser` (both already exist, no signature change)

- [ ] **Step 1: Write the failing test**

```python
"""Regression test: audio_classification's train.py and test_onnx.py must NOT
define --sample-rate. It was dead code -- GoogleSpeechCommandsDataset
(audio_dataset.py) only ever reads self.sampling_rate, set from the shared
base parser's --sampling-rate (train_base.py:113), which tinyml-modelmaker's
production orchestration (audio_base.py) already passes correctly. Keeping
--sample-rate around silently does nothing while looking like the flag that
controls audio sample rate -- a maintainability trap for anyone reading or
invoking these scripts directly."""
from tinyml_tinyverse.references.audio_classification import train, test_onnx


def _dests(parser):
    return {action.dest for action in parser._actions}


def test_train_parser_has_no_dead_sample_rate_flag():
    dests = _dests(train.get_args_parser())
    assert "sample_rate" not in dests, (
        "train.py still defines --sample-rate, but GoogleSpeechCommandsDataset "
        "never reads self.sample_rate -- only self.sampling_rate (from the "
        "shared --sampling-rate flag). This dead flag should be removed."
    )
    assert "sampling_rate" in dests, (
        "the real, working flag (--sampling-rate, from the shared base parser) "
        "must still be present."
    )


def test_test_onnx_parser_has_no_dead_sample_rate_flag():
    dests = _dests(test_onnx.get_args_parser())
    assert "sample_rate" not in dests, (
        "test_onnx.py still defines --sample-rate, but GoogleSpeechCommandsDataset "
        "never reads self.sample_rate -- only self.sampling_rate (from the "
        "shared --sampling-rate flag). This dead flag should be removed."
    )
    assert "sampling_rate" in dests, (
        "the real, working flag (--sampling-rate, from the shared base parser) "
        "must still be present."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_audio_sample_rate_dead_flag_removed.py -v`
Expected: FAIL — both `test_..._has_no_dead_sample_rate_flag` tests fail, `"sample_rate"` is present in `dests`

- [ ] **Step 3: Remove the dead flag from train.py**

In `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/train.py`, delete line 132:

```python
    parser.add_argument('--sample-rate', help='Audio sample rate in Hz', default=16000, type=int)
```

- [ ] **Step 4: Remove the dead flag from test_onnx.py**

In `tinyml-tinyverse/tinyml_tinyverse/references/audio_classification/test_onnx.py`, delete line 69:

```python
    parser.add_argument('--sample-rate', help='Audio sample rate in Hz', default=16000, type=int)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd tinyml-tinyverse && python -m pytest tests/test_audio_sample_rate_dead_flag_removed.py -v`
Expected: PASS

- [ ] **Step 6: Run the full existing test suite for this module to confirm no collateral breakage**

Run: `cd tinyml-tinyverse && python -m pytest tests/ -v -k "audio"`
Expected: all pass (or only the same pre-existing failures already known from the compile-hardening plan's Task 3 report, if any — confirm via `git stash` on this task's own diff that any failure pre-exists and is unrelated, the same verification pattern used in that plan's Task 1-3 reports)

- [ ] **Step 7: Manual sanity check — confirm --sampling-rate alone still drives the real sample rate correctly**

Reuse the synthetic audio fixture from the compile-hardening plan's Task 3 (`make_audio_fixture.py` in the session scratchpad — regenerate if not present) and drive `train.get_args_parser().parse_args([...]); train.run(args)` with `--sampling-rate 16000` (no `--sample-rate`, since it no longer exists), a couple of epochs, `--device cpu`. Confirm it completes cleanly and produces the same MFCC feature shape as before (spot-check `n_fft`/`hop_length` implied by the log or a quick shape print) — this is the same invocation this session's own benchmark driver already used successfully, now with one less (dead) flag in its argv.

- [ ] **Step 8: Commit**

```bash
cd tinyml-tinyverse
git add tinyml_tinyverse/references/audio_classification/train.py tinyml_tinyverse/references/audio_classification/test_onnx.py tests/test_audio_sample_rate_dead_flag_removed.py
git commit -m "fix: remove dead --sample-rate flag from audio_classification scripts"
```
