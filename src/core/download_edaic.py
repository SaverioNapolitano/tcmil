"""Download E-DAIC transcripts and convert them to DAIC-WOZ format.

This project's text-only pipeline (``preprocess_raw.py``) expects each raw
transcript at ``raw/{ID}_TRANSCRIPT.csv``, tab-separated, with columns
``start_time, stop_time, speaker, value`` and speaker labels ``Ellie`` /
``Participant``. The official E-DAIC release instead ships one gzipped tarball
per participant (``{ID}_P.tar.gz``) containing, among the audio/visual feature
files, a comma-separated ``{ID}_Transcript.csv`` with columns
``Start_Time, End_Time, Speaker, Text``.

This script downloads each participant tarball, extracts *only* the transcript
(audio/visual features are discarded — we are text-only), converts it to the
DAIC-WOZ schema, and writes it to ``{out_dir}/raw/{ID}_TRANSCRIPT.csv`` so the
existing preprocessing/inference code runs unchanged for the zero-shot
generalization experiment (TCMIL trained on DAIC-WOZ, tested on E-DAIC).

Labels are NOT downloaded and are NOT shipped with the repo: place the official
AVEC 2019 / E-DAIC label splits (``train_split.csv``, ``dev_split.csv``,
``test_split.csv``, ``Detailed_PHQ8_Labels.csv``) in ``data/e-daic/labels/`` by
hand first (see README "Get the data"). As a convenience, if you instead drop the
official ``labels2019.tar.gz`` in ``data/e-daic/``, this script extracts the
splits from it.

Participant IDs are read from those label split CSVs, so only patients with a
known PHQ-8 label are fetched. The run is resumable: transcripts already on disk
are skipped.

Usage
-----
    python -m src.core.download_edaic                  # full dataset
    python -m src.core.download_edaic --splits test    # only the test split
    python -m src.core.download_edaic --workers 8      # parallel downloads

Reference: Ringeval et al., "AVEC 2019 Workshop and Challenge", AVEC '19.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# E-DAIC is access-controlled and may NOT be redistributed. Request access
# (see README "Get the data"); you receive a base URL for the participant
# tarballs. Paste it here, or pass it at runtime with --base-url.
PLACEHOLDER_BASE_URL = "PASTE_YOUR_APPROVED_EDAIC_BASE_URL_HERE"
DEFAULT_BASE_URL = PLACEHOLDER_BASE_URL
# Per-socket-op deadline (s) so a stalled stream raises instead of hanging.
SOCKET_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_BACKOFF = 5  # seconds, multiplied by attempt number
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "e-daic"
LABELS_TARBALL = "labels2019.tar.gz"
SPLIT_FILES = {
    "train": "train_split.csv",
    "dev": "dev_split.csv",
    "test": "test_split.csv",
}

# Column-name fuzzy matching: map E-DAIC headers -> DAIC-WOZ schema.
_START_KEYS = ("start_time", "start time", "starttime")
_STOP_KEYS = ("stop_time", "end_time", "stop time", "end time", "endtime", "stoptime")
_SPEAKER_KEYS = ("speaker",)
_VALUE_KEYS = ("value", "text", "utterance", "transcript")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def ensure_labels(out_dir: Path) -> Path:
    """Ensure the AVEC 2019 label splits are in ``{out_dir}/labels/``.

    Returns the labels directory. If the split CSVs are already there (placed by
    hand, see README), uses them as-is. Otherwise, as a fallback, extracts them
    from a manually-supplied ``labels2019.tar.gz`` — nothing is downloaded.
    """
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if all((labels_dir / f).exists() for f in SPLIT_FILES.values()):
        return labels_dir

    tarball = out_dir / LABELS_TARBALL
    if not tarball.exists():
        raise FileNotFoundError(
            f"No E-DAIC labels in {labels_dir} and no {tarball}. The labels are "
            "not shipped — place the split CSVs by hand (see README \"Get the "
            "data\") or drop the official labels2019.tar.gz in "
            f"{out_dir}."
        )

    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Flatten any leading "labels/" prefix.
            name = Path(member.name).name
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            (labels_dir / name).write_bytes(extracted.read())

    print(f"[labels] extracted into {labels_dir}")
    return labels_dir


def read_participant_ids(labels_dir: Path, splits: list[str]) -> list[int]:
    """Collect participant IDs from the requested label splits (sorted union)."""
    ids: set[int] = set()
    for split in splits:
        path = labels_dir / SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path)
        ids.update(int(pid) for pid in df["Participant_ID"].tolist())
    return sorted(ids)


# ---------------------------------------------------------------------------
# Transcript download + conversion
# ---------------------------------------------------------------------------

def _find_column(columns: list[str], keys: tuple[str, ...]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for key in keys:
        if key in lowered:
            return lowered[key]
    # Fallback: substring match.
    for low, orig in lowered.items():
        if any(key in low for key in keys):
            return orig
    return None


def convert_transcript(csv_bytes: bytes, pid: int) -> pd.DataFrame:
    """Convert raw E-DAIC transcript bytes to the DAIC-WOZ schema DataFrame.

    Output columns: start_time, stop_time, speaker, value (+ confidence if
    present). The conversion is faithful — one row per source ASR segment, no
    merging. Segmentation into MIL instances happens later in
    ``preprocess_raw.process_edaic_transcript`` (gap-based merge).

    E-DAIC ASR transcripts have NO speaker diarization (headers
    ``Start_Time, End_Time, Text, Confidence`` — the virtual agent Ellie's
    prompts are not transcribed). All rows are therefore the participant, and we
    label them ``Participant`` so the text-only pipeline has a valid schema. If a
    speaker column ever IS present, it is honored and normalized.
    """
    # E-DAIC transcripts are comma-separated; be tolerant of stray bad lines.
    df = pd.read_csv(io.BytesIO(csv_bytes), on_bad_lines="skip")

    cols = list(df.columns)
    start_c = _find_column(cols, _START_KEYS)
    stop_c = _find_column(cols, _STOP_KEYS)
    speaker_c = _find_column(cols, _SPEAKER_KEYS)
    value_c = _find_column(cols, _VALUE_KEYS)
    conf_c = _find_column(cols, ("confidence", "conf"))

    missing = [
        name
        for name, c in (
            ("start_time", start_c),
            ("stop_time", stop_c),
            ("value", value_c),
        )
        if c is None
    ]
    if missing:
        raise ValueError(
            f"Participant {pid}: transcript missing columns {missing}; "
            f"got headers {cols}"
        )

    if speaker_c is None:
        # ASR transcript with no diarization -> all participant speech.
        speaker = pd.Series(["Participant"] * len(df))
    else:
        speaker = df[speaker_c].astype(str).str.strip().replace(
            {"participant": "Participant", "ellie": "Ellie"}
        )

    out = pd.DataFrame({
        "start_time": df[start_c],
        "stop_time": df[stop_c],
        "speaker": speaker,
        "value": df[value_c],
    })
    if conf_c is not None:
        out["confidence"] = df[conf_c]
    return out


def download_one(pid: int, base_url: str, raw_dir: Path, skip_existing: bool) -> str:
    """Download, extract, and convert one participant's transcript.

    Returns a short status string for the run summary.
    """
    out_path = raw_dir / f"{pid}_TRANSCRIPT.csv"
    if skip_existing and out_path.exists():
        return f"{pid}: skip (exists)"

    url = f"{base_url.rstrip('/')}/{pid}_P.tar.gz"
    last_err = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Stream the tarball and stop as soon as the transcript member is read.
            # Each tarball is ~280 MB (audio + A/V features we discard); the
            # transcript sits early in the archive, so aborting the stream after
            # it avoids transferring the bulk of the file.
            #
            # ``timeout`` is the per-socket-operation deadline (seconds): if the
            # server stalls mid-stream and no bytes arrive within this window,
            # the read raises instead of blocking forever. Retries handle the
            # transient stalls the public E-DAIC host throws under concurrency.
            csv_bytes = None
            with urllib.request.urlopen(url, timeout=SOCKET_TIMEOUT) as resp:  # noqa: S310 (trusted host)
                with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue
                        name = Path(member.name).name.lower()
                        if "transcript" in name and name.endswith(".csv"):
                            extracted = tar.extractfile(member)
                            csv_bytes = extracted.read() if extracted else None
                            break  # close stream early; skip the rest of the tar

            if csv_bytes is None:
                return f"{pid}: ERROR no transcript in tarball"

            df = convert_transcript(csv_bytes, pid)
            df.to_csv(out_path, sep="\t", index=False)
            suffix = "" if attempt == 1 else f" [retry {attempt}]"
            return f"{pid}: ok ({len(df)} segments){suffix}"
        except Exception as e:  # noqa: BLE001 - report and continue per participant
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return f"{pid}: ERROR {last_err} (after {MAX_RETRIES} tries)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"E-DAIC base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output dataset dir (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_FILES),
        default=sorted(SPLIT_FILES),
        help="Which label splits to fetch (default: all).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads.")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download even if the transcript already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N participants (smoke test).",
    )
    args = parser.parse_args()

    if not args.base_url or args.base_url == PLACEHOLDER_BASE_URL:
        sys.exit(
            "ERROR: no E-DAIC base URL. The dataset is access-controlled and "
            "cannot be redistributed.\nRequest access (see README \"Get the "
            "data\"), then pass the approved URL with --base-url <URL> or set "
            "PLACEHOLDER_BASE_URL in this file."
        )

    out_dir: Path = args.out_dir
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = ensure_labels(out_dir)
    ids = read_participant_ids(labels_dir, args.splits)
    if args.limit is not None:
        ids = ids[: args.limit]

    print(
        f"[edaic] {len(ids)} participants from splits {args.splits} "
        f"-> {raw_dir} (workers={args.workers})"
    )

    skip_existing = not args.no_skip_existing
    n_ok = n_skip = n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, pid, args.base_url, raw_dir, skip_existing): pid
            for pid in ids
        }
        for fut in as_completed(futures):
            status = fut.result()
            print(f"  {status}")
            if "ERROR" in status:
                n_err += 1
            elif "skip" in status:
                n_skip += 1
            else:
                n_ok += 1

    print(f"[edaic] done: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
