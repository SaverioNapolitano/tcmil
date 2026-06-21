"""Download DAIC-WOZ transcripts (text-only) for reproducibility.

This is the DAIC-WOZ analogue of ``download_edaic.py``. The text-only pipeline
(``preprocess_raw.py`` / ``dataset.py``) expects each raw transcript at
``{out_dir}/raw/{ID}_TRANSCRIPT.csv``, tab-separated, with columns
``start_time, stop_time, speaker, value`` and speaker labels ``Ellie`` /
``Participant``.

The official DAIC-WOZ release ships one zip per participant
(``{ID}_P.zip``) containing the audio/visual feature files **and** a
``{ID}_TRANSCRIPT.csv`` that is *already* in exactly that DAIC-WOZ schema. So,
unlike E-DAIC (which needs column remapping and has no diarization), DAIC-WOZ
transcripts are extracted verbatim — no conversion. This script downloads each
participant zip, extracts *only* the transcript (audio/visual features are
discarded — we are text-only), and writes it to ``{out_dir}/raw/{ID}_TRANSCRIPT.csv``.

Unlike E-DAIC tarballs (``.tar.gz``, streamable so we can abort the transfer as
soon as the transcript member is read), DAIC-WOZ ships ``.zip`` archives whose
central directory lives at the *end* of the file — there is no reliable way to
stop early, so each archive is downloaded in full to a temp file, the transcript
is extracted, and the temp file is deleted.

Labels are NOT downloaded: ``data/daic-woz/labels/`` already bundles the
official AVEC 2017 split CSVs (``train_split_Depression_AVEC2017.csv``,
``dev_split_Depression_AVEC2017.csv``, ``full_test_split.csv``). Participant IDs
are read from those split CSVs, so only patients with a known PHQ-8 label are
fetched. The run is resumable: transcripts already on disk are skipped.

Usage
-----
    python -m src.core.download_daicwoz                  # full dataset
    python -m src.core.download_daicwoz --splits test    # only the test split
    python -m src.core.download_daicwoz --workers 8      # parallel downloads

Reference: Gratch et al., "The Distress Analysis Interview Corpus of human and
computer interviews", LREC 2014; Valstar et al., "AVEC 2016", AVEC '16.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# DAIC-WOZ is access-controlled and may NOT be redistributed. Request access
# (see README "Get the data"); you receive a base URL for the participant zips.
# Paste it here, or pass it at runtime with --base-url.
PLACEHOLDER_BASE_URL = "PASTE_YOUR_APPROVED_DAICWOZ_BASE_URL_HERE"
DEFAULT_BASE_URL = PLACEHOLDER_BASE_URL
# Per-socket-op deadline (s) so a stalled stream raises instead of hanging.
SOCKET_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_BACKOFF = 5  # seconds, multiplied by attempt number
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "daic-woz"
SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}


# ---------------------------------------------------------------------------
# Labels (already bundled — used only to enumerate participant IDs)
# ---------------------------------------------------------------------------

def read_participant_ids(labels_dir: Path, splits: list[str]) -> list[int]:
    """Collect participant IDs from the requested label splits (sorted union)."""
    ids: set[int] = set()
    for split in splits:
        path = labels_dir / SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split file: {path}. Expected the bundled AVEC 2017 "
                f"DAIC-WOZ labels under {labels_dir}."
            )
        df = pd.read_csv(path)
        ids.update(int(pid) for pid in df["Participant_ID"].tolist())
    return sorted(ids)


# ---------------------------------------------------------------------------
# Transcript download + extraction
# ---------------------------------------------------------------------------

def download_one(pid: int, base_url: str, raw_dir: Path, skip_existing: bool) -> str:
    """Download one participant's zip and extract its transcript verbatim.

    DAIC-WOZ ``{ID}_TRANSCRIPT.csv`` is already in the target DAIC-WOZ TSV
    schema, so it is written out unchanged. Returns a short status string for
    the run summary.
    """
    out_path = raw_dir / f"{pid}_TRANSCRIPT.csv"
    if skip_existing and out_path.exists():
        return f"{pid}: skip (exists)"

    url = f"{base_url.rstrip('/')}/{pid}_P.zip"
    last_err = "unknown"
    for attempt in range(1, MAX_RETRIES + 1):
        tmp_path = None
        try:
            # Zip central directory is at the end of the file, so (unlike the
            # E-DAIC tarballs) we cannot stop the transfer early — download the
            # whole archive to a temp file, then extract the single transcript.
            #
            # ``timeout`` is the per-socket-operation deadline (seconds): if the
            # server stalls mid-stream and no bytes arrive within this window,
            # the read raises instead of blocking forever. Retries handle the
            # transient stalls the public USC host throws under concurrency.
            with urllib.request.urlopen(url, timeout=SOCKET_TIMEOUT) as resp:  # noqa: S310 (trusted host)
                with tempfile.NamedTemporaryFile(
                    suffix=".zip", delete=False, dir=raw_dir
                ) as tmp:
                    tmp_path = Path(tmp.name)
                    while True:
                        chunk = resp.read(1 << 20)  # 1 MiB
                        if not chunk:
                            break
                        tmp.write(chunk)

            with zipfile.ZipFile(tmp_path) as zf:
                member = next(
                    (
                        n
                        for n in zf.namelist()
                        if Path(n).name.lower().endswith("_transcript.csv")
                    ),
                    None,
                )
                if member is None:
                    return f"{pid}: ERROR no transcript in zip"
                csv_bytes = zf.read(member)

            # DAIC-WOZ transcripts are already tab-separated in the target
            # schema; write verbatim. Parse once only to report row count.
            out_path.write_bytes(csv_bytes)
            n_rows = sum(1 for _ in csv_bytes.splitlines()) - 1
            suffix = "" if attempt == 1 else f" [retry {attempt}]"
            return f"{pid}: ok ({n_rows} segments){suffix}"
        except Exception as e:  # noqa: BLE001 - report and continue per participant
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
    return f"{pid}: ERROR {last_err} (after {MAX_RETRIES} tries)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"DAIC-WOZ base URL (default: {DEFAULT_BASE_URL})",
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
            "ERROR: no DAIC-WOZ base URL. The dataset is access-controlled and "
            "cannot be redistributed.\nRequest access (see README \"Get the "
            "data\"), then pass the approved URL with --base-url <URL> or set "
            "PLACEHOLDER_BASE_URL in this file."
        )

    out_dir: Path = args.out_dir
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"

    ids = read_participant_ids(labels_dir, args.splits)
    if args.limit is not None:
        ids = ids[: args.limit]

    print(
        f"[daic-woz] {len(ids)} participants from splits {args.splits} "
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

    print(f"[daic-woz] done: {n_ok} ok, {n_skip} skipped, {n_err} errors")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
