"""Lightweight speaker diarization for a known speaker count.

Uses ffmpeg (already in the image) to decode audio, extracts short-time
spectral features per Whisper segment, then clusters into N speakers with
scikit-learn. Labels are remapped to Person 1..N by first appearance order.

Note: pyannote was removed because the torch/CUDA stack ballooned the Docker
image to ~11GB. This fallback stays lean and works offline with no HF token.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from app.models.schemas import TranscriptSegment

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000


def _person_labels_only(
    segments: list[TranscriptSegment], label: str = "Person 1"
) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=seg.start, end=seg.end, text=seg.text, speaker=label)
        for seg in segments
    ]


def _load_mono_pcm(audio_path: Path, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed to decode audio: {stderr.strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _segment_feature(samples: np.ndarray) -> np.ndarray:
    if samples.size < 256:
        samples = np.pad(samples, (0, 256 - samples.size))

    frame = 512
    hop = 256
    window = np.hanning(frame).astype(np.float32)
    spectra: list[np.ndarray] = []
    for start in range(0, max(1, samples.size - frame + 1), hop):
        chunk = samples[start : start + frame]
        if chunk.size < frame:
            chunk = np.pad(chunk, (0, frame - chunk.size))
        spectrum = np.abs(np.fft.rfft(chunk * window))
        spectra.append(np.log1p(spectrum))

    stacked = np.stack(spectra, axis=0)
    keep = max(16, stacked.shape[1] // 2)
    mean_spec = stacked[:, :keep].mean(axis=0)
    energy = float(np.sqrt(np.mean(samples**2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2.0)
    return np.concatenate([mean_spec, np.array([energy, zcr], dtype=np.float32)])


def assign_speakers(
    audio_path: Path,
    segments: list[TranscriptSegment],
    num_speakers: int,
    *,
    hf_token: Optional[str] = None,
    device: str = "cpu",
) -> list[TranscriptSegment]:
    """Attach Person N labels to Whisper segments."""
    del hf_token, device  # kept for API compatibility with Transcriber

    if not segments:
        return segments

    n = max(1, min(10, int(num_speakers)))
    if n == 1:
        return _person_labels_only(segments, "Person 1")

    try:
        audio = _load_mono_pcm(audio_path)
    except Exception:
        logger.exception("Diarization decode failed; labeling all as Person 1")
        return _person_labels_only(segments, "Person 1")

    features: list[np.ndarray] = []
    usable_idx: list[int] = []
    for i, seg in enumerate(segments):
        start = max(0, int(seg.start * _SAMPLE_RATE))
        end = min(audio.size, int(seg.end * _SAMPLE_RATE))
        if end <= start:
            continue
        features.append(_segment_feature(audio[start:end]))
        usable_idx.append(i)

    if len(usable_idx) < 2:
        return _person_labels_only(segments, "Person 1")

    x = StandardScaler().fit_transform(np.stack(features))
    k = min(n, len(usable_idx))
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(x)

    appearance: dict[int, int] = {}
    for cluster in labels:
        cid = int(cluster)
        if cid not in appearance:
            appearance[cid] = len(appearance) + 1

    labeled = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text, speaker=None)
        for s in segments
    ]
    for local_i, seg_i in enumerate(usable_idx):
        labeled[seg_i] = TranscriptSegment(
            start=segments[seg_i].start,
            end=segments[seg_i].end,
            text=segments[seg_i].text,
            speaker=f"Person {appearance[int(labels[local_i])]}",
        )

    last = "Person 1"
    for i, seg in enumerate(labeled):
        if seg.speaker:
            last = seg.speaker
        else:
            labeled[i] = TranscriptSegment(
                start=seg.start, end=seg.end, text=seg.text, speaker=last
            )
    return labeled
