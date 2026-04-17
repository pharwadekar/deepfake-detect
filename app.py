from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import face_recognition
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from fakers import HybridDeepfakeDetector, extract_fft  # noqa: E402


Mode = Literal["hybrid", "rgb", "fft"]

PAGE_TITLE = "DeepFakeDetect"
SAMPLED_FRAMES = 15
TOP_FRAMES = 4
DEFAULT_MODE: Mode = "hybrid"
CHECKPOINTS = {
    "rgb": "model_rgb_20260416_091501_9",
    "fft": "model_fft_20260416_095640_8",
    "hybrid": "model_hybrid_20260416_102842_7",
}
MODE_LABELS = {"hybrid": "Hybrid", "rgb": "RGB", "fft": "FFT"}
TRANSFORM = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


@dataclass
class FramePrediction:
    frame_index: int
    timestamp_seconds: float
    score: float
    face_image: np.ndarray


@dataclass
class AnalysisResult:
    filename: str
    mode: Mode
    score: float
    verdict: str
    confidence: float
    frames_analyzed: int
    frames_sampled: int
    face_coverage: float
    suspicious_frames: list[FramePrediction]
    notes: list[str]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(155, 112, 255, 0.22), transparent 34%),
                radial-gradient(circle at 85% 18%, rgba(206, 114, 255, 0.12), transparent 20%),
                linear-gradient(180deg, #090811 0%, #0f0d17 42%, #13121d 100%);
            color: #f7f2ff;
        }

        header[data-testid="stHeader"] {
            background: linear-gradient(180deg, rgba(13, 11, 22, 0.96) 0%, rgba(13, 11, 22, 0.82) 100%);
            border-bottom: 1px solid rgba(193, 165, 255, 0.12);
            position: relative;
            top: auto;
            height: auto;
            z-index: auto;
        }

        [data-testid="stToolbar"] {
            visibility: visible;
            height: auto;
            position: static;
        }

        [data-testid="stToolbar"] button,
        [data-testid="stToolbar"] div,
        [data-testid="stToolbar"] span {
            color: #f3ecff !important;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2.6rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, p, label, span, div {
            color: #f7f2ff;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(17, 16, 28, 0.86);
            border: 1px solid rgba(193, 165, 255, 0.24);
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            min-height: 170px;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(193, 165, 255, 0.42);
            background: rgba(21, 20, 34, 0.92);
        }

        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderInstructions"] span {
            color: #ece4ff !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            min-height: 3.6rem;
            min-width: 460px;
            background: linear-gradient(135deg, #9f6fff 0%, #d79cff 100%);
            color: #120d18;
            border: 0;
            border-radius: 18px;
            font-weight: 800;
            font-size: 1rem;
            box-shadow: 0 18px 36px rgba(149, 104, 255, 0.4);
        }

        [data-testid="stFileUploaderFile"] {
            opacity: 0 !important;
            pointer-events: none !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            overflow: hidden !important;
        }

        [data-testid="stFileUploaderFile"] > div {
            width: 0 !important;
            min-width: 0 !important;
        }

        [data-testid="stFileUploaderFile"] > div > div {
            width: 0 !important;
            min-width: 0 !important;
        }

        [data-testid="stFileUploaderFile"],
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderFile"] div,
        [data-testid="stFileUploaderFile"] span,
        [data-testid="stFileUploaderFile"] small,
        [data-testid="stFileUploaderFile"] p,
        [data-testid="stFileUploaderFile"] a,
        [data-testid="stFileUploaderFile"] label {
            color: #17131f !important;
            -webkit-text-fill-color: #17131f !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileSize"] {
            color: #17131f !important;
            -webkit-text-fill-color: #17131f !important;
            fill: #17131f !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploader"] section {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        [data-testid="stFileUploader"] section > div {
            width: 100%;
        }

        [data-testid="stFileUploaderDropzone"] > div {
            max-width: 920px;
            margin-left: auto;
            margin-right: auto;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
            text-align: center;
            width: 100%;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] div {
            justify-content: center;
            text-align: center;
            width: 100%;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderInstructions"] small,
        [data-testid="stFileUploaderInstructions"] span {
            display: block;
            width: 100%;
            text-align: center !important;
            margin-left: auto;
            margin-right: auto;
        }

        [data-testid="stFileUploaderFileName"] {
            font-size: 1.2rem;
            font-weight: 760;
            line-height: 1.3;
        }

        [data-testid="stFileUploaderFileSize"] {
            font-size: 1rem;
            color: #17131f !important;
        }

        [data-testid="stFileUploaderFile"] svg {
            width: 1.55rem;
            height: 1.55rem;
        }

        [data-testid="stFileUploaderDeleteBtn"] {
            background: rgba(141, 99, 255, 0.18);
            border-radius: 999px;
        }

        [data-testid="stFileUploaderDeleteBtn"]:hover {
            background: rgba(141, 99, 255, 0.28);
        }

        [data-testid="stFileUploader"] section button[kind="secondary"] {
            color: #f3ecff !important;
            border-color: rgba(193, 165, 255, 0.24) !important;
            background: rgba(141, 99, 255, 0.12) !important;
        }

        [data-testid="stFileUploader"] section [aria-label="+"] {
            display: none !important;
        }

        [data-testid="stFileUploader"] section > button:not([data-testid]),
        [data-testid="stFileUploader"] section > div + button {
            display: none !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            background: linear-gradient(135deg, #ae82ff 0%, #e3b0ff 100%);
            color: #100c18;
        }

        [data-testid="stFileUploader"] small {
            text-align: center;
            display: block;
        }

        div.stButton > button {
            min-height: 3.6rem;
            border: 0;
            border-radius: 18px;
            background: linear-gradient(135deg, #9f6fff 0%, #d79cff 100%);
            color: #120d18;
            font-size: 1rem;
            font-weight: 800;
            box-shadow: 0 18px 36px rgba(149, 104, 255, 0.4);
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, #ae82ff 0%, #e3b0ff 100%);
            color: #100c18;
        }

        div.stButton > button:focus:not(:active) {
            border: 0;
            box-shadow: 0 0 0 0.2rem rgba(194, 155, 255, 0.35);
        }

        [data-testid="stSelectbox"] > div {
            background: rgba(17, 16, 28, 0.86);
            border: 1px solid rgba(193, 165, 255, 0.24);
            border-radius: 16px;
        }

        [data-testid="stSelectbox"] * {
            color: #f7f2ff !important;
        }

        .hero {
            text-align: center;
            margin-bottom: 1.6rem;
        }

        .brand-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.55rem 1rem 0.55rem 0.65rem;
            border-radius: 999px;
            background: rgba(20, 19, 32, 0.9);
            border: 1px solid rgba(191, 166, 255, 0.18);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.22);
            margin-bottom: 1.35rem;
        }

        .brand-badge {
            width: 2.2rem;
            height: 2.2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #8d63ff 0%, #d09dff 100%);
            color: #0f0c17;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .brand-name {
            font-weight: 700;
            color: #f8f3ff;
            letter-spacing: 0.01em;
        }

        .hero h1 {
            font-size: clamp(2.3rem, 5vw, 3.5rem);
            line-height: 1.05;
            font-weight: 800;
            margin: 0 0 0.9rem;
            letter-spacing: -0.03em;
        }

        .hero p {
            max-width: 840px;
            margin: 0 auto;
            color: #d8cff0;
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .section-card {
            background: rgba(18, 17, 29, 0.82);
            border: 1px solid rgba(193, 165, 255, 0.18);
            border-radius: 26px;
            padding: 1.45rem 1.5rem 1.2rem;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(14px);
            margin-bottom: 1rem;
        }

        .section-kicker {
            color: #bfaaff;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
            text-align: center;
        }

        .section-card h2, .section-card h3 {
            margin: 0;
            font-size: 1.35rem;
            font-weight: 750;
            text-align: center;
        }

        .section-copy {
            margin-top: 0.55rem;
            color: #d3caeb;
            line-height: 1.7;
            text-align: center;
        }

        .mode-caption {
            text-align: center;
            color: #bfb6d8;
            font-size: 0.92rem;
            margin-top: 0.4rem;
            margin-bottom: 1rem;
        }

        .center-title {
            text-align: center;
            margin: 2rem 0 0.35rem;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .center-caption {
            text-align: center;
            color: #beb4d8;
            margin-bottom: 1.2rem;
        }

        .loading-state {
            text-align: center;
            color: #efe8ff;
            font-size: 1.05rem;
            padding: 1rem 0 1.3rem;
        }

        .verdict-card {
            border-radius: 26px;
            padding: 1.45rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.28);
        }

        .verdict-card.fake {
            background: linear-gradient(135deg, rgba(250, 83, 175, 0.26), rgba(143, 78, 255, 0.28));
        }

        .verdict-card.real {
            background: linear-gradient(135deg, rgba(56, 183, 121, 0.26), rgba(32, 112, 91, 0.3));
        }

        .verdict-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(255, 255, 255, 0.76);
            margin-bottom: 0.35rem;
            text-align: center;
        }

        .verdict-value {
            font-size: clamp(2rem, 4vw, 2.8rem);
            font-weight: 800;
            margin-bottom: 0.35rem;
            text-align: center;
        }

        .verdict-copy {
            color: rgba(247, 242, 255, 0.9);
            font-size: 1rem;
            line-height: 1.6;
            text-align: center;
        }

        .summary-title {
            font-size: 1.15rem;
            font-weight: 760;
            margin-bottom: 0.5rem;
            text-align: center;
        }

        .summary-copy {
            color: #ddd4f4;
            line-height: 1.7;
            margin-bottom: 0.95rem;
            text-align: center;
        }

        .metric-line {
            color: #f6f1ff;
            font-size: 0.98rem;
            line-height: 1.8;
            text-align: center;
        }

        .notes-card {
            text-align: center;
            background: rgba(18, 17, 29, 0.82);
            border: 1px solid rgba(193, 165, 255, 0.18);
            border-radius: 22px;
            padding: 1.15rem 1.2rem;
            margin-top: 1rem;
        }

        .notes-card h4 {
            margin: 0 0 0.55rem;
            font-size: 1rem;
        }

        .notes-card p {
            color: #dbd2f0;
            margin: 0.35rem 0;
        }

        .upload-file-card {
            max-width: 920px;
            margin: 1.15rem auto 0.85rem;
            background: rgba(18, 17, 29, 0.92);
            border: 1px solid rgba(193, 165, 255, 0.2);
            border-radius: 24px;
            padding: 1.25rem 1.5rem;
            text-align: center;
            box-shadow: 0 18px 38px rgba(0, 0, 0, 0.22);
        }

        .upload-file-label {
            color: #bfaaff;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.42rem;
        }

        .upload-file-name {
            font-size: 1.3rem;
            font-weight: 760;
            color: #f6f0ff;
            margin-bottom: 0.3rem;
            word-break: break-word;
        }

        .upload-file-meta {
            color: #d7cdec;
            font-size: 1.02rem;
        }

        .frame-caption {
            text-align: center;
            color: #d9d0ef;
            font-size: 0.9rem;
            margin-top: 0.45rem;
            margin-bottom: 1rem;
            line-height: 1.45;
        }

        .stImage img {
            border-radius: 18px;
            border: 1px solid rgba(193, 165, 255, 0.18);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mode_display_name(mode: Mode) -> str:
    return MODE_LABELS[mode]


def find_checkpoint(mode: Mode) -> Path:
    preferred = ROOT / "models" / CHECKPOINTS[mode]
    if preferred.exists():
        return preferred

    matches = sorted((ROOT / "models").glob(f"model_{mode}_*"))
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for mode '{mode}'.")
    return matches[-1]


@st.cache_resource(show_spinner=False)
def load_model(mode: Mode) -> tuple[HybridDeepfakeDetector, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridDeepfakeDetector(mode=mode).to(device)
    checkpoint_path = find_checkpoint(mode)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def preprocess_face(face_image: np.ndarray, mode: Mode) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    rgb_tensor = None
    fft_tensor = None

    if mode in ("rgb", "hybrid"):
        rgb_tensor = TRANSFORM(image=face_image)["image"]

    if mode in ("fft", "hybrid"):
        fft_image = extract_fft(cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
        fft_tensor = TRANSFORM(image=cv2.cvtColor(fft_image, cv2.COLOR_BGR2RGB))["image"]

    return rgb_tensor, fft_tensor


def sample_video_frames(video_path: Path, num_frames: int) -> list[tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0:
        cap.release()
        raise ValueError("The uploaded video has no readable frames.")

    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0

    indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    sampled_frames: list[tuple[int, float, np.ndarray]] = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled_frames.append((int(frame_index), frame_index / fps, rgb_frame))

    cap.release()
    return sampled_frames


def crop_first_face(frame: np.ndarray) -> np.ndarray | None:
    locations = face_recognition.face_locations(frame)
    if not locations:
        return None

    top, right, bottom, left = locations[0]
    height, width = frame.shape[:2]
    top = max(0, top)
    left = max(0, left)
    bottom = min(height, bottom)
    right = min(width, right)

    if top >= bottom or left >= right:
        return None

    return frame[top:bottom, left:right]


def analyze_video(video_path: Path, filename: str, mode: Mode) -> AnalysisResult:
    sampled_frames = sample_video_frames(video_path, SAMPLED_FRAMES)
    if not sampled_frames:
        raise ValueError("No frames could be sampled from the uploaded video.")

    model, device = load_model(mode)
    predictions: list[FramePrediction] = []

    for frame_index, timestamp_seconds, frame in sampled_frames:
        face_crop = crop_first_face(frame)
        if face_crop is None or face_crop.size == 0:
            continue

        rgb_tensor, fft_tensor = preprocess_face(face_crop, mode)
        rgb_batch = rgb_tensor.unsqueeze(0).to(device) if rgb_tensor is not None else None
        fft_batch = fft_tensor.unsqueeze(0).to(device) if fft_tensor is not None else None

        with torch.inference_mode():
            raw_score = model(rgb_x=rgb_batch, fft_x=fft_batch)
            score = float(raw_score.item() if hasattr(raw_score, "item") else raw_score)

        predictions.append(
            FramePrediction(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                score=max(0.0, min(1.0, score)),
                face_image=face_crop,
            )
        )

    if not predictions:
        raise ValueError("No faces were detected in the sampled frames.")

    final_score = float(np.mean([prediction.score for prediction in predictions]))
    final_score = max(0.0, min(1.0, final_score))
    verdict = "Fake" if final_score >= 0.5 else "Not Fake"
    confidence = min(abs(final_score - 0.5) * 200.0, 99.9)
    face_coverage = (len(predictions) / max(len(sampled_frames), 1)) * 100.0

    notes: list[str] = []
    if face_coverage < 50:
        notes.append("Face coverage is limited, so the result may be less reliable than usual.")
    if len(predictions) < 5:
        notes.append("Only a small number of face crops were analyzed from this video.")
    if len(predictions) < len(sampled_frames):
        notes.append("Some sampled frames did not contain a detectable face and were skipped.")

    suspicious_frames = sorted(predictions, key=lambda item: item.score, reverse=True)[:TOP_FRAMES]

    return AnalysisResult(
        filename=filename,
        mode=mode,
        score=final_score,
        verdict=verdict,
        confidence=confidence,
        frames_analyzed=len(predictions),
        frames_sampled=len(sampled_frames),
        face_coverage=face_coverage,
        suspicious_frames=suspicious_frames,
        notes=notes,
    )


def save_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="brand-chip">
                <span class="brand-badge">DF</span>
                <span class="brand-name">got real one anot?</span>
            </div>
            <h1>Got doubts if it's for real or not?</h1>
            <p>
                A simple deepfake detection interface for quick checks.<br />
                Upload a clip and leave it to us tell if got real or anot!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_intro() -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Upload</div>
            <div class="section-copy">
                Add a video file to begin
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> None:
    size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(
        f"""
        <div class="upload-file-card">
            <div class="upload-file-label">Uploaded file</div>
            <div class="upload-file-name">{uploaded_file.name}</div>
            <div class="upload-file-meta">{size_mb:.2f} MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_uploaded_file_from_state(filename: str, size_bytes: int) -> None:
    size_mb = size_bytes / (1024 * 1024)
    st.markdown(
        f"""
        <div class="upload-file-card">
            <div class="upload-file-label">Uploaded file</div>
            <div class="upload-file-name">{filename}</div>
            <div class="upload-file-meta">{size_mb:.2f} MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(result: AnalysisResult) -> None:
    verdict_class = "fake" if result.verdict == "Fake" else "real"
    summary_text = (
        "The system found enough deepfake-like signal in the uploaded video to mark it as fake."
        if result.verdict == "Fake"
        else "The system did not find strong deepfake-like signal in the uploaded video, so it is marked as not fake."
    )

    st.markdown('<div class="center-title">Results</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="center-caption">Latest analysis: {result.filename}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="verdict-card {verdict_class}">
            <div class="verdict-label">Verdict</div>
            <div class="verdict-value">{result.verdict}</div>
            <div class="verdict-copy">
                Deepfake score: {result.score * 100:.1f}%. This answer is based on the selected Hybrid analysis mode.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-card">
            <div class="summary-title">What this result means</div>
            <div class="summary-copy">{summary_text}</div>
            <div class="metric-line">
                Result: {result.verdict} | Deepfake score: {result.score * 100:.1f}% | Confidence: {result.confidence:.1f}% |
                Frames analyzed: {result.frames_analyzed} | Face coverage: {result.face_coverage:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="summary-title">Key frames</div>
            <div class="summary-copy">
                The highest-scoring detected face crops from the latest analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.suspicious_frames:
        columns = st.columns(len(result.suspicious_frames))
        for column, frame in zip(columns, result.suspicious_frames):
            with column:
                st.image(frame.face_image, width="stretch")
                st.markdown(
                    f"""
                    <div class="frame-caption">
                        Frame {frame.frame_index} | {frame.timestamp_seconds:.2f}s | Deepfake score {frame.score * 100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No suspicious face crops were available to display.")

    if result.notes:
        notes_markup = "".join(f"<p>{note}</p>" for note in result.notes)
        st.markdown(
            f"""
            <div class="notes-card">
                <h4>Reliability notes</h4>
                {notes_markup}
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    inject_styles()
    render_header()
    render_upload_intro()

    if "uploader_version" not in st.session_state:
        st.session_state["uploader_version"] = 0
    if "uploaded_video_bytes" not in st.session_state:
        st.session_state["uploaded_video_bytes"] = None
        st.session_state["uploaded_video_name"] = None
        st.session_state["uploaded_video_size"] = None

    uploader_key = f"video_uploader_{st.session_state['uploader_version']}"
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
        key=uploader_key,
    )

    if uploaded_file:
        st.session_state["uploaded_video_bytes"] = uploaded_file.getvalue()
        st.session_state["uploaded_video_name"] = uploaded_file.name
        st.session_state["uploaded_video_size"] = uploaded_file.size
        st.session_state["uploader_version"] += 1
        st.rerun()

    current_video_bytes = st.session_state.get("uploaded_video_bytes")
    current_video_name = st.session_state.get("uploaded_video_name")
    current_video_size = st.session_state.get("uploaded_video_size")

    if current_video_bytes and current_video_name and current_video_size is not None:
        render_uploaded_file_from_state(current_video_name, current_video_size)

        analyze_clicked = st.button("Analyze Video", width="stretch")
        if analyze_clicked:
            status_placeholder = st.empty()
            status_placeholder.markdown(
                '<div class="loading-state">Analyzing your video...</div>',
                unsafe_allow_html=True,
            )

            video_path: Path | None = None
            try:
                suffix = Path(current_video_name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(current_video_bytes)
                    video_path = Path(temp_file.name)

                result = analyze_video(video_path, current_video_name, DEFAULT_MODE)
                st.session_state["latest_result"] = result
                st.session_state["latest_upload_name"] = current_video_name
                status_placeholder.empty()
            except Exception as error:
                status_placeholder.empty()
                st.error(str(error))
            finally:
                if video_path and video_path.exists():
                    video_path.unlink(missing_ok=True)

        latest_result = st.session_state.get("latest_result")
        latest_upload_name = st.session_state.get("latest_upload_name")
        if latest_result and latest_upload_name == current_video_name:
            render_results(latest_result)


if __name__ == "__main__":
    main()
