"""Best-of-N face-similarity selector (Sprint 2.2-lite, 2026-07-04).

The Kaggle IP-Adapter sync path was retired (free T4s can't generate a
FLUX frame in production time). This module recovers most of the
face-consistency benefit at zero GPU cost: for hero scenes, generate N
schnell candidates and keep the one whose face is closest to the
character's approved master anchor (assets/character_anchors/).

Models: OpenCV Zoo YuNet (face detection, ~230KB) + SFace (face
embedding, ~37MB) — pure opencv-python + numpy, CPU inference in
milliseconds, no torch. Downloaded on first use into
assets/face_models/ (committed to the repo cache after first run, or
re-downloaded — both fine).

Guardrails (plan-review 2026-07-04 — the "missing face" trap):
  - Faces below _MIN_FACE_PX or detection score below _MIN_DET_SCORE are
    ignored (a 30px mid-ground face embeds pure noise).
  - A candidate with NO valid face scores 0.0 — never crashes.
  - If ALL candidates score 0.0 the caller keeps candidate 0 (the
    stable-seed baseline — i.e. today's behavior, no regression).
Every failure path degrades to "no selection" rather than raising.
"""
import os
import urllib.request

_MIN_FACE_PX = 50
_MIN_DET_SCORE = 0.80

_MODEL_DIR = os.path.join("assets", "face_models")
_YUNET = os.path.join(_MODEL_DIR, "face_detection_yunet_2023mar.onnx")
_SFACE = os.path.join(_MODEL_DIR, "face_recognition_sface_2021dec.onnx")
_URLS = {
    _YUNET: ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    _SFACE: ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}

_detector = None
_recognizer = None
_anchor_cache: dict = {}


def _ensure_models() -> bool:
    """Download the two ONNX models if absent. False on any failure."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for path, url in _URLS.items():
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            continue
        try:
            print(f"    [face-select] downloading {os.path.basename(path)}...")
            urllib.request.urlretrieve(url, path)
            if os.path.getsize(path) < 10_000:
                raise IOError("download too small — likely an LFS pointer")
        except Exception as e:
            print(f"    [face-select] model download failed ({str(e)[:80]}) "
                  f"— selector disabled this run")
            return False
    return True


def _load() -> bool:
    """Lazy-init detector + recognizer. False if cv2/models unavailable."""
    global _detector, _recognizer
    if _detector is not None and _recognizer is not None:
        return True
    try:
        import cv2
    except ImportError:
        print("    [face-select] opencv not installed — selector disabled")
        return False
    if not _ensure_models():
        return False
    try:
        _detector = cv2.FaceDetectorYN.create(
            _YUNET, "", (320, 320), score_threshold=_MIN_DET_SCORE)
        _recognizer = cv2.FaceRecognizerSF.create(_SFACE, "")
        return True
    except Exception as e:
        print(f"    [face-select] model init failed: {str(e)[:100]}")
        _detector = _recognizer = None
        return False


def _primary_face_embedding(bgr) -> "object | None":
    """Embedding of the LARGEST valid face in a BGR image, or None.
    Valid = detection score >= _MIN_DET_SCORE AND bbox >= _MIN_FACE_PX."""
    import cv2
    import numpy as np
    h, w = bgr.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    valid = [f for f in faces
             if f[2] >= _MIN_FACE_PX and f[3] >= _MIN_FACE_PX
             and f[-1] >= _MIN_DET_SCORE]
    if not valid:
        return None
    primary = max(valid, key=lambda f: f[2] * f[3])
    aligned = _recognizer.alignCrop(bgr, primary)
    feat = _recognizer.feature(aligned)
    return np.asarray(feat).flatten()


def get_anchor_embedding(hero: str) -> "object | None":
    """Embedding of the hero's approved master anchor (cached per run)."""
    key = (hero or "").strip().lower()
    if not key:
        return None
    if key in _anchor_cache:
        return _anchor_cache[key]
    emb = None
    path = os.path.join("assets", "character_anchors", f"{key}_anchor.png")
    if os.path.exists(path) and _load():
        try:
            import cv2
            bgr = cv2.imread(path)
            if bgr is not None:
                emb = _primary_face_embedding(bgr)
                if emb is None:
                    print(f"    [face-select] no valid face in anchor "
                          f"'{key}' — selector skipped for this hero")
        except Exception as e:
            print(f"    [face-select] anchor embed failed: {str(e)[:80]}")
    _anchor_cache[key] = emb
    return emb


def score_candidate(img_bytes: bytes, anchor_emb) -> float:
    """Cosine similarity of the candidate's primary face vs the anchor.
    0.0 for no-valid-face / any failure (never raises)."""
    if anchor_emb is None or not _load():
        return 0.0
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return 0.0
        emb = _primary_face_embedding(bgr)
        if emb is None:
            return 0.0
        a = np.asarray(anchor_emb, dtype=np.float64)
        b = np.asarray(emb, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception as e:
        print(f"    [face-select] scoring failed: {str(e)[:80]}")
        return 0.0
