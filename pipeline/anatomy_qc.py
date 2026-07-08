"""Anatomy QC gate (2026-07-07).

User forensic on x_XuuiIB-2I: an Arjuna bow-draw frame shipped with THREE
arms on the left side. Extra limbs are FLUX-schnell's signature failure in
complex action poses (crossed arms + bow + arrow lines at 8 steps), and
distilled schnell largely ignores negative prompts — so the fix is
DETECTION, not prompting: after each frame generates, Gemini Flash vision
counts the main figure's limbs; flagged frames are regenerated once with a
fresh seed by the caller (image_generator).

Design rules:
  - FAIL-OPEN everywhere: any API/parse/import problem returns (True, ...)
    — a render is never blocked or slowed more than one vision call.
  - Cheap: image is thumbnailed to ~512px before upload; flash-lite model;
    ~1-2s per frame, well inside free-tier RPD across the key cascade.
  - Env: ANATOMY_QC=false disables (caller checks), ANATOMY_QC_MODEL
    overrides the model.
"""
import os

# Calibration (2026-07-07): STRUCTURAL errors only. The first draft also
# flagged finger-count issues and flagged 8/10 frames of a real video —
# finger errors are ubiquitous in distilled FLUX output, nearly invisible
# at video speed, and unreliably counted by vision models (channel doctrine:
# bias prompts away from hands, don't fight fingers). Extra ARMS/legs/heads
# and merged bodies are the artifacts viewers actually notice.
_PROMPT = (
    "You are an image quality checker for AI-generated images of human "
    "(or humanoid divine) figures. Look ONLY at the most prominent figure. "
    "Flag ONLY these errors: (1) three or more arms, "
    "(2) three or more legs, (3) two or more heads, (4) two bodies merged "
    "into one, (5) a limb growing from an impossible place (e.g. an arm "
    "emerging from the chest or head), (6) EXPOSED NUDITY on a FEMALE "
    "figure: visible bare breast or nipple, or fabric so sheer it "
    "clearly reveals them (a bare-chested MALE warrior is traditional "
    "attire and always acceptable), or exposed genitals on anyone, "
    "(7) ONLY when the face is LARGE in frame (close-up): eyes that are "
    "blank voids or severely malformed with no visible iris or pupil. "
    "DO NOT flag finger or hand details, "
    "face quality, proportions, or style. Ornaments, weapons, bows, quivers, "
    "arrows, cloth folds and background objects are NOT limbs — do not "
    "confuse a bow, arrow shaft or drape with an arm. If the figure is a "
    "dark silhouette, partially out of frame, or too small to judge "
    "confidently, answer OK. When in doubt, answer OK. Answer with EXACTLY "
    "one line: 'OK' or 'BAD: <short reason>'."
)


def check_anatomy_confirmed(image_path: str) -> tuple[bool, str]:
    """Vote-confirmed check. Single verdicts are unstable on stylized frames
    (ground-truthed 2026-07-07: a clean Krishna frame flipped OK->BAD between
    runs). Strategy: 1 call; only if it says BAD, spend 2 more calls and flag
    only when >=2/3 agree BAD. Most frames cost exactly one call."""
    ok1, why1 = check_anatomy(image_path)
    if ok1:
        return True, why1
    votes_bad, reasons = 1, [why1]
    for _ in range(2):
        ok_n, why_n = check_anatomy(image_path)
        if not ok_n:
            votes_bad += 1
            reasons.append(why_n)
    if votes_bad >= 2:
        return False, f"{votes_bad}/3 votes: {reasons[0][:100]}"
    return True, f"unconfirmed flag (1/3): {why1[:80]}"


def check_anatomy(image_path: str) -> tuple[bool, str]:
    """Returns (ok, reason). ok=True also on any infrastructure failure
    (fail-open — QC must never cost a render)."""
    try:
        from PIL import Image
        from google import genai

        from pipeline.script_generator import _gemini_keys

        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 912), Image.LANCZOS)

        model = os.environ.get("ANATOMY_QC_MODEL", "gemini-2.5-flash").strip()
        last_err = "no gemini keys configured"
        for label, key in _gemini_keys():
            try:
                client = genai.Client(api_key=key)
                resp = client.models.generate_content(
                    model=model, contents=[_PROMPT, img])
                text = (getattr(resp, "text", "") or "").strip()
                up = text.upper()
                if up.startswith("OK"):
                    return True, "ok"
                if up.startswith("BAD"):
                    return False, text[:140]
                # Unparseable answer → fail-open
                return True, f"unparsed:{text[:60]}"
            except Exception as e:
                last_err = f"{label}: {str(e)[:80]}"
                continue
        return True, f"qc-unavailable ({last_err})"
    except Exception as e:
        return True, f"qc-error ({str(e)[:80]})"
