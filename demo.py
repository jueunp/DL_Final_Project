"""
AudioDream — Sound to Living Scene
====================================
오디오 한 클립 → 움직이는 장면 영상

Pipeline:
  🎤 Audio (mic / file)
      ↓
  [1] Qwen2-Audio-7B-Instruct  : 오디오 이해 → 장면 묘사 + SDXL 프롬프트 생성
      ↓
  [2] SDXL-Turbo               : 프롬프트 → 이미지 생성 (4 steps, ~3초)
      ↓
  [3] Depth Anything V2        : 이미지 → depth map 추출
      ↓
  parallax renderer            : depth-aware 2.5D 움직이는 영상 (OpenCV)
      ↓
  🎬 Output video + 원본 오디오 합성
"""

import json
import os
import re
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image

# ──────────────────────────────────────────────
# 0. Device
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"[INFO] Device: {DEVICE} | dtype: {DTYPE}")

# ──────────────────────────────────────────────
# 1. Qwen2-Audio  — audio understanding + prompt generation
# ──────────────────────────────────────────────
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

print("[1/3] Loading Qwen2-Audio-7B-Instruct...")
qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    torch_dtype=DTYPE,
    device_map="auto",
)

QWEN_SYSTEM_PROMPT = """You are an expert at analyzing audio and translating it into vivid visual scenes.
When given audio, you must respond ONLY with a valid JSON object (no markdown, no explanation) in this exact format:
{
  "scene_description": "one sentence describing what this audio evokes visually",
  "mood": "one word: e.g. peaceful / tense / joyful / melancholic / epic / eerie / energetic",
  "sdxl_prompt": "a detailed visual prompt for image generation, describing colors, lighting, atmosphere, objects, style",
  "negative_prompt": "blurry, low quality, text, watermark, ugly, deformed"
}"""

def analyze_audio(audio_path: str) -> dict:
    """Run Qwen2-Audio on audio file → returns scene dict."""
    audio_array, sr = librosa.load(
        audio_path,
        sr=qwen_processor.feature_extractor.sampling_rate,
        mono=True,
    )
    # Qwen2-Audio works best with clips under 30 seconds
    max_samples = 30 * qwen_processor.feature_extractor.sampling_rate
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]

    conversation = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {
                    "type": "text",
                    "text": (
                        "Listen to this audio carefully. "
                        "What visual scene does it evoke? "
                        "Respond ONLY with the JSON object as specified."
                    ),
                },
            ],
        },
    ]

    text_input = qwen_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = qwen_processor(
        text=text_input,
        audio=audio_array,
        sampling_rate=qwen_processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
    # Decode only newly generated tokens
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = qwen_processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    # Parse JSON — be robust to slight formatting issues
    try:
        # strip any accidental markdown fences
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
    except json.JSONDecodeError:
        # fallback: extract fields with regex
        def _extract(key, text):
            m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', text)
            return m.group(1) if m else ""
        result = {
            "scene_description": _extract("scene_description", raw) or raw[:120],
            "mood": _extract("mood", raw) or "atmospheric",
            "sdxl_prompt": _extract("sdxl_prompt", raw) or raw[:200],
            "negative_prompt": "blurry, low quality, text, watermark",
        }
    return result

# ──────────────────────────────────────────────
# 2. SDXL-Turbo  — text → image (fast, 4 steps)
# ──────────────────────────────────────────────
from diffusers import AutoPipelineForText2Image

print("[2/3] Loading SDXL-Turbo...")
sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=DTYPE,
    variant="fp16" if DEVICE == "cuda" else None,
)
sdxl_pipe = sdxl_pipe.to(DEVICE)

def generate_image(scene: dict, seed: int = 42) -> Image.Image:
    prompt   = scene.get("sdxl_prompt", "a beautiful landscape")
    neg      = scene.get("negative_prompt", "blurry, low quality")
    mood     = scene.get("mood", "")

    # Enrich prompt with mood
    full_prompt = f"{prompt}, {mood} mood, cinematic lighting, highly detailed, 8k"

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = sdxl_pipe(
        prompt=full_prompt,
        negative_prompt=neg,
        num_inference_steps=4,   # SDXL-Turbo: 1-4 steps
        guidance_scale=0.0,      # Turbo must have guidance_scale=0
        width=512,
        height=512,
        generator=generator,
    ).images[0]
    return result

# ──────────────────────────────────────────────
# 3. Depth Anything V2  — monocular depth
# ──────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

print("[3/3] Loading Depth Anything V2...")
depth_pipe = hf_pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if DEVICE == "cuda" else -1,
)

def get_depth(image: Image.Image) -> np.ndarray:
    """Returns normalized depth map as float32 array (0=near, 1=far)."""
    result = depth_pipe(image)
    depth  = np.array(result["depth"], dtype=np.float32)
    d_min, d_max = depth.min(), depth.max()
    return (depth - d_min) / (d_max - d_min + 1e-8)

print("[INFO] All models loaded ✓")

# ──────────────────────────────────────────────
# 4. Parallax renderer  — depth → 2.5D video
# ──────────────────────────────────────────────

def apply_parallax_shift(
    image_np: np.ndarray,
    depth_np: np.ndarray,
    shift_x: float,
    shift_y: float,
    max_shift: int = 20,
) -> np.ndarray:
    """
    Shift each pixel proportionally to its depth.
    Near objects (depth≈0) shift more; far objects (depth≈1) shift less.
    Uses cv2.remap for fast per-pixel displacement.
    """
    h, w = image_np.shape[:2]
    # Parallax factor: near=1 (max shift), far=0 (no shift)
    parallax = 1.0 - depth_np  # invert: near=1

    # Build displacement maps
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))

    map_x += parallax * shift_x * max_shift
    map_y += parallax * shift_y * max_shift

    shifted = cv2.remap(
        image_np, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return shifted


def render_parallax_video(
    image: Image.Image,
    depth: np.ndarray,
    audio_path: str,
    fps: int = 24,
    duration: float = 6.0,
    max_shift: int = 18,
) -> str:
    """
    Render 2.5D parallax animation and mux with original audio.
    Returns path to output .mp4
    """
    image_np = np.array(image.convert("RGB"))
    depth_rs = cv2.resize(depth, (image_np.shape[1], image_np.shape[0]))
    n_frames = int(fps * duration)

    out_path  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    silent_path = out_path.replace(".mp4", "_silent.mp4")

    h, w = image_np.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (w, h))

    for i in range(n_frames):
        t = i / n_frames  # 0 → 1
        # Smooth figure-8 camera path
        angle = t * 2 * np.pi
        sx = np.sin(angle)           # left ↔ right
        sy = np.sin(angle * 2) * 0.5 # slight vertical bob

        frame = apply_parallax_shift(image_np, depth_rs, sx, sy, max_shift)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()

    # Mux with audio using ffmpeg
    cmd = (
        f"ffmpeg -y -i {silent_path} -i {audio_path} "
        f"-c:v libx264 -preset fast -crf 22 "
        f"-c:a aac -b:a 128k "
        f"-shortest {out_path} -loglevel error"
    )
    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(out_path):
        return silent_path  # fallback: silent video

    os.remove(silent_path)
    return out_path

# ──────────────────────────────────────────────
# 5. Full pipeline
# ──────────────────────────────────────────────

def run_pipeline(
    audio_input,          # (sample_rate, np.ndarray) from gr.Audio
    seed: int,
    max_shift: int,
    video_duration: float,
    progress=gr.Progress(),
):
    if audio_input is None:
        raise gr.Error("오디오를 업로드하거나 마이크로 녹음해주세요.")

    sr, audio_np = audio_input
    # Convert to mono float32 if needed
    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)
    if audio_np.max() > 1.0:
        audio_np /= 32768.0  # int16 → float

    # Save audio to temp file
    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_audio.name, audio_np, sr)
    audio_path = tmp_audio.name

    # ── Step 1: Qwen2-Audio ──────────────────
    progress(0.05, desc="[1/3] Qwen2-Audio: 오디오 분석 중...")
    scene = analyze_audio(audio_path)

    scene_text = (
        f"**🎵 Scene:** {scene.get('scene_description', '')}\n\n"
        f"**🌡️ Mood:** {scene.get('mood', '')}\n\n"
        f"**🖊️ Prompt:** {scene.get('sdxl_prompt', '')}"
    )

    # ── Step 2: SDXL-Turbo ──────────────────
    progress(0.35, desc="[2/3] SDXL-Turbo: 이미지 생성 중...")
    image = generate_image(scene, seed=int(seed))

    # ── Step 3: Depth Anything V2 ───────────
    progress(0.60, desc="=[3/3] Depth Anything V2: 깊이 추정 중...")
    depth = get_depth(image)
    depth_vis = Image.fromarray((depth * 255).astype(np.uint8)).convert("RGB")

    # ── Step 4: Parallax render ─────────────
    progress(0.75, desc="Parallax 영상 렌더링 중...")
    video_path = render_parallax_video(
        image, depth, audio_path,
        max_shift=int(max_shift),
        duration=float(video_duration),
    )

    progress(1.0, desc="완료!")
    return image, depth_vis, video_path, scene_text

# ──────────────────────────────────────────────
# 6. Gradio UI
# ──────────────────────────────────────────────

with gr.Blocks(title="Audio to Video Demo", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 오디오를 영상으로 만들어보자
        ### Foundation Model Pipeline: Qwen2-Audio · SDXL-Turbo · Depth Anything V2

        오디오를 들려주세요 — AI가 오디오에 맞는 영상을 생성합니다.
        """
    )

    with gr.Row():
        # ── Input ──
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="🎤 오디오 입력 (마이크 녹음 or 파일 업로드)",
                sources=["microphone", "upload"],
                type="numpy",
            )

            with gr.Accordion("⚙️ 설정", open=False):
                seed = gr.Slider(0, 9999, value=42, step=1, label="Seed")
                max_shift = gr.Slider(
                    5, 40, value=18, step=1,
                    label="Parallax 강도 (높을수록 3D 효과 강함)",
                )
                video_duration = gr.Slider(
                    3.0, 12.0, value=6.0, step=1.0,
                    label="영상 길이 (초)",
                )

            run_btn = gr.Button("장면 생성", variant="primary", size="lg")

        # ── Output ──
        with gr.Column(scale=2):
            scene_md   = gr.Markdown(label="Qwen2-Audio 분석 결과")
            with gr.Row():
                image_out = gr.Image(label="생성된 장면 (SDXL-Turbo)", height=260)
                depth_out = gr.Image(label="Depth Map (Depth Anything V2)", height=260)
            video_out = gr.Video(label="2.5D Parallax 영상")

    with gr.Accordion("ℹ️ 파이프라인 설명", open=False):
        gr.Markdown(
            """
            | 단계 | 모델 | HuggingFace ID | 역할 |
            |------|------|----------------|------|
            | 1 | **Qwen2-Audio** | `Qwen/Qwen2-Audio-7B-Instruct` | 오디오 → 장면 이해 + SDXL 프롬프트 생성 |
            | 2 | **SDXL-Turbo** | `stabilityai/sdxl-turbo` | 프롬프트 → 이미지 생성 (4 steps) |
            | 3 | **Depth Anything V2** | `depth-anything/Depth-Anything-V2-Small-hf` | 이미지 → 깊이 맵 추출 |
            | 4 | **Parallax Renderer** | OpenCV (custom) | 깊이 맵 → 2.5D 움직이는 영상 |

            **왜 이 조합인가?**
            Qwen2-Audio는 오디오를 *이해*하고, SDXL-Turbo는 그것을 *시각화*하고,
            Depth Anything V2는 정적인 이미지에 *공간감*을 부여합니다.
            소리 → 장면 → 공간 → 생동감으로 이어지는 완전한 순차 파이프라인입니다.
            """
        )

    run_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, seed, max_shift, video_duration],
        outputs=[image_out, depth_out, video_out, scene_md],
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
