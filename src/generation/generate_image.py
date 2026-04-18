import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

_pipe = None

def _load_pipe():
    global _pipe
    if _pipe is None:
        print("Loading image generation model...")
        # SD-Turbo — much faster on CPU, same quality
        _pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        # faster scheduler
        _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            _pipe.scheduler.config
        )
        _pipe.to("cpu")
        _pipe.enable_attention_slicing()   # cuts memory use in half
    return _pipe


def build_prompt(objects: list[str], scene: str, transcript: str) -> str:
    subject = "A person" if "person" in objects else "A scene"

    scene_map = {
        "street":  "a busy urban street with cars",
        "indoor":  "a modern indoor room",
        "beach":   "a sunny beach with waves",
        "sports":  "a sports venue with crowd",
        "nature":  "a green natural landscape",
        "unknown": "an everyday environment",
    }
    env = scene_map.get(scene, "an everyday environment")

    # broader keyword matching
    action_map = {
        "walking":  ["walk", "stroll", "pace", "move"],
        "running":  ["run", "sprint", "jog", "chase"],
        "dancing":  ["danc", "sway", "move to"],
        "playing":  ["play", "game", "match", "sport"],
        "swimming": ["swim", "pool", "water"],
        "talking":  ["talk", "speak", "say", "tell", "discuss", "interview"],
        "cooking":  ["cook", "kitchen", "food", "eat"],
        "working":  ["work", "office", "desk", "type", "computer"],
        "exercising": ["exercise", "workout", "train", "gym", "lift"],
        "haircut":  ["hair", "cut", "style", "blow"],     # ← catches your BlowDryHair
        "surfing":  ["surf", "wave", "board"],             # ← catches your Surfing
        "skating":  ["skate", "ice", "rink", "dance"],    # ← catches IceDancing
        "knitting": ["knit", "yarn", "sew"],
        "mopping":  ["mop", "clean", "floor"],
    }

    action = "standing"
    t = transcript.lower()
    for act, keywords in action_map.items():
        if any(kw in t for kw in keywords):
            action = act
            break

    # also check detected objects as action hint if transcript is empty
    if action == "standing":
        obj_action_map = {
            "sports ball": "playing soccer",
            "surfboard":   "surfing",
            "tennis racket": "playing tennis",
            "scissors":    "getting a haircut",
            "hair drier":  "getting a blowdry",
            "frisbee":     "throwing a frisbee",
        }
        for obj in objects:
            if obj in obj_action_map:
                action = obj_action_map[obj]
                break

    prompt = (
        f"{subject} {action} in {env}, "
        f"photorealistic, natural lighting, high detail"
    )
    return prompt

def generate_image(
    prompt: str,
    output_path: str = "outputs/generated.png",
    steps: int = 4,       # sd-turbo only needs 4 steps
    guidance: float = 0.0,  # sd-turbo uses guidance=0
) -> Image.Image:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipe = _load_pipe()

    image = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    image.save(output_path)
    print(f"Saved → {output_path}")
    return image