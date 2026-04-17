import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

# EfficientNet - lightweight, accurate, downloads automatically
_model = None

def _load_model():
    global _model
    if _model is None:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        net     = models.efficientnet_b0(weights=weights)
        net.eval()
        _model = (net, weights.transforms())
    return _model


# ImageNet has 1000 classes - map the ones we care about to readable scene labels
SCENE_KEYWORDS = {
    "beach":    ["seashore", "beach", "sandbar", "coast"],
    "street":   ["street", "road", "sidewalk", "traffic", "intersection", "crosswalk"],
    "indoor":   ["living room", "bedroom", "office", "gym", "restaurant", "kitchen", "corridor"],
    "sports":   ["basketball", "tennis", "soccer", "swimming", "skiing", "skating", "golf"],
    "nature":   ["forest", "mountain", "lake", "valley", "field", "cliff", "river"],
    "vehicle":  ["car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane"],
}

def _map_to_scene(label: str) -> str:
    label_lower = label.lower()
    for scene, keywords in SCENE_KEYWORDS.items():
        if any(kw in label_lower for kw in keywords):
            return scene
    return "unknown"


def classify_frame(frame_bgr: np.ndarray) -> dict:
    """Classify a single BGR frame. Returns scene label + top ImageNet class."""
    net, transform = _load_model()

    img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = net(tensor)
        probs  = torch.softmax(logits, dim=1)

    top_prob, top_idx = probs.topk(3)
    labels = net.__class__  # placeholder — use weights meta

    weights  = models.EfficientNet_B0_Weights.DEFAULT
    top_labs = [weights.meta["categories"][i] for i in top_idx[0].tolist()]
    top_prbs = top_prob[0].tolist()

    best_label = top_labs[0]
    scene      = _map_to_scene(best_label)

    return {
        "scene":       scene,
        "top_class":   best_label,
        "confidence":  round(top_prbs[0], 3),
        "top3":        list(zip(top_labs, [round(p, 3) for p in top_prbs])),
    }


def classify_video(video_path: str, every_n: int = 30) -> list[dict]:
    """Run scene classification on sampled frames of a video."""
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0
    results   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n == 0:
            info = classify_frame(frame)
            results.append({
                "frame":  frame_idx,
                "second": round(frame_idx / fps, 2),
                **info,
            })

        frame_idx += 1

    cap.release()
    return results