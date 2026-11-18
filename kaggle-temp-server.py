# Kaggle 临时测试服务器：Gradio 三模型测试页 + Cloudflare 隧道公开链接
# 用法：Kaggle 新建 Notebook，Accelerator = GPU T4 x2，Internet = On。
# Cell 1 是下面注释里的 shell 命令；Cell 2 是本文件正文（去掉开头的注释块后直接粘贴）。
#
# ---- Cell 1（shell 命令，逐行执行）----
# !pip install -q gradio onnxruntime-gpu
# !curl -sSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared && /tmp/cloudflared --version
# !mkdir -p /kaggle/working/models
# !curl -sSL -o /kaggle/working/models/debris-best-clean.onnx https://raw.githubusercontent.com/Sky-Sheepfold/reservoir-model-test/main/models/debris-best-clean.onnx
# !curl -sSL -o /kaggle/working/models/yolo11m.onnx https://raw.githubusercontent.com/Sky-Sheepfold/reservoir-model-test/main/models/yolo11m.onnx
# !ls -lh /kaggle/working/models/
#
# ---- Cell 2（以下全部是 Python，直接粘贴到一个 cell 里运行）----

import os
import re
import subprocess
import threading
import time
import urllib.request

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

MODEL_DIR = "/kaggle/working/models"
MODELS = {
    "debris": {"file": f"{MODEL_DIR}/debris-best-clean.onnx", "name": "漂浮物",
               "color": (34, 197, 94), "conf": 0.25, "cls": None},
    "person": {"file": f"{MODEL_DIR}/yolo11m.onnx", "name": "人员",
               "color": (59, 130, 246), "conf": 0.15, "cls": 0},
    "boat": {"file": f"{MODEL_DIR}/yolo11m.onnx", "name": "船只",
             "color": (249, 115, 22), "conf": 0.15, "cls": 8},
}
SESSIONS = {}

print("ONNX providers available:", ort.get_available_providers(), flush=True)

# 示例图：模型以前没见过的新照片，从 GitHub 仓库下载到本地（已去 EXIF，最宽 1280）
EXAMPLE_DIR = "/kaggle/working/examples"
EXAMPLE_FILES = [
    ("debris-4-river-litter.jpg", "debris", 0.25),
    ("debris-5-danube-bottle.jpg", "debris", 0.25),
    ("debris-6-plastic-persistence.jpg", "debris", 0.25),
    ("person-3-lake-walk.jpg", "person", 0.15),
    ("person-4-lake-edward.jpg", "person", 0.15),
    ("person-5-yellowstone.jpg", "person", 0.15),
    ("boat-3-oroville.jpg", "boat", 0.15),
    ("boat-4-central-park.jpg", "boat", 0.15),
    ("boat-5-phayao.jpg", "boat", 0.15),
]
os.makedirs(EXAMPLE_DIR, exist_ok=True)
for _fname, _, _ in EXAMPLE_FILES:
    _path = os.path.join(EXAMPLE_DIR, _fname)
    if not os.path.exists(_path):
        _url = f"https://raw.githubusercontent.com/Sky-Sheepfold/reservoir-model-test/main/server-examples/{_fname}"
        urllib.request.urlretrieve(_url, _path)
        print("downloaded example", _fname, os.path.getsize(_path), "bytes", flush=True)


def get_session(path):
    if path in SESSIONS:
        return SESSIONS[path]
    try:
        sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    SESSIONS[path] = sess
    return sess


def letterbox(img, size=640):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr, scale, pad_x, pad_y


def parse_output(data, dims, model, conf, scale, pad_x, pad_y, img_w, img_h):
    _, c, n = dims
    data = np.asarray(data).reshape(-1)
    boxes = []
    for i in range(n):
        if model["cls"] is None:
            score = float(data[4 * n + i])
            cls = 0
        else:
            score = 0.0
            cls = -1
            for cc in range(80):
                s = float(data[(4 + cc) * n + i])
                if s > score:
                    score = s
                    cls = cc
            if cls != model["cls"]:
                continue
        if score < conf:
            continue
        cx = float(data[0 * n + i])
        cy = float(data[1 * n + i])
        w = float(data[2 * n + i])
        h = float(data[3 * n + i])
        x1 = max(0, (cx - w / 2 - pad_x) / scale)
        y1 = max(0, (cy - h / 2 - pad_y) / scale)
        x2 = min(img_w, (cx + w / 2 - pad_x) / scale)
        y2 = min(img_h, (cy + h / 2 - pad_y) / scale)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": score})
    boxes.sort(key=lambda b: -b["score"])
    kept = []
    for b in boxes:
        ok = True
        for k in kept:
            ix1, iy1 = max(b["x1"], k["x1"]), max(b["y1"], k["y1"])
            ix2, iy2 = min(b["x2"], k["x2"]), min(b["y2"], k["y2"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) + (k["x2"] - k["x1"]) * (k["y2"] - k["y1"]) - inter
            if union > 0 and inter / union > 0.45:
                ok = False
                break
        if ok:
            kept.append(b)
    return kept


def draw_boxes(img, boxes, model):
    out = img.copy()
    d = ImageDraw.Draw(out)
    lw = max(2, img.size[0] // 400)
    for b in boxes:
        d.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=model["color"], width=lw)
        label = f"{model['name']} {b['score']:.2f}"
        tx, ty = b["x1"], max(0, b["y1"] - 22)
        d.rectangle([tx, ty, tx + len(label) * 11 + 8, ty + 20], fill=model["color"])
        d.text((tx + 4, ty + 3), label, fill="white")
    return out


def detect(image, model_key, conf):
    import time as _time
    if image is None:
        return None, "请先上传一张照片。"
    model = MODELS[model_key]
    if image.mode != "RGB":
        image = image.convert("RGB")
    sess = get_session(model["file"])
    inp, scale, pad_x, pad_y = letterbox(image)
    t0 = _time.time()
    out = sess.run(None, {"images": inp})[0]
    dt = _time.time() - t0
    boxes = parse_output(out, out.shape, model, conf, scale, pad_x, pad_y, image.size[0], image.size[1])
    annotated = draw_boxes(image, boxes, model)
    if boxes:
        text = f"找到 {len(boxes)} 个{model['name']}，用时约 {dt:.1f} 秒（门槛 {conf}）。"
    else:
        text = f"这张图里没找到{model['name']}，用时约 {dt:.1f} 秒。可以试试把门槛调低一点，或者换张图。"
    return annotated, text


def on_model_change(model_key):
    return gr.update(value=MODELS[model_key]["conf"])


with gr.Blocks(title="水库监控 · 三模型测试") as demo:
    gr.Markdown("# 水库监控 · 三模型测试\n传一张照片，就能看到模型框出什么。推理在 Kaggle 的显卡上跑，照片不会保存。")
    with gr.Row():
        model_radio = gr.Radio(
            choices=[("漂浮物", "debris"), ("人员", "person"), ("船只", "boat")],
            value="debris",
            label="选模型",
        )
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="上传照片（比如水面、坝区、岸边拍的照片）")
            conf_slider = gr.Slider(0.05, 0.5, value=0.25, step=0.01, label="灵敏度（门槛）：越低找到越多、也越容易认错")
            run_btn = gr.Button("开始检测", variant="primary")
        with gr.Column():
            img_out = gr.Image(type="pil", label="检测结果")
            result_text = gr.Textbox(label="结果", lines=3)
    gr.Markdown(
        "三个模型：漂浮物是训练好的（100 个能找到 71 个）；人员 100 个人能找到 70 个；"
        "船只 100 条船能找到 54 条。第一次用某个模型要先加载模型文件，请耐心等一下；"
        "人员和船只用的是同一个模型，来回切换不用重新加载。这是临时测试，链接只在测试期间有效。"
    )
    gr.Examples(
        examples=[[os.path.join(EXAMPLE_DIR, f), m, c] for f, m, c in EXAMPLE_FILES],
        inputs=[img_in, model_radio, conf_slider],
        label="点一张试试（不用自己传照片）",
    )
    gr.Markdown(
        "上面 9 张是新找的测试图，模型以前没见过：漂浮物 3 张（河面垃圾、多瑙河上的瓶子、水面漂浮的塑料）、"
        "人员 3 张（湖边散步的人、黄石湖边的游客、爱德华湖的渔民）、船只 3 张（奥罗维尔湖的船、中央公园湖面的船、帕夭湖畔的小船）。"
        "图来自 Wikimedia Commons（CC0/CC BY/CC BY-SA）和 Pexels（免费图库），都是能免费用的公开照片。"
        "点一张会自动选好对应的模型，再点“开始检测”就行。"
    )
    model_radio.change(on_model_change, inputs=model_radio, outputs=conf_slider)
    run_btn.click(detect, inputs=[img_in, model_radio, conf_slider], outputs=[img_out, result_text])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True, prevent_thread_lock=True)
print("Gradio started on :7860 (share link above, if any)", flush=True)

# 预热：三个模型各跑一次，确认能加载、确认用的是 GPU 还是 CPU
rng = np.random.default_rng(0)
warm = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
for k in ("debris", "person", "boat"):
    _, txt = detect(warm, k, MODELS[k]["conf"])
    print("warm-up", k, "->", txt, flush=True)
print("providers in use:", {p: get_session(MODELS[p]["file"]).get_providers() for p in ("debris", "person")}, flush=True)

# Cloudflare 隧道：把本地 7860 暴露成公开链接；断了自动重连并打印新链接
URL_RE = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")


def tunnel_loop():
    while True:
        proc = subprocess.Popen(
            ["/tmp/cloudflared", "tunnel", "--url", "http://localhost:7860"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            m = URL_RE.search(line)
            if m:
                print("PUBLIC_URL:", m.group(0), flush=True)
        print("cloudflared exited (code", proc.wait(), ") - restarting in 5s", flush=True)
        time.sleep(5)


threading.Thread(target=tunnel_loop, daemon=True).start()

# 保活：这个 cell 一直跑，Kaggle 会话就不掉线；每 5 分钟打一个心跳
n = 0
while True:
    time.sleep(300)
    n += 1
    print(f"heartbeat {n}: server still running", flush=True)
