import os
import json
import uuid
import threading
from flask import Flask, request, jsonify, render_template
from analyzer import analyze_video

app = Flask(__name__)

PETS_JSON = "pets.json"
PETS_DIR = "pets"
FRAMES_DIR = "frames"

os.makedirs(PETS_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

task_store = {}


def load_pets():
    if not os.path.exists(PETS_JSON):
        return []
    with open(PETS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pets(pets):
    with open(PETS_JSON, "w", encoding="utf-8") as f:
        json.dump(pets, f, ensure_ascii=False, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/pets", methods=["GET"])
def get_pets():
    pets = load_pets()
    return jsonify(pets)


@app.route("/api/pets", methods=["POST"])
def add_pet():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"error": "宠物名字不能为空"}), 400

    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "请上传宠物图片"}), 400

    pet_id = str(uuid.uuid4())
    ext = os.path.splitext(image_file.filename)[1] or ".jpg"
    image_filename = f"{pet_id}{ext}"
    image_path = os.path.join(PETS_DIR, image_filename)
    image_file.save(image_path)

    pets = load_pets()
    pet = {
        "id": pet_id,
        "name": name,
        "image_path": image_path,
        "image_filename": image_filename,
    }
    pets.append(pet)
    save_pets(pets)

    return jsonify(pet), 201


@app.route("/api/analyze", methods=["POST"])
def start_analyze():
    data = request.get_json()
    if not data or not data.get("video_url"):
        return jsonify({"error": "请提供视频URL"}), 400

    video_url = data["video_url"].strip()
    task_id = str(uuid.uuid4())

    task_store[task_id] = {
        "status": "processing",
        "progress": 0,
        "message": "任务已创建，准备开始...",
        "result": None,
    }

    pets = load_pets()

    t = threading.Thread(
        target=analyze_video,
        args=(task_id, video_url, pets, task_store),
        daemon=True,
    )
    t.start()

    return jsonify({"task_id": task_id}), 202


@app.route("/api/task/<task_id>", methods=["GET"])
def get_task(task_id):
    task = task_store.get(task_id)
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(task)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
