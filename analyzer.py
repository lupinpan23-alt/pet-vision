import os
import json
import base64
import subprocess
import glob
import requests
from config import VITA_API_URL, VITA_API_KEY, VITA_MODEL


def call_vita_api(content_list):
    headers = {
        "Authorization": f"Bearer {VITA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VITA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content_list,
            }
        ],
    }
    resp = requests.post(VITA_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    return raw


def img_to_base64_url(img_path):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def detect_animal(frame_path):
    """检测帧中是否有动物，返回 True/False"""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": img_to_base64_url(frame_path)},
        },
        {
            "type": "text",
            "text": '这张图片中是否有动物？请只返回JSON格式：{"has_animal": true/false}',
        },
    ]
    try:
        raw = call_vita_api(content)
        # 尝试提取 JSON
        raw = raw.strip()
        # 可能包裹在 markdown 代码块中
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw = raw[start:end]
        result = json.loads(raw)
        return bool(result.get("has_animal", False))
    except Exception:
        return False


def compare_pet(ref_path, frame_path):
    """比对参考图和帧图，返回 (is_same, similarity)"""
    content = [
        {
            "type": "image_url",
            "image_url": {"url": img_to_base64_url(ref_path)},
        },
        {
            "type": "image_url",
            "image_url": {"url": img_to_base64_url(frame_path)},
        },
        {
            "type": "text",
            "text": (
                "第一张是参考图片，第二张是视频帧截图，请判断两张图片中是否为同一只动物。"
                '请只返回JSON格式：{"is_same": true/false, "similarity": 85, "reason": "简短说明"}'
            ),
        },
    ]
    try:
        raw = call_vita_api(content)
        raw = raw.strip()
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            raw = raw[start:end]
        result = json.loads(raw)
        return bool(result.get("is_same", False)), int(result.get("similarity", 0))
    except Exception:
        return False, 0


def analyze_video(task_id, video_url, pets, task_store):
    frames_dir = os.path.join("frames", task_id)
    os.makedirs(frames_dir, exist_ok=True)

    def update(status, progress, message, result=None):
        task_store[task_id] = {
            "status": status,
            "progress": progress,
            "message": message,
            "result": result,
        }

    try:
        # Step 1: ffmpeg 抽帧
        update("processing", 5, "正在从视频URL抽取帧...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_url,
            "-vf", "fps=1",
            os.path.join(frames_dir, "frame_%04d.jpg"),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if proc.returncode != 0:
            stderr_msg = proc.stderr.decode(errors="ignore")[-500:]
            update("failed", 0, f"ffmpeg 抽帧失败：{stderr_msg}")
            return

        # 获取所有帧文件，按序号排列
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        total_frames = len(frame_files)

        if total_frames == 0:
            update("failed", 0, "未能抽取到任何视频帧，请检查视频URL是否可访问")
            return

        update("processing", 15, f"抽帧完成，共 {total_frames} 帧，开始逐帧分析...")

        # Step 2: 逐帧检测动物
        prev_has_animal = False
        name_labels = []  # [(timestamp_sec, name)]

        for idx, frame_path in enumerate(frame_files):
            frame_num = idx + 1
            timestamp_sec = idx  # 每秒1帧，帧序号即秒数

            progress = 15 + int((idx / total_frames) * 60)
            update("processing", progress, f"正在分析第 {frame_num}/{total_frames} 帧...")

            has_animal = detect_animal(frame_path)

            # 仅在首次出现（上一帧无、本帧有）时做身份比对
            if has_animal and not prev_has_animal:
                print(f"[检测] 第{frame_num}帧首次出现动物，开始身份比对")
                if pets:
                    best_name = None
                    best_similarity = 0
                    for pet in pets:
                        pet_img_path = pet["image_path"]
                        if not os.path.exists(pet_img_path):
                            print(f"[比对] 宠物图片不存在: {pet_img_path}")
                            continue
                        is_same, similarity = compare_pet(pet_img_path, frame_path)
                        print(f"[比对] 宠物={pet['name']}, is_same={is_same}, similarity={similarity}")
                        if similarity > best_similarity:
                            best_similarity = similarity
                            if is_same and similarity > 80:
                                best_name = pet["name"]

                    if best_name:
                        name_labels.append((timestamp_sec, best_name))
                        print(f"[识别] 匹配成功：{best_name}，相似度={best_similarity}")
                    else:
                        print(f"[识别] 未匹配任何宠物，最高相似度={best_similarity}")
            elif has_animal:
                print(f"[检测] 第{frame_num}帧动物持续出现，跳过比对")
            else:
                print(f"[检测] 第{frame_num}帧无动物")

            prev_has_animal = has_animal

        # Step 3: 汇总名字标签，调用 VITA 完整视频分析
        update("processing", 80, "正在进行最终视频分析...")

        if name_labels:
            label_strs = []
            for ts, name in name_labels:
                minutes = ts // 60
                seconds = ts % 60
                label_strs.append(f"{name}（{minutes:02d}:{seconds:02d}附近出现）")
            name_labels_text = "、".join(label_strs)
            prompt_text = (
                f"请详细分析这段视频的内容。"
                f"重要提示：视频中的动物已经过身份识别系统确认，识别结果如下：{name_labels_text}。"
                f"你必须严格按照识别结果，用动物的专属名字来称呼它们，禁止使用「猫咪」「小狗」等泛称代替已知名字。"
                f"例如识别结果中有「奥利奥」，则全程用「奥利奥」而不是「猫咪」或「这只猫」。"
            )
        else:
            prompt_text = "请详细分析这段视频的内容。"

        final_content = [
            {
                "type": "video_url",
                "video_url": {"url": video_url},
            },
            {
                "type": "text",
                "text": prompt_text,
            },
        ]

        final_result = call_vita_api(final_content)

        update(
            "completed",
            100,
            "分析完成",
            {
                "analysis": final_result,
                "name_labels": [
                    {"timestamp": ts, "name": name} for ts, name in name_labels
                ],
                "total_frames": total_frames,
            },
        )

    except Exception as e:
        update("failed", 0, f"分析过程出错：{str(e)}")
