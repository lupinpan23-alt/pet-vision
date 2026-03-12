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

        BASE_PROMPT = (
            "你是一个严格的视频事件摘要专家。你的任务是在内部仔细分析视频内容，然后只输出一个高度精简、低 Token 消耗的 JSON 摘要。\n\n"
            "核心原则：\n"
            "- 事实准确：摘要必须准确反映视频中的核心事实。\n"
            "- 事件为核：必须首先识别出场景中的关键动态事件，所有输出围绕核心事件展开。\n\n"
            "第 1 步：内部分析（不输出）\n"
            "- 活动：首先鉴别是否存在异常行为（抽搐、呕吐等），其次是正常行为（玩耍、跑跳、饮水、进食、如厕、撕咬物品等）。\n"
            "- 叫声意图：推测声音意图，如狗叫是警戒、开心还是撒娇，没有声音则返回 null。\n"
            "- 异常行为：如呕吐、抽搐。抽搐定义为连续多帧中动物呈现非常类似的身体姿势（如侧躺在地仿佛在奔跑）。若出现请严格输出，避免漏判。\n"
            "- 人形：若有人物出现，描述外貌特征、衣服颜色等，没有则返回 null。\n"
            "- 是否精彩：出现伸懒腰、哈欠、跳跃、玩玩具、多宠互动等行为，且画面光线充足、不抖动时，标记\"精彩\"并给出具体描述。否则标注 null。异常行为时也标注 null。\n\n"
            "第 2 步：严格按以下 JSON 格式输出，所有字段值尽可能简短。\n\n"
            "{\"活动\": \"[一句话描述，概括宠物对象及其执行的连续动作]\", "
            "\"叫声意图\": \"[4-6字概括叫声意图，如无叫声输出\\\"无\\\"]\", "
            "\"异常行为\": [\"[严格输出异常行为，如\\\"呕吐\\\"\\\"抽搐\\\"，无则为空数组]\"], "
            "\"是否精彩\": \"[精彩则给出10字左右描述，否则为null]\"}\n\n"
            "严禁在 JSON 之外添加任何文本、解释或 ```json 标记。全部输出必须且只能是上述 JSON 结构。"
        )

        if name_labels:
            label_strs = []
            for ts, name in name_labels:
                minutes = ts // 60
                seconds = ts % 60
                label_strs.append(f"{name}（{minutes:02d}:{seconds:02d}附近出现）")
            name_labels_text = "、".join(label_strs)
            name_inject = (
                f"身份识别系统已确认视频中的动物身份：{name_labels_text}。"
                f"在后续所有输出中，你必须严格使用这些专属名字称呼对应动物，禁止用「猫咪」「小狗」等泛称替代已知名字。\n\n"
            )
            prompt_text = name_inject + BASE_PROMPT.replace(
                "[一句话描述，概括宠物对象及其执行的连续动作]",
                "[一句话描述，用已识别的宠物名字称呼动物，概括宠物对象及其执行的连续动作]"
            )
        else:
            prompt_text = BASE_PROMPT

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
