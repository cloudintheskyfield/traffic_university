import argparse
import base64
import concurrent.futures
import fcntl
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from PIL import Image


DEFAULT_RESULTS_DIR = "/mnt1/mnt2/data3/nlp/ws/proj/课题组/results"
DEFAULT_DATASET_FILE = "/mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1/train.jsonl"
DEFAULT_IMAGE_DIR = "/mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1/train_images"

RUBRIC = {
    "core_correctness": {
        "points": 40,
        "zh": "核心结论/操作是否正确",
        "desc": "是否回答了问题的核心要求，和参考答案的主结论、主操作一致。",
    },
    "key_details": {
        "points": 25,
        "zh": "关键细节完整性",
        "desc": "是否覆盖参考答案中的重要条件、步骤、注意事项，是否遗漏会影响判断的要点。",
    },
    "visual_grounding": {
        "points": 20,
        "zh": "图像证据一致性",
        "desc": "是否与图片可见内容一致；图片与参考答案冲突时，以图片中能确认的信息为准。",
    },
    "railway_safety": {
        "points": 10,
        "zh": "铁路安全/专业性",
        "desc": "是否符合铁路场景下谨慎、限速、停车准备、信号/调度等安全原则。",
    },
    "clarity": {
        "points": 5,
        "zh": "表达清晰与无幻觉",
        "desc": "表达是否清楚，是否有明显无关、编造、过度推断或只输出选项字母的问题。",
    },
}


def read_jsonl(path, strict=True):
    rows = []
    bad_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if strict:
                        raise
                    bad_lines.append(
                        {
                            "source": path,
                            "line_number": line_number,
                            "error": str(exc),
                            "line": line,
                        }
                    )
    if bad_lines and not strict:
        bad_path = path + ".bad_lines.jsonl"
        with open(bad_path, "a", encoding="utf-8") as f:
            for row in bad_lines:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[WARN] skipped {len(bad_lines)} malformed lines from {path}; saved to {bad_path}", flush=True)
    return rows


def read_dataset(path):
    items = {}
    for row in read_jsonl(path):
        items[row["id"]] = row
    return items


def result_by_id(path):
    return {row["id"]: row for row in read_jsonl(path)}


def normalize_answer(row, use_extracted_answer):
    full_output = str(row.get("model_full_output") or "").strip()
    extracted = str(row.get("extracted_answer") or "").strip()
    if use_extracted_answer and extracted:
        return extracted
    return full_output or extracted


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {text[:500]}")
    return json.loads(match.group(0))


def call_chat(api_base, model, messages, max_tokens, timeout, retries):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }
    last_error = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 + attempt)
    raise RuntimeError(f"Judge API failed after {retries + 1} attempts: {last_error}")


def image_to_data_url(image_path, max_image_side, jpeg_quality):
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_image_side:
        image.thumbnail((max_image_side, max_image_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_messages(question, reference, baseline_answer, ours_answer, image_data_url=None):
    system = (
        "你是严格、公平、稳定的铁路视觉问答评卷员。"
        "你需要结合图片、问题、参考答案，分别给 baseline 和 ours 两个候选回复独立打分。"
        "如果参考答案和图片明显冲突，优先相信图片里能直接确认的信息，并在理由中说明。"
        "不要因为答案更长就加分；只奖励正确、完整、与图像一致、符合铁路安全原则的内容。"
        "只返回 JSON，不要输出 Markdown。"
        "中文解释中如果需要引用选项或短语，请使用中文引号，不要在 JSON 字符串内部直接使用未转义的英文双引号。"
    )
    user = f"""
请对同一道 QA 题的两个候选回复分别按 1-100 分打分。

问题：
{question}

参考答案：
{reference}

候选回复 baseline：
{baseline_answer}

候选回复 ours：
{ours_answer}

评分机制，总分 100：
1. 核心结论/操作是否正确：40 分。主结论、主操作、主判断与参考答案一致才给高分。
2. 关键细节完整性：25 分。覆盖参考答案中的重要条件、步骤、限制和注意事项。
3. 图像证据一致性：20 分。回复必须与图片可见内容一致，不能编造图片中没有依据的事实。
4. 铁路安全/专业性：10 分。符合铁路场景下限速、停车准备、信号/调度、安全优先等原则。
5. 表达清晰与无幻觉：5 分。表达清楚、不过度推断、不答非所问。

评分边界：
- 90-100：几乎完全正确，关键细节充分，与图片和参考答案一致。
- 75-89：主要正确，只有少量次要遗漏或表述不够完整。
- 60-74：部分正确，抓住主方向，但遗漏多个关键点。
- 40-59：有少量相关内容，但核心操作/判断明显不完整或不可靠。
- 20-39：大部分错误、空泛、与题目关系弱，或存在明显安全/事实问题。
- 1-19：几乎无效；只输出 A/B/C/D、空答案、答非所问、严重矛盾时通常落在这个区间。

请严格输出一个 JSON 对象，字段和类型如下：
{{
  "baseline": {{
    "score": 0,
    "core_correctness": 0,
    "key_details": 0,
    "visual_grounding": 0,
    "railway_safety": 0,
    "clarity": 0,
    "strengths": "中文简述优点",
    "weaknesses": "中文简述扣分点"
  }},
  "ours": {{
    "score": 0,
    "core_correctness": 0,
    "key_details": 0,
    "visual_grounding": 0,
    "railway_safety": 0,
    "clarity": 0,
    "strengths": "中文简述优点",
    "weaknesses": "中文简述扣分点"
  }},
  "winner": "baseline|ours|tie",
  "reason": "中文解释为什么这样打分"
}}

要求：
- score 必须是 1 到 100 的整数。
- 五个分项分数必须分别落在对应满分范围内，且相加等于 score。
- baseline 和 ours 必须分别独立评分，不要只做相对比较。
- 所有字符串值必须是合法 JSON 字符串；不要在字符串中使用未转义的英文双引号。
""".strip()
    user_content = [{"type": "text", "text": user}]
    if image_data_url:
        user_content.insert(0, {"type": "image_url", "image_url": {"url": image_data_url}})
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_json_retry_messages(messages, bad_output, error):
    retry_text = f"""
上一次输出不是合法 JSON，解析错误如下：
{error}

上一次输出：
{bad_output[:2000]}

请重新输出一个合法 JSON 对象。不要解释，不要 Markdown，不要代码块。
必须保持同样的 schema，并且所有字符串内部不要使用未转义的英文双引号。
""".strip()
    return messages + [{"role": "user", "content": [{"type": "text", "text": retry_text}]}]


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_append_jsonl(path, row, lock):
    with lock:
        append_jsonl(path, row)


def load_seen(path):
    if not os.path.exists(path):
        return set(), []
    rows = read_jsonl(path, strict=False)
    finished = {row["id"] for row in rows if "baseline_score" in row and "ours_score" in row}
    return finished, rows


def latest_valid_rows(rows):
    by_id = {}
    for row in rows:
        if "baseline_score" in row and "ours_score" in row:
            by_id[row["id"]] = row
    return [by_id[key] for key in sorted(by_id)]


def summarize(rows):
    total = len(rows)
    baseline_sum = sum(float(row["baseline_score"]) for row in rows)
    ours_sum = sum(float(row["ours_score"]) for row in rows)
    wins = {"baseline": 0, "ours": 0, "tie": 0}
    baseline_dims = {key: 0.0 for key in RUBRIC}
    ours_dims = {key: 0.0 for key in RUBRIC}
    for row in rows:
        winner = row.get("winner", "tie")
        wins[winner if winner in wins else "tie"] += 1
        for key in RUBRIC:
            baseline_dims[key] += float(row.get("baseline_dimensions", {}).get(key, 0))
            ours_dims[key] += float(row.get("ours_dimensions", {}).get(key, 0))
    return {
        "total": total,
        "baseline_avg": baseline_sum / total if total else None,
        "ours_avg": ours_sum / total if total else None,
        "baseline_sum": baseline_sum,
        "ours_sum": ours_sum,
        "wins": wins,
        "baseline_dim_avg": {key: value / total if total else None for key, value in baseline_dims.items()},
        "ours_dim_avg": {key: value / total if total else None for key, value in ours_dims.items()},
    }


def write_summary(path, summary, output_jsonl, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# QA 大模型裁判评估汇总（100 分制）\n\n")
        f.write(f"- 评估样本数：{summary['total']}\n")
        f.write(f"- Baseline 平均分：{summary['baseline_avg']:.2f}/100\n")
        f.write(f"- Ours 平均分：{summary['ours_avg']:.2f}/100\n")
        f.write(f"- Baseline 总分：{summary['baseline_sum']:.1f}\n")
        f.write(f"- Ours 总分：{summary['ours_sum']:.1f}\n")
        f.write(f"- 胜负统计：`{json.dumps(summary['wins'], ensure_ascii=False)}`\n")
        f.write(f"- 并发数：{args.workers}\n")
        f.write(f"- 输入模式：{'纯文本' if args.text_only else '图片+问题+参考答案+两个候选回复'}\n")
        f.write(f"- 明细文件：`{output_jsonl}`\n")
        f.write("\n## 分项平均分\n\n")
        f.write("| 维度 | 满分 | Baseline | Ours |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for key, spec in RUBRIC.items():
            f.write(
                f"| {spec['zh']} | {spec['points']} | "
                f"{summary['baseline_dim_avg'][key]:.2f} | {summary['ours_dim_avg'][key]:.2f} |\n"
            )
        f.write("\n## 评分机制\n\n")
        for index, (key, spec) in enumerate(RUBRIC.items(), start=1):
            f.write(f"{index}. {spec['zh']}（{spec['points']} 分）：{spec['desc']}\n")
        f.write("\n## 分数解释\n\n")
        f.write("- 90-100：几乎完全正确，关键细节充分，与图片和参考答案一致。\n")
        f.write("- 75-89：主要正确，只有少量次要遗漏或表述不够完整。\n")
        f.write("- 60-74：部分正确，抓住主方向，但遗漏多个关键点。\n")
        f.write("- 40-59：有少量相关内容，但核心操作/判断明显不完整或不可靠。\n")
        f.write("- 20-39：大部分错误、空泛、与题目关系弱，或存在明显安全/事实问题。\n")
        f.write("- 1-19：几乎无效；只输出 A/B/C/D、空答案、答非所问、严重矛盾时通常落在这个区间。\n")


def clamp_int(value, low, high):
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        value = low
    return max(low, min(high, value))


def normalize_candidate_judgement(judged, key):
    data = judged.get(key, {})
    dimensions = {}
    for dim_key, spec in RUBRIC.items():
        dimensions[dim_key] = clamp_int(data.get(dim_key, 0), 0, spec["points"])
    score_from_dims = sum(dimensions.values())
    if score_from_dims < 1:
        dimensions["clarity"] = 1
        score_from_dims = 1
    score = clamp_int(data.get("score", score_from_dims), 1, 100)
    if abs(score - score_from_dims) > 3:
        score = clamp_int(score_from_dims, 1, 100)
    return {
        "score": score,
        "dimensions": dimensions,
        "strengths": str(data.get("strengths", "")).strip(),
        "weaknesses": str(data.get("weaknesses", "")).strip(),
    }


def judge_one(item_id, dataset, baseline, ours, args):
    item = dataset[item_id]
    use_extracted_answer = args.use_extracted_answer and not args.use_full_output
    baseline_answer = normalize_answer(baseline[item_id], use_extracted_answer)
    ours_answer = normalize_answer(ours[item_id], use_extracted_answer)
    image_file = os.path.join(args.image_dir, item["image_id"])
    image_data_url = None
    if not args.text_only:
        image_data_url = image_to_data_url(image_file, args.max_image_side, args.jpeg_quality)
    messages = build_messages(
        item["question"],
        item["answer"],
        baseline_answer,
        ours_answer,
        image_data_url,
    )
    raw = None
    judged = None
    current_messages = messages
    parse_error = None
    for parse_attempt in range(args.parse_retries + 1):
        raw = call_chat(args.api_base, args.model, current_messages, args.max_tokens, args.timeout, args.retries)
        try:
            judged = extract_json(raw)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            parse_error = exc
            if parse_attempt >= args.parse_retries:
                raise
            current_messages = build_json_retry_messages(messages, raw, exc)
    if judged is None:
        raise RuntimeError(f"Judge output parse failed: {parse_error}")
    baseline_judged = normalize_candidate_judgement(judged, "baseline")
    ours_judged = normalize_candidate_judgement(judged, "ours")
    if baseline_judged["score"] > ours_judged["score"]:
        winner = "baseline"
    elif ours_judged["score"] > baseline_judged["score"]:
        winner = "ours"
    else:
        winner = "tie"
    return {
        "id": item_id,
        "image_file": image_file,
        "question": item["question"],
        "reference_answer": item["answer"],
        "baseline_answer": baseline_answer,
        "ours_answer": ours_answer,
        "baseline_score": baseline_judged["score"],
        "ours_score": ours_judged["score"],
        "baseline_dimensions": baseline_judged["dimensions"],
        "ours_dimensions": ours_judged["dimensions"],
        "baseline_strengths": baseline_judged["strengths"],
        "baseline_weaknesses": baseline_judged["weaknesses"],
        "ours_strengths": ours_judged["strengths"],
        "ours_weaknesses": ours_judged["weaknesses"],
        "winner": winner,
        "judge_winner": judged.get("winner", "tie"),
        "reason": judged.get("reason", ""),
        "judge_input_mode": "text_only" if args.text_only else "image_question_answer",
        "judge_model": args.model,
        "score_scale": "1-100",
        "judge_raw_output": raw,
    }


def main(args):
    dataset = read_dataset(args.dataset_file)
    baseline = result_by_id(args.baseline_results)
    ours = result_by_id(args.ours_results)
    seen, existing = load_seen(args.output_jsonl)

    qa_ids = [
        item_id for item_id, item in dataset.items()
        if item.get("task_type") == "qa" and item_id in baseline and item_id in ours
    ]
    qa_ids.sort()
    if args.limit:
        qa_ids = qa_ids[:args.limit]

    pending_ids = [item_id for item_id in qa_ids if item_id not in seen]
    print(
        f"QA items to evaluate: {len(qa_ids)}, already finished: {len(seen)}, "
        f"remaining: {len(pending_ids)}, workers: {args.workers}"
    )

    rows = list(existing)
    lock = threading.Lock()
    start_time = time.time()
    last_print = start_time
    completed_now = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_one, item_id, dataset, baseline, ours, args): item_id
            for item_id in pending_ids
        }
        for future in concurrent.futures.as_completed(futures):
            item_id = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "id": item_id,
                    "error": str(exc),
                    "score_scale": "1-100",
                    "judge_model": args.model,
                    "judge_input_mode": "text_only" if args.text_only else "image_question_answer",
                }
                print(f"[ERROR] id={item_id}: {exc}")
            atomic_append_jsonl(args.output_jsonl, row, lock)
            rows.append(row)
            completed_now += 1
            now = time.time()
            if completed_now % args.print_every == 0 or now - last_print >= args.print_seconds:
                valid_rows = latest_valid_rows(rows)
                summary = summarize(valid_rows)
                elapsed = max(now - start_time, 1e-6)
                speed = completed_now / elapsed
                remaining = len(pending_ids) - completed_now
                eta = remaining / speed if speed > 0 else 0
                print(
                    f"[{len(rows)}/{len(qa_ids)}] "
                    f"new={completed_now}/{len(pending_ids)}, "
                    f"speed={speed:.2f} sample/s, eta={eta/60:.1f} min, "
                    f"baseline_avg={summary['baseline_avg']:.2f}, "
                    f"ours_avg={summary['ours_avg']:.2f}, wins={summary['wins']}",
                    flush=True,
                )
                last_print = now

    valid_rows = latest_valid_rows(rows)
    summary = summarize(valid_rows)
    write_summary(args.output_md, summary, args.output_jsonl, args)
    print(f"Done. Summary saved to {args.output_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--baseline-results", default=os.path.join(DEFAULT_RESULTS_DIR, "baseline_vllm_resume.jsonl"))
    parser.add_argument("--ours-results", default=os.path.join(DEFAULT_RESULTS_DIR, "ours_vllm_resume.jsonl"))
    parser.add_argument("--output-jsonl", default=os.path.join(DEFAULT_RESULTS_DIR, "qa_judge_qwen35_100_details.jsonl"))
    parser.add_argument("--output-md", default=os.path.join(DEFAULT_RESULTS_DIR, "qa_judge_qwen35_100_summary.md"))
    parser.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen35-35b-a3b-judge")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--print-seconds", type=int, default=30)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--use-full-output", action="store_true", help="Deprecated: full model output is already the default for QA judging.")
    parser.add_argument("--use-extracted-answer", action="store_true", help="Use extracted_answer instead of model_full_output.")
    parser.add_argument("--text-only", action="store_true", help="Do not send images to the judge model.")
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    main(parser.parse_args())
