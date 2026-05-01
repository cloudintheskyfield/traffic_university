import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime


def load_records(path):
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported result format: {path}")


def count_dataset(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def normalize_answer(value):
    text = str(value).strip()
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return re.sub(r"\s+", " ", text).lower()


def summarize(path, dataset_file=None, label=None, score_open=False):
    rows = load_records(path)
    total_dataset = count_dataset(dataset_file)
    by_task = Counter(row.get("task_type", "unknown") for row in rows)
    scored_by_task = defaultdict(lambda: {"correct": 0, "total": 0})
    scored_total = 0
    correct_total = 0

    for row in rows:
        task = row.get("task_type", "unknown")
        pred = row.get("extracted_answer")
        gold = row.get("ground_truth")
        is_correct = row.get("is_correct")
        if task != "mc" and not score_open:
            continue
        if is_correct is None and pred is not None and gold is not None:
            is_correct = normalize_answer(pred) == normalize_answer(gold)
        if is_correct is None:
            continue
        scored_by_task[task]["total"] += 1
        scored_total += 1
        if bool(is_correct):
            scored_by_task[task]["correct"] += 1
            correct_total += 1

    stat = os.stat(path)
    summary = {
        "label": label or os.path.basename(path),
        "path": path,
        "file_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "completed": len(rows),
        "dataset_total": total_dataset,
        "completion_rate": (len(rows) / total_dataset) if total_dataset else None,
        "task_counts": dict(by_task),
        "scored_total": scored_total,
        "correct_total": correct_total,
        "accuracy": (correct_total / scored_total) if scored_total else None,
        "accuracy_by_task": {
            task: {
                "correct": item["correct"],
                "total": item["total"],
                "accuracy": (item["correct"] / item["total"]) if item["total"] else None,
            }
            for task, item in sorted(scored_by_task.items())
        },
    }
    return summary


def pct(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def render_markdown(summaries):
    lines = ["# RailVQA 结果汇总", ""]
    lines.append("| 运行名称 | 已完成 | 完成率 | 参与计分 | 正确数 | 准确率 | 文件更新时间 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for item in summaries:
        total = item["dataset_total"]
        completed = f"{item['completed']}/{total}" if total else str(item["completed"])
        lines.append(
            f"| {item['label']} | {completed} | {pct(item['completion_rate'])} | "
            f"{item['scored_total']} | {item['correct_total']} | {pct(item['accuracy'])} | {item['file_modified']} |"
        )
    lines.append("")
    for item in summaries:
        lines.append(f"## {item['label']}")
        lines.append(f"- 结果文件：`{item['path']}`")
        lines.append(f"- 任务数量：`{json.dumps(item['task_counts'], ensure_ascii=False)}`")
        lines.append("- 分任务准确率：")
        if item["accuracy_by_task"]:
            for task, task_stat in item["accuracy_by_task"].items():
                lines.append(
                    f"  - `{task}`: {task_stat['correct']}/{task_stat['total']} = {pct(task_stat['accuracy'])}"
                )
        else:
            lines.append("  - 暂无可计分样本。")
        lines.append("")
    if len(summaries) >= 2:
        baseline = summaries[0]
        for other in summaries[1:]:
            diff = None
            if baseline["accuracy"] is not None and other["accuracy"] is not None:
                diff = other["accuracy"] - baseline["accuracy"]
            lines.append(
                f"`{other['label']}` 相比 `{baseline['label']}`："
                f"准确率变化 = {pct(diff) if diff is not None else 'N/A'}"
            )
    lines.append("")
    lines.append("> 默认只把 `mc` 选择题纳入准确率；开放问答 `qa` 只统计数量。若要用归一化完全匹配粗略计分开放问答，可运行时加 `--score-open`。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Result JSON/JSONL files. Put baseline first, ours second for comparison.")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--score-open", action="store_true", help="Also score open-ended QA with normalized exact match.")
    args = parser.parse_args()

    labels = args.labels or []
    summaries = [
        summarize(path, dataset_file=args.dataset_file, label=labels[i] if i < len(labels) else None, score_open=args.score_open)
        for i, path in enumerate(args.results)
    ]
    markdown = render_markdown(summaries)
    print(markdown)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(markdown)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
