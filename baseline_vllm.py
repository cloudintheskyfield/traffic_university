import argparse
import json
import os
import re
import time

from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

DEFAULT_DATASET_ROOT = "/mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1"
DEFAULT_MODEL_PATH = "/mnt1/mnt2/data3/nlp/ws/model/Qwen3-VL-8B-Instruct"
DEFAULT_DATASET_FILE = os.path.join(DEFAULT_DATASET_ROOT, "train.jsonl")
DEFAULT_IMAGE_DIR = os.path.join(DEFAULT_DATASET_ROOT, "train_images")
DEFAULT_OUT_DIR = "/mnt1/mnt2/data3/nlp/ws/proj/课题组/results"

SYSTEM_PROMPT = (
    "You are an expert train driver and railway safety inspector. "
    "Carefully analyze the provided railway scene image and answer the question following a strict 4-step cognitive process.\n\n"
    "Format your response EXACTLY as follows:\n"
    "Perception: Start with 'Visual analysis: ' and describe the environment, weather, tracks, trains, and signals.\n"
    "Reasoning: Start with 'Logical analysis: ' and analyze the situation based on safety protocols and your perception.\n"
    "Planning: Start with 'Action plan: ' and detail the necessary actions to ensure safety.\n"
    "Answer: Provide the final direct answer or correct option letter."
)

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def load_records(dataset_file):
    with open(dataset_file, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_question(data):
    raw_question = data["question"].strip()
    task_type = data.get("task_type", "qa")
    if task_type == "mc" and "options" in data:
        options = "\n\nOptions:"
        for key in sorted(data["options"].keys()):
            options += f"\n{key}: {data['options'][key]}"
        options += "\n\nFor the 'Answer' section, explicitly provide ONLY the correct option letter (e.g., A, B, C, or D)."
        return raw_question + options
    return raw_question


def build_prompt(question):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{IMAGE_PLACEHOLDER}{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_answer(text, task_type):
    match = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        ans = match.group(1).strip()
        if task_type == "mc":
            mc_match = re.search(r"([A-D])", ans, re.IGNORECASE)
            return mc_match.group(1).upper() if mc_match else ans
        return ans
    return text


def make_requests(records, image_dir):
    requests = []
    meta = []
    for data in records:
        image_file = os.path.join(image_dir, data["image_id"])
        question = build_question(data)
        image = Image.open(image_file).convert("RGB")
        task_type = data.get("task_type", "qa")
        requests.append({
            "prompt": build_prompt(question),
            "multi_modal_data": {"image": image},
        })
        meta.append({
            "id": data["id"],
            "task_type": task_type,
            "answer": data["answer"],
            "image_file": image_file,
            "question": question,
        })
    return requests, meta


def iter_chunks(items, size):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def load_existing_results(resume_path):
    if not resume_path or not os.path.exists(resume_path):
        return [], set()
    results = []
    seen_ids = set()
    with open(resume_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            results.append(item)
            seen_ids.add(item["id"])
    return results, seen_ids


def append_jsonl(path, items):
    if not path or not items:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_trace_log(path, items):
    if not path or not items:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write("=" * 100 + "\n")
            f.write(f"ID: {item['id']}\n")
            f.write(f"Task: {item['task_type']}\n")
            f.write(f"Image: {item['image_file']}\n")
            f.write(f"Question:\n{item['question']}\n\n")
            f.write(f"Ground Truth:\n{item['ground_truth']}\n\n")
            f.write(f"Model Full Output:\n{item['model_full_output']}\n\n")
            f.write(f"Extracted Answer:\n{item['extracted_answer']}\n")
            f.write(f"Is Correct: {item['is_correct']}\n")
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def format_duration(seconds):
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def build_run_summary(outputs, total, elapsed, dataset_file, result_path, resume_path, start_completed):
    task_counts = {}
    mc_total = 0
    mc_correct = 0
    for item in outputs:
        task = item.get("task_type", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
        if task == "mc":
            mc_total += 1
            if item.get("is_correct") is True:
                mc_correct += 1
    new_samples = max(len(outputs) - start_completed, 0)
    return {
        "dataset_file": dataset_file,
        "result_path": result_path,
        "resume_path": resume_path,
        "completed": len(outputs),
        "dataset_total": total,
        "completion_rate": len(outputs) / total if total else None,
        "task_counts": task_counts,
        "mc_correct": mc_correct,
        "mc_total": mc_total,
        "mc_accuracy": mc_correct / mc_total if mc_total else None,
        "elapsed_seconds_this_run": elapsed,
        "elapsed_readable_this_run": format_duration(elapsed),
        "new_samples_this_run": new_samples,
        "avg_seconds_per_new_sample": elapsed / max(new_samples, 1),
    }


def write_summary_files(summary, out_dir):
    json_path = os.path.join(out_dir, "baseline_vllm_run_summary.json")
    md_path = os.path.join(out_dir, "baseline_vllm_run_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    accuracy = summary["mc_accuracy"]
    accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"
    completion = summary["completion_rate"]
    completion_text = f"{completion * 100:.2f}%" if completion is not None else "N/A"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Baseline vLLM 运行汇总\n\n")
        f.write(f"- 已完成：{summary['completed']}/{summary['dataset_total']} ({completion_text})\n")
        f.write(f"- 本次运行耗时：{summary['elapsed_readable_this_run']}\n")
        f.write(f"- 本次新增样本：{summary['new_samples_this_run']}\n")
        f.write(f"- 平均耗时：{summary['avg_seconds_per_new_sample']:.2f} 秒/条\n")
        f.write(f"- MC 准确率：{summary['mc_correct']}/{summary['mc_total']} = {accuracy_text}\n")
        f.write(f"- 任务数量：`{json.dumps(summary['task_counts'], ensure_ascii=False)}`\n")
        f.write(f"- 结果文件：`{summary['result_path']}`\n")
        f.write(f"- 断点文件：`{summary['resume_path']}`\n")
    return json_path, md_path


def main(args):
    records = load_records(args.dataset_file)
    os.makedirs(args.out_dir, exist_ok=True)
    resume_path = args.resume_file or os.path.join(args.out_dir, "baseline_vllm_resume.jsonl")
    trace_path = args.trace_file or os.path.join(args.out_dir, "baseline_vllm_trace.log")
    existing_outputs, seen_ids = load_existing_results(resume_path)
    records_to_run = [record for record in records if record["id"] not in seen_ids]
    if seen_ids:
        print(f"Resume enabled: loaded {len(seen_ids)} finished samples from {resume_path}")
    if not records_to_run:
        final_path = os.path.join(args.out_dir, f"baseline_vllm_results_{time.strftime('%y%m%d_%H%M')}.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(existing_outputs, f, indent=4, ensure_ascii=False)
        print(f"All samples already finished. Results saved to {final_path}")
        return

    print(f"Loading vLLM Qwen-VL from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    outputs = existing_outputs
    total = len(records)
    start_completed = len(seen_ids)
    print(
        f"Starting vLLM Baseline Inference: total={total}, "
        f"finished={start_completed}, remaining={len(records_to_run)}, batch_size={args.batch_size}"
    )
    start_time = time.perf_counter()
    progress = tqdm(total=total, initial=start_completed, unit="sample", desc="Baseline vLLM", dynamic_ncols=True)
    for batch_idx, (_, chunk) in enumerate(iter_chunks(records_to_run, args.batch_size), start=1):
        batch_start = time.perf_counter()
        requests, meta = make_requests(chunk, args.image_dir)
        responses = llm.generate(
            requests,
            sampling_params=sampling_params,
            use_tqdm=args.vllm_progress,
        )
        batch_outputs = []
        for response, item in zip(responses, meta):
            full_text = response.outputs[0].text
            extracted_ans = extract_answer(full_text, item["task_type"])
            result = {
                "id": item["id"],
                "task_type": item["task_type"],
                "image_file": item["image_file"],
                "question": item["question"],
                "model_full_output": full_text,
                "extracted_answer": extracted_ans,
                "ground_truth": item["answer"],
                "is_correct": str(extracted_ans).strip().lower() == str(item["answer"]).strip().lower()
                if item["task_type"] == "mc" else None,
            }
            batch_outputs.append(result)
        append_jsonl(resume_path, batch_outputs)
        if not args.no_trace_log:
            append_trace_log(trace_path, batch_outputs)
        outputs.extend(batch_outputs)
        progress.update(len(chunk))
        elapsed = time.perf_counter() - start_time
        completed = len(outputs)
        newly_completed = completed - start_completed
        avg = elapsed / newly_completed if newly_completed else 0
        remaining = avg * (total - completed)
        batch_elapsed = time.perf_counter() - batch_start
        progress.set_postfix({
            "batch": batch_idx,
            "batch_s": f"{batch_elapsed:.1f}",
            "avg_s": f"{avg:.2f}",
            "eta": format_duration(remaining),
        })
    progress.close()

    out_path = os.path.join(args.out_dir, f"baseline_vllm_results_{time.strftime('%y%m%d_%H%M')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    elapsed = time.perf_counter() - start_time
    summary = build_run_summary(outputs, total, elapsed, args.dataset_file, out_path, resume_path, start_completed)
    summary_json, summary_md = write_summary_files(summary, args.out_dir)
    print(
        f"Baseline vLLM finished: {len(outputs)}/{total} samples, "
        f"elapsed={format_duration(elapsed)}, avg={elapsed / max(len(outputs) - start_completed, 1):.2f}s/new_sample"
    )
    print(f"Results saved to {out_path}")
    print(f"Resume checkpoint kept at {resume_path}")
    if not args.no_trace_log:
        print(f"Trace log kept at {trace_path}")
    print(f"Run summary saved to {summary_md} and {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset-file", type=str, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--min-pixels", type=int, default=28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--vllm-progress", action="store_true", help="Show vLLM's per-batch internal progress bars.")
    parser.add_argument("--resume-file", type=str, default=None, help="JSONL checkpoint file. Defaults to OUT_DIR/baseline_vllm_resume.jsonl.")
    parser.add_argument("--trace-file", type=str, default=None, help="Readable input/output trace log. Defaults to OUT_DIR/baseline_vllm_trace.log.")
    parser.add_argument("--no-trace-log", action="store_true", help="Disable readable input/output trace logging.")
    main(parser.parse_args())
