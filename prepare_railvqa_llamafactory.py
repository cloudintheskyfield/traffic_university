import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert train driver and railway safety inspector. "
    "Carefully analyze the provided railway scene image and answer the question following a strict 4-step cognitive process.\n\n"
    "Format your response EXACTLY as follows:\n"
    "Perception: Start with 'Visual analysis: ' and describe the environment, weather, tracks, trains, and signals.\n"
    "Reasoning: Start with 'Logical analysis: ' and analyze the situation based on safety protocols and your perception.\n"
    "Planning: Start with 'Action plan: ' and detail the necessary actions to ensure safety.\n"
    "Answer: Provide the final direct answer or correct option letter."
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_question(row):
    question = row["question"].strip()
    if row.get("task_type") == "mc" and row.get("options"):
        lines = [question, "", "Options:"]
        for key in sorted(row["options"]):
            lines.append(f"{key}: {row['options'][key]}")
        lines.append("")
        lines.append("For the 'Answer' section, explicitly provide ONLY the correct option letter (A, B, C, or D).")
        return "\n".join(lines)
    return question


def build_output(row, include_cot):
    answer = str(row["answer"]).strip()
    if not include_cot:
        return f"Answer: {answer}"
    return "\n".join(
        [
            f"Perception: {row.get('cot_perception', '').strip()}",
            f"Reasoning: {row.get('cot_reasoning', '').strip()}",
            f"Planning: {row.get('cot_planning', '').strip()}",
            f"Answer: {answer}",
        ]
    ).strip()


def convert(args):
    dataset_file = Path(args.dataset_file)
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {}
    for row in read_jsonl(dataset_file):
        task_type = row.get("task_type", "qa")
        if args.task_type != "all" and task_type != args.task_type:
            continue
        image_path = image_dir / row["image_id"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for id={row.get('id')}: {image_path}")
        counts[task_type] = counts.get(task_type, 0) + 1
        rows.append(
            {
                "instruction": "<image>\n" + build_question(row),
                "input": "",
                "output": build_output(row, args.include_cot),
                "system": SYSTEM_PROMPT,
                "images": [str(image_path.resolve())],
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    dataset_info = {
        args.dataset_name: {
            "file_name": output_file.name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
                "images": "images",
            },
        }
    }
    with open(output_file.parent / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(rows)} samples to {output_file}")
    print(f"Wrote dataset info to {output_file.parent / 'dataset_info.json'}")
    print(f"Task counts: {json.dumps(counts, ensure_ascii=False)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", default="/mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1/train.jsonl")
    parser.add_argument("--image-dir", default="/mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1/train_images")
    parser.add_argument("--output-file", default="/mnt1/mnt2/data3/nlp/ws/llamafactory_data/railvqa_train_qwen3vl.json")
    parser.add_argument("--dataset-name", default="railvqa_train")
    parser.add_argument("--task-type", choices=["all", "qa", "mc"], default="all")
    parser.add_argument("--include-cot", action=argparse.BooleanOptionalAction, default=True)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
