import argparse
import json
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

DEFAULT_DATASET_ROOT = "/mnt1/mnt2/data3/nlp/ws/course_data/dataset_divided_1"
DEFAULT_DATASET_FILE = f"{DEFAULT_DATASET_ROOT}/train.jsonl"
DEFAULT_IMAGE_DIR = f"{DEFAULT_DATASET_ROOT}/train_images"
DEFAULT_YOLO_PATH = "/mnt1/mnt2/data3/nlp/ws/model/YOLO26/yolo26m.pt"
DEFAULT_OUT_DIR = "results/yolo_io_test"
DEFAULT_LIMIT = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image filename under --image-dir, or an absolute image path",
    )
    parser.add_argument("--dataset-file", type=str, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--yolo-path", type=str, default=DEFAULT_YOLO_PATH)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--font-size", type=int, default=10)
    return parser.parse_args()


def resolve_image_path(image_arg, image_dir):
    image_path = Path(image_arg)
    if image_path.is_file():
        return image_path

    candidate = Path(image_dir) / image_arg
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Image not found: {image_arg}. Checked direct path and {candidate}"
    )


def load_first_records(dataset_file, limit):
    records = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def extract_detections(result):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return detections

    names = result.names
    xyxy_list = boxes.xyxy.cpu().tolist()
    xywh_list = boxes.xywh.cpu().tolist()
    conf_list = boxes.conf.cpu().tolist()
    cls_list = boxes.cls.cpu().tolist()

    for xyxy, xywh, conf, cls_id in zip(xyxy_list, xywh_list, conf_list, cls_list):
        cls_index = int(cls_id)
        detections.append({
            "class_id": cls_index,
            "class_name": names.get(cls_index, str(cls_index)) if isinstance(names, dict) else names[cls_index],
            "confidence": round(float(conf), 6),
            "bbox_xyxy": [round(float(v), 2) for v in xyxy],
            "bbox_xywh": [round(float(v), 2) for v in xywh],
        })
    return detections


def run_yolo_on_image(model, image_path, args, out_dir):
    image = Image.open(image_path).convert("RGB")
    infer_kwargs = {
        "verbose": False,
        "conf": args.conf,
        "imgsz": args.imgsz,
    }
    if args.device is not None:
        infer_kwargs["device"] = args.device

    print(f"Running YOLO on {image_path}...")
    results = model(image, **infer_kwargs)
    result = results[0]

    annotated_bgr = result.plot(line_width=args.line_width, font_size=args.font_size, labels=True)
    annotated_image = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    detections = extract_detections(result)

    annotated_path = out_dir / f"{image_path.stem}_annotated{image_path.suffix}"
    json_path = out_dir / f"{image_path.stem}_detections.json"
    annotated_image.save(annotated_path)

    payload = {
        "input_image": str(image_path),
        "image_dir": args.image_dir,
        "annotated_image": str(annotated_path),
        "yolo_model": args.yolo_path,
        "device": args.device,
        "confidence_threshold": args.conf,
        "imgsz": args.imgsz,
        "num_detections": len(detections),
        "detections": detections,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Annotated image saved to: {annotated_path}")
    print(f"Detection json saved to: {json_path}")
    print(f"Detections: {len(detections)}")
    for idx, item in enumerate(detections, start=1):
        print(
            f"[{idx}] class={item['class_name']} id={item['class_id']} "
            f"conf={item['confidence']:.4f} bbox_xyxy={item['bbox_xyxy']}"
        )

    return payload


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model from {args.yolo_path}...")
    model = YOLO(args.yolo_path)

    if args.image:
        image_path = resolve_image_path(args.image, args.image_dir)
        run_yolo_on_image(model, image_path, args, out_dir)
        return

    records = load_first_records(args.dataset_file, args.limit)
    print(f"Loaded {len(records)} records from {args.dataset_file}")

    summary = []
    for idx, record in enumerate(records, start=1):
        image_id = record["image_id"]
        image_path = resolve_image_path(image_id, args.image_dir)
        print(f"[{idx}/{len(records)}] image_id={image_id}")
        payload = run_yolo_on_image(model, image_path, args, out_dir)
        payload["sample_id"] = record.get("id")
        payload["image_id"] = image_id
        summary.append(payload)

    summary_path = out_dir / "first_10_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
