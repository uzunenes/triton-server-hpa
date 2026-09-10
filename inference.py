#!/usr/bin/env python3
"""Triton client for the end2end YOLOv7-tiny TensorRT model.

Two modes:

  detect  run a single inference and write an annotated image (default)
  load    keep N threads hammering the server so GPU utilisation rises and
          the HorizontalPodAutoscaler has something to react to

The model is exported with `--end2end`, so the TensorRT EfficientNMS plugin
has already applied NMS on the server side. The client only has to undo the
letterbox padding and draw the boxes.
"""

import argparse
import queue
import sys
import threading
import time

import cv2
import numpy as np
import tritonclient.http as httpclient

MODEL_INPUT = "images"
MODEL_OUTPUTS = ("num_dets", "det_boxes", "det_scores", "det_classes")
INPUT_SIZE = 640

# COCO class names, in the order YOLOv7 was trained on.
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def letterbox(image, size=INPUT_SIZE):
    """Resize preserving aspect ratio and pad to a square, like YOLOv7 does.

    Returns the padded image plus the scale and padding needed to map boxes
    back onto the original image.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x, pad_y = (size - new_w) / 2, (size - new_h) / 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    bottom, right = size - new_h - top, size - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, scale, (left, top)


def preprocess(image):
    """BGR uint8 HWC image -> NCHW float32 batch of 1, plus letterbox metadata."""
    padded, scale, pad = letterbox(image)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)  # YOLOv7 is trained on RGB
    chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(chw[None, ...]), scale, pad


def build_request(batch):
    """Fresh InferInput/InferRequestedOutput objects. These are NOT thread safe,
    so every thread builds its own."""
    inp = httpclient.InferInput(MODEL_INPUT, batch.shape, "FP32")
    inp.set_data_from_numpy(batch)
    outs = [httpclient.InferRequestedOutput(name) for name in MODEL_OUTPUTS]
    return [inp], outs


def parse_detections(response, scale, pad, threshold):
    """Turn a Triton response into (box, score, class_id) in original-image space."""
    num_dets = int(response.as_numpy("num_dets")[0][0])
    boxes = response.as_numpy("det_boxes")[0]
    scores = response.as_numpy("det_scores")[0]
    classes = response.as_numpy("det_classes")[0]

    pad_x, pad_y = pad
    detections = []
    for i in range(num_dets):
        score = float(scores[i])
        if score < threshold:
            continue
        x_min, y_min, x_max, y_max = boxes[i]
        detections.append(
            (
                [
                    (x_min - pad_x) / scale,
                    (y_min - pad_y) / scale,
                    (x_max - pad_x) / scale,
                    (y_max - pad_y) / scale,
                ],
                score,
                int(classes[i]),
            )
        )
    return detections


def draw(image, detections):
    for (x_min, y_min, x_max, y_max), score, class_id in detections:
        p1 = (int(x_min), int(y_min))
        p2 = (int(x_max), int(y_max))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        cv2.putText(
            image, f"{name} {score:.2f}", (p1[0], max(p1[1] - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return image


def run_detect(args):
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")

    batch, scale, pad = preprocess(image)
    inputs, outputs = build_request(batch)

    with httpclient.InferenceServerClient(url=args.url) as client:
        response = client.infer(
            model_name=args.model, model_version=args.model_version,
            inputs=inputs, outputs=outputs,
        )

    detections = parse_detections(response, scale, pad, args.threshold)
    print(f"{len(detections)} detection(s) above {args.threshold}")
    for (x_min, y_min, x_max, y_max), score, class_id in detections:
        name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
        print(f"  {name:<15} {score:.3f}  [{x_min:.0f} {y_min:.0f} {x_max:.0f} {y_max:.0f}]")

    cv2.imwrite(args.output, draw(image, detections))
    print(f"wrote {args.output}")


def run_load(args):
    """Saturate the GPU for --duration seconds so the HPA sees load."""
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")
    batch, _, _ = preprocess(image)

    stop = threading.Event()
    errors = queue.Queue()
    counts = [0] * args.threads

    def worker(index):
        # Per-thread client and tensors: tritonclient objects are not thread safe.
        inputs, outputs = build_request(batch)
        try:
            with httpclient.InferenceServerClient(url=args.url) as client:
                while not stop.is_set():
                    client.infer(
                        model_name=args.model, model_version=args.model_version,
                        inputs=inputs, outputs=outputs,
                    )
                    counts[index] += 1
                    if args.interval:
                        stop.wait(args.interval)
        except Exception as exc:  # noqa: BLE001 - report and let the run continue
            errors.put(f"thread {index}: {exc}")

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(args.threads)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()

    print(f"{args.threads} threads against {args.url} for {args.duration}s "
          f"(Ctrl-C to stop early)")
    # Redraw one line on a terminal; append whole lines when piped to a file.
    interactive = sys.stdout.isatty()
    tick = 1 if interactive else 15
    try:
        next_report = tick
        while time.monotonic() - started < args.duration:
            time.sleep(0.25)
            elapsed = time.monotonic() - started
            if elapsed < next_report:
                continue
            next_report = elapsed + tick
            total = sum(counts)
            print(f"  {elapsed:6.1f}s  {total:8d} requests  "
                  f"{total / max(elapsed, 1e-9):7.1f} req/s",
                  end="\r" if interactive else "\n", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=10)

    elapsed = time.monotonic() - started
    total = sum(counts)
    print(f"\n{total} requests in {elapsed:.1f}s -> {total / elapsed:.1f} req/s")

    failures = []
    while not errors.empty():
        failures.append(errors.get())
    if failures:
        print(f"{len(failures)} thread(s) failed:")
        for failure in failures[:5]:
            print(f"  {failure}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("detect", "load"), default="detect")
    parser.add_argument("--url", default="localhost:8000",
                        help="Triton HTTP endpoint, no scheme (default: %(default)s)")
    parser.add_argument("--model", default="yolov7tiny")
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--image", default="docs/images/input_image.jpg")
    parser.add_argument("--output", default="detection_result.jpg")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--threads", type=int, default=32, help="load mode only")
    parser.add_argument("--duration", type=float, default=120, help="load mode seconds")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="load mode: sleep between requests, 0 = as fast as possible")
    args = parser.parse_args()

    if args.mode == "detect":
        run_detect(args)
    else:
        run_load(args)


if __name__ == "__main__":
    main()
