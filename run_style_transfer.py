"""Command-line entry point for classic and fast TensorFlow style transfer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

from style_transfer.config import TransferConfig, validate_image_size
from style_transfer.console import configure_utf8_console
from style_transfer.runtime import configure_runtime_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural Image Style Transfer bằng TensorFlow.")
    parser.add_argument("--content", required=True, help="Ảnh cung cấp nội dung/bố cục.")
    parser.add_argument("--style", required=True, help="Ảnh cung cấp phong cách.")
    parser.add_argument("--output", required=True, help="Đường dẫn ảnh kết quả.")
    parser.add_argument("--method", choices=("classic", "fast"), default="classic")
    parser.add_argument("--max-dim", type=int, default=512, help="Cạnh ảnh tối đa.")
    parser.add_argument("--steps", type=int, default=400, help="Số bước cho phương pháp classic.")
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--content-weight", type=float, default=1e4)
    parser.add_argument("--style-weight", type=float, default=1e-2)
    parser.add_argument("--variation-weight", type=float, default=30.0)
    parser.add_argument("--init", choices=("content", "noise"), default="content")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Lưu ảnh trung gian sau mỗi N bước; 0 là không lưu.",
    )
    return parser


def _configure_device(tf: Any, requested: str) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if requested == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError("Không thể chuyển sang CPU sau khi TensorFlow đã khởi tạo GPU.") from exc
        return "CPU"
    if requested == "gpu" and not gpus:
        raise RuntimeError("Bạn chọn GPU nhưng TensorFlow không phát hiện GPU nào.")
    return gpus[0].name if gpus else "CPU"


def _intermediate_path(output: Path, step: int) -> Path:
    return output.with_name(f"{output.stem}_step_{step:04d}{output.suffix}")


def main() -> int:
    configure_utf8_console()
    configure_runtime_cache()
    args = build_parser().parse_args()
    try:
        validate_image_size(args.max_dim)
        if args.log_every <= 0:
            raise ValueError("--log-every phải lớn hơn 0.")
        if args.save_every < 0:
            raise ValueError("--save-every không được âm.")
    except ValueError as exc:
        print(f"Lỗi tham số: {exc}")
        return 2

    try:
        import tensorflow as tf

        from style_transfer.image_utils import load_image, save_image, validate_image_path
    except ImportError as exc:
        print(
            "Thiếu dependency TensorFlow. Tạo môi trường và chạy "
            "'python -m pip install -r requirements.txt' trước."
        )
        print(f"Chi tiết: {exc}")
        return 1

    try:
        content_path = validate_image_path(args.content)
        style_path = validate_image_path(args.style)
        output_path = Path(args.output).expanduser().resolve()
        device = _configure_device(tf, args.device)
        content_image = load_image(content_path, args.max_dim)
        style_image = load_image(style_path, args.max_dim)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Không thể chuẩn bị input: {exc}")
        return 1

    print(f"TensorFlow: {tf.__version__}")
    print(f"Thiết bị: {device}")
    print(f"Content: {content_path}")
    print(f"Style: {style_path}")
    print(f"Phương pháp: {args.method}")
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []

    try:
        if args.method == "fast":
            from style_transfer.hub_transfer import run_fast_style_transfer

            result = run_fast_style_transfer(content_image, style_image)
        else:
            from style_transfer.engine import StepMetrics, run_classic_style_transfer

            config = TransferConfig(
                steps=args.steps,
                learning_rate=args.learning_rate,
                content_weight=args.content_weight,
                style_weight=args.style_weight,
                total_variation_weight=args.variation_weight,
                init=args.init,
                seed=args.seed,
            )

            def report(metrics: StepMetrics, current_image: Any) -> None:
                history.append(asdict(metrics))
                if metrics.step == 1 or metrics.step % args.log_every == 0:
                    print(
                        f"Step {metrics.step:4d}/{config.steps}: "
                        f"loss={metrics.total_loss:.4f}, "
                        f"style={metrics.style_loss:.4f}, "
                        f"content={metrics.content_loss:.4f}"
                    )
                if args.save_every and metrics.step % args.save_every == 0:
                    save_image(current_image, _intermediate_path(output_path, metrics.step))

            result, _ = run_classic_style_transfer(
                content_image,
                style_image,
                config,
                progress_callback=report,
            )
        saved_output = save_image(result, output_path)
    except Exception as exc:
        print(f"Style transfer thất bại: {type(exc).__name__}: {exc}")
        return 1

    elapsed = time.perf_counter() - started
    metrics_path = saved_output.with_suffix(".metrics.json")
    payload = {
        "method": args.method,
        "content": str(content_path),
        "style": str(style_path),
        "output": str(saved_output),
        "max_dim": args.max_dim,
        "steps": args.steps if args.method == "classic" else None,
        "device": device,
        "tensorflow_version": tf.__version__,
        "elapsed_seconds": elapsed,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "history": history,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hoàn tất sau {elapsed:.2f} giây.")
    print(f"Ảnh kết quả: {saved_output}")
    print(f"Thông số/loss: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

