import argparse
from pathlib import Path

import numpy as np
import torch

from .models import create_mobilenetv3_encoder, create_resnet50_encoder


def export_to_onnx(
    encoder: torch.nn.Module,
    output_path: str,
    embed_dim: int = 256,
    opset_version: int = 14,
):
    """
    Export a PyTorch encoder to ONNX format.

    Args:
        encoder: trained encoder model
        output_path: path to save .onnx file
        embed_dim: embedding dimension
        opset_version: ONNX opset version
    """
    encoder.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX model to {output_path}")


def export_to_tflite(
    onnx_path: str,
    output_path: str,
    quantize: bool = True,
):
    """
    Convert ONNX model to TFLite format.

    Args:
        onnx_path: path to .onnx model
        output_path: path to save .tflite model
        quantize: whether to apply int8 quantization
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        print("Required packages not installed. Run:")
        print("  pip install onnx onnx-tf tensorflow")
        return

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Convert to TensorFlow
    tf_rep = prepare(onnx_model)
    tf_path = output_path.replace(".tflite", "_tf")
    tf_rep.export_graph(tf_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Representative dataset for int8 quantization
        def representative_dataset():
            for _ in range(100):
                yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"Exported TFLite model to {output_path} ({size_mb:.1f} MB)")

    # Cleanup TF SavedModel
    import shutil
    if Path(tf_path).exists():
        shutil.rmtree(tf_path)


def verify_export(
    original_encoder: torch.nn.Module,
    onnx_path: str,
    num_tests: int = 10,
    tolerance: float = 1e-3,
):
    """
    Verify that ONNX export produces the same embeddings as PyTorch.

    Args:
        original_encoder: trained PyTorch encoder
        onnx_path: path to exported ONNX model
        num_tests: number of random test inputs
        tolerance: max allowed difference per element
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Run: pip install onnxruntime")
        return False

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    original_encoder.eval()
    all_close = True

    for i in range(num_tests):
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

        # PyTorch inference
        with torch.no_grad():
            torch_out = original_encoder(torch.from_numpy(dummy)).numpy()

        # ONNX inference
        ort_out = session.run(None, {input_name: dummy})[0]

        diff = np.abs(torch_out - ort_out).max()
        if diff > tolerance:
            print(f"Test {i}: Max diff {diff:.6f} > tolerance {tolerance}")
            all_close = False

    if all_close:
        print(f"All {num_tests} verification tests passed (max diff < {tolerance})")
    return all_close


def export(
    checkpoint_path: str,
    output_dir: str,
    embed_dim: int = 256,
    export_satellite: bool = True,
    export_phone: bool = True,
    quantize_tflite: bool = True,
):
    """
    Full export pipeline: PyTorch → ONNX → TFLite.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if export_satellite and "sat_encoder" in checkpoint:
        print("Exporting satellite encoder...")
        sat_encoder = create_resnet50_encoder(embed_dim=embed_dim, pretrained=False)
        sat_encoder.load_state_dict(checkpoint["sat_encoder"])
        sat_encoder.eval()

        onnx_path = str(output_path / "satellite_encoder.onnx")
        export_to_onnx(sat_encoder, onnx_path, embed_dim)
        verify_export(sat_encoder, onnx_path)

    if export_phone and "phone_encoder" in checkpoint:
        print("Exporting phone encoder (teacher)...")
        phone_encoder = create_resnet50_encoder(embed_dim=embed_dim, pretrained=False)
        phone_encoder.load_state_dict(checkpoint["phone_encoder"])
        phone_encoder.eval()

        onnx_path = str(output_path / "phone_encoder_teacher.onnx")
        export_to_onnx(phone_encoder, onnx_path, embed_dim)
        verify_export(phone_encoder, onnx_path)

    # Export student model if available
    student_path = output_path.parent / "student" / "best_student.pt"
    if student_path.exists():
        print("Exporting MobileNetV3 student...")
        student = create_mobilenetv3_encoder(embed_dim=embed_dim, pretrained=False)
        student_ckpt = torch.load(student_path, map_location=device, weights_only=False)
        student.load_state_dict(student_ckpt["student"])
        student.eval()

        onnx_path = str(output_path / "phone_encoder.onnx")
        export_to_onnx(student, onnx_path, embed_dim)
        verify_export(student, onnx_path)

        # Convert to TFLite
        tflite_path = str(output_path / "phone_encoder.tflite")
        export_to_tflite(onnx_path, tflite_path, quantize=quantize_tflite)

    print("\nExport complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output_dir", type=str, default="./exported_models")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--no_satellite", action="store_true")
    parser.add_argument("--no_phone", action="store_true")
    parser.add_argument("--no_quantize", action="store_true")
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        export_satellite=not args.no_satellite,
        export_phone=not args.no_phone,
        quantize_tflite=not args.no_quantize,
    )
