import faiss
import numpy as np
import torch
import torchaudio
from insightface.app import FaceAnalysis


def check_pytorch_gpu():
    print("=== PyTorch GPU ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch GPU not available")


def check_torchaudio_gpu():
    print("\n=== torchaudio GPU ===")
    try:
        device = torch.device("cuda")
        x = torch.randn(1, 16000).to(device)
        resampler = torchaudio.transforms.Resample(16000, 8000).to(device)
        y = resampler(x)
        print(f"torchaudio GPU operation success, output device: {y.device}")
    except Exception as e:
        print(f"torchaudio GPU test failed: {e}")


def check_faiss_gpu():
    print("\n=== FAISS GPU ===")
    print(f"FAISS version: {faiss.__version__}")
    ngpu = faiss.get_num_gpus()
    print(f"GPUs detected by FAISS: {ngpu}")
    if ngpu > 0:
        try:
            index = faiss.IndexFlatL2(128)
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
            print("FAISS GPU index created successfully")
            # Clean up
            del gpu_index
        except Exception as e:
            print(f"FAISS GPU test failed: {e}")


if __name__ == "__main__":
    check_pytorch_gpu()
    check_torchaudio_gpu()
    check_faiss_gpu()

    print("NumPy version:", np.__version__)
    print("PyTorch CUDA available:", torch.cuda.is_available())
    print("TorchAudio version:", torchaudio.__version__)

    # 测试 FaceAnalysis 初始化
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    print("InsightFace FaceAnalysis initialized successfully")
