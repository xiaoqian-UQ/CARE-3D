"""Own one local Qwen judge server for one training job; clean it up on exit.

Usage (from the repository root):
    SCENECOT_ANSWER_JUDGE_BACKEND=local_qwen \
    SCENECOT_ANSWER_JUDGE_MODEL=Qwen/Qwen3-VL-4B-Instruct \
    python scripts/rl/run_with_local_qwen_judge.py bash scripts/rl/train_care3d.sh
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[2]


def server_configuration(env: dict[str, str], port: int) -> tuple[list[str], dict[str, str]]:
    visible = [item.strip() for item in env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",") if item.strip()]
    gpu_index = int(env.get("SCENECOT_LOCAL_JUDGE_GPU_INDEX", "1"))
    if not 0 <= gpu_index < len(visible):
        raise ValueError("SCENECOT_LOCAL_JUDGE_GPU_INDEX must index a GPU in CUDA_VISIBLE_DEVICES")
    python = Path(env.get("SCENECOT_LOCAL_JUDGE_PYTHON", sys.executable)).absolute()
    model = Path(env.get("SCENECOT_LOCAL_JUDGE_MODEL_PATH", str(REPO_ROOT / "model_assets/qwen3-vl-4b-instruct"))).resolve()
    served_model = env.get("SCENECOT_ANSWER_JUDGE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    qwen35 = served_model == "Qwen/Qwen3.5-2B"
    command = [
        str(python), "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model), "--served-model-name", served_model,
        "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", "1",
        "--dtype", "bfloat16", "--enforce-eager",
        "--gpu-memory-utilization", env.get("SCENECOT_LOCAL_JUDGE_GPU_MEMORY_UTIL", "0.12" if qwen35 else "0.18"),
        "--max-model-len", env.get("SCENECOT_LOCAL_JUDGE_MAX_MODEL_LEN", "4096"),
        "--max-num-seqs", env.get("SCENECOT_LOCAL_JUDGE_MAX_NUM_SEQS", "8" if qwen35 else "4"),
        "--max-num-batched-tokens", "4096", "--seed", "42",
    ]
    if qwen35:
        command += ["--language-model-only", "--default-chat-template-kwargs", '{"enable_thinking": false}']
    else:
        # Compatible with the Qwen3-VL vLLM stack already used by verl.
        # Judge requests contain text only; no multimodal profiling is needed.
        command += ["--limit-mm-per-prompt", '{"image": 0, "video": 0}']
    # The independent server must not import training sitecustomize/patches,
    # inherit its VLLM settings, or receive cloud API credentials.
    server_env = {k: v for k, v in env.items() if k not in {
        "PYTHONPATH", "PYTHONHOME", "OPENAI_API_KEY", "GEMINI_API_KEY", "WANDB_API_KEY",
        "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
    } and not k.startswith(("VLLM_", "RAY_", "SCENECOT_"))}
    server_env.update(CUDA_VISIBLE_DEVICES=visible[gpu_index], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not qwen35:
        # vLLM 0.12 auto-selection imports every candidate, including FlashInfer.
        # flashinfer/cubin versions can disagree on some clusters; select FlashAttention
        # explicitly so that discovery never imports the broken candidate.
        server_env.update(VLLM_ATTENTION_BACKEND="FLASH_ATTN", VLLM_USE_FLASHINFER_SAMPLER="0")
        # On H100, vLLM 0.12 also imports FlashInfer for kernel autotuning even
        # with FLASH_ATTN selected. Match the existing training launcher's
        # compatibility setting before starting the judge (training starts later).
        server_env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    return command, server_env


def check_cuda_environment(python: str, env: dict[str, str]) -> None:
    """Check the actual serving interpreter/GPU before loading model weights."""
    subprocess.run([python, "-c", '''
import sys
import os
import torch
import vllm
if torch.version.cuda is None:
    raise RuntimeError("Local judge has CPU-only PyTorch; use the CUDA-enabled verl environment")
if not torch.cuda.is_available():
    raise RuntimeError("Local judge cannot access CUDA; launch on a GPU node")
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
if os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK") == "1":
    from vllm.utils.flashinfer import has_flashinfer, autotune
    if has_flashinfer():
        import flashinfer
        # Exercise the same lazy import/context entry as vLLM's H100 warmup.
        # This does not run model kernels or prove cross-version binary compatibility.
        with torch.inference_mode(), autotune():
            pass
        print(f"Judge FlashInfer preflight OK: version={flashinfer.__version__} "
              "FLASHINFER_DISABLE_VERSION_CHECK=1", flush=True)
print(f"Judge environment OK: python={sys.executable} torch={torch.__version__} "
      f"cuda={torch.version.cuda} vllm={vllm.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
'''], env=env, cwd="/tmp", check=True)


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    # Kill the owned process group as vLLM may have spawned an engine child.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main() -> int:
    env = os.environ.copy()
    os.chdir(REPO_ROOT)
    dry_run = env.get("PRINT_HYDRA_CMD_ONLY") == "1"
    if dry_run:
        port = 18080
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    env["SCENECOT_ANSWER_JUDGE_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    command, server_env = server_configuration(env, port)
    train_command = sys.argv[1:] or ["bash", "scripts/rl/train_care3d.sh"]
    if dry_run:
        print(f"Local judge plan: python={command[0]} GPU={server_env['CUDA_VISIBLE_DEVICES']} "
              f"model={command[command.index('--model') + 1]} (server not started)", flush=True)
        return subprocess.call(train_command, env=env)
    if not Path(command[0]).is_file() or not (Path(command[command.index("--model") + 1]) / "config.json").is_file():
        raise RuntimeError("Local judge Python or model config.json missing; check SCENECOT_LOCAL_JUDGE_PYTHON "
                           "and SCENECOT_LOCAL_JUDGE_MODEL_PATH (qwen3 uses verl and existing actor base weights)")
    check_cuda_environment(command[0], server_env)
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"local_qwen_judge_{env.get('SLURM_JOB_ID', str(os.getpid()))}.log"
    training = server = None
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        with log_path.open("w") as log:
            server = subprocess.Popen(command, env=server_env, cwd="/tmp", stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            print(f"Local judge PID={server.pid} GPU={server_env['CUDA_VISIBLE_DEVICES']} log={log_path}", flush=True)
            print(f"Local judge attention backend={server_env.get('VLLM_ATTENTION_BACKEND', 'auto')}", flush=True)
            print("Local judge FLASHINFER_DISABLE_VERSION_CHECK="
                  f"{server_env.get('FLASHINFER_DISABLE_VERSION_CHECK', 'unset')}", flush=True)
            deadline = time.monotonic() + float(env.get("SCENECOT_LOCAL_JUDGE_STARTUP_TIMEOUT", "600"))
            opener = request.build_opener(request.ProxyHandler({}))
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Local judge exited during startup; see {log_path}")
                try:
                    with opener.open(env["SCENECOT_ANSWER_JUDGE_BASE_URL"] + "/models", timeout=2) as response:
                        models = json.loads(response.read())["data"]
                    if any(m.get("id") == env["SCENECOT_ANSWER_JUDGE_MODEL"] for m in models):
                        break
                except (OSError, ValueError, KeyError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Local judge readiness timed out; see {log_path}")
                time.sleep(2)
            # Fail before the expensive actor rollout if the model/schema is incompatible.
            subprocess.run([env.get("PYTHON_BIN", sys.executable), "scripts/rl/check_answer_judge.py"],
                           env=env, check=True)
            training = subprocess.Popen(train_command, env=env, start_new_session=True)
            while training.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Local judge died during training; see {log_path}")
                time.sleep(2)
            return training.returncode
    finally:
        stop_process(training)
        stop_process(server)


if __name__ == "__main__":
    raise SystemExit(main())
