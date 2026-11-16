"""CPU regressions for independent Ray clusters sharing a host's /tmp."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import zmq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weight_sync_ipc", ROOT / "verl/verl/utils/weight_sync_ipc.py")
ipc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ipc)


def test_clusters_jobs_and_ranks_have_distinct_paths():
    handles = {
        ipc.weight_sync_ipc_handle(ipc.weight_sync_namespace(cluster, job), replica, rank)
        for cluster in ("10.0.0.1:40001", "10.0.0.1:40002")
        for job in ("01000000", "02000000")
        for replica in (0, 1)
        for rank in (0, 1)
    }
    assert len(handles) == 16
    assert all(len(handle.removeprefix("ipc://").encode()) < 108 for handle in handles)
    ipv6 = ipc.weight_sync_namespace("[2001:db8:1234:5678:90ab:cdef:1234:5678]:65535", "01000000")
    assert len(ipc.weight_sync_ipc_handle(ipv6, 1, 0).removeprefix("ipc://").encode()) < 108


@pytest.mark.parametrize("namespace", ["", "01000000", "../unsafe"])
def test_missing_or_legacy_namespace_is_rejected(namespace):
    with pytest.raises(ValueError, match="VERL_WEIGHT_SYNC_NAMESPACE"):
        ipc.weight_sync_ipc_handle(namespace, 0, 0)


def _load_method(file, class_name, method_name, globals_dict):
    # Exercise the actual wiring without importing GPU-only vLLM dependencies.
    tree = ast.parse((ROOT / file).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=[*future, method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), file, "exec"), globals_dict)
    return globals_dict[method_name]


def test_server_forwards_namespace_and_worker_resolves_same_path(monkeypatch):
    for key in (ipc.WEIGHT_SYNC_NAMESPACE_ENV, "VERL_RAY_JOB_ID", "VERL_REPLICA_RANK", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.setenv(key, "old")
    context = SimpleNamespace(gcs_address="10.0.0.1:40002", get_job_id=lambda: "01000000")
    scope = dict(os=os, ray=SimpleNamespace(get_runtime_context=lambda: context),
                 get_visible_devices_keyword=lambda: "CUDA_VISIBLE_DEVICES",
                 WEIGHT_SYNC_NAMESPACE_ENV=ipc.WEIGHT_SYNC_NAMESPACE_ENV,
                 weight_sync_namespace=ipc.weight_sync_namespace)
    init = _load_method("verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py",
                        "vLLMHttpServer", "__init__", scope)
    class StopBeforeGPU(Exception):
        pass
    def stop(config):
        raise StopBeforeGPU()
    with pytest.raises(StopBeforeGPU):
        init(SimpleNamespace(_init_config=stop), config=None, model_config=None,
             rollout_mode=None, workers=[], replica_rank=1, node_rank=0,
             gpus_per_node=1, nnodes=1, cuda_visible_devices="1")
    namespace = ipc.weight_sync_namespace(context.gcs_address, context.get_job_id())
    assert os.environ[ipc.WEIGHT_SYNC_NAMESPACE_ENV] == namespace
    get_handle = _load_method("verl/verl/workers/rollout/vllm_rollout/utils.py",
                              "vLLMColocateWorkerExtension", "_get_zmq_handle",
                              dict(os=os, WEIGHT_SYNC_NAMESPACE_ENV=ipc.WEIGHT_SYNC_NAMESPACE_ENV,
                                   weight_sync_ipc_handle=ipc.weight_sync_ipc_handle))
    assert get_handle(SimpleNamespace(local_rank=0)) == ipc.weight_sync_ipc_handle(namespace, 1, 0)
    monkeypatch.delenv(ipc.WEIGHT_SYNC_NAMESPACE_ENV)
    with pytest.raises(ValueError, match="VERL_WEIGHT_SYNC_NAMESPACE"):
        get_handle(SimpleNamespace(local_rank=0))


def test_two_live_ipc_channels_do_not_cross_or_delete_each_other():
    # Reproduce equal Ray job/replica/local IDs across two independent clusters.
    run = uuid.uuid4().hex
    handles = [ipc.weight_sync_ipc_handle(ipc.weight_sync_namespace(f"{run}:{port}", "01000000"), 1, 0)
               for port in (40001, 40002)]
    context = zmq.Context()
    sockets = []
    try:
        for handle in handles:
            sender, receiver = context.socket(zmq.REQ), context.socket(zmq.REP)
            sockets.extend((sender, receiver))
            for sock in (sender, receiver):
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.SNDTIMEO, 3000)
                sock.setsockopt(zmq.RCVTIMEO, 3000)
            sender.bind(handle)
            receiver.connect(handle)
        for index in (0, 1):
            sockets[2 * index].send_string(f"weights-{index}")
        for index in (1, 0):
            assert sockets[2 * index + 1].recv_string() == f"weights-{index}"
            sockets[2 * index + 1].send(b"ack")
            assert sockets[2 * index].recv() == b"ack"
        for sock in sockets[:2]:
            sock.close()
        Path(handles[0].removeprefix("ipc://")).unlink(missing_ok=True)
        sockets[2].send(b"next-step")
        assert sockets[3].recv() == b"next-step"
        sockets[3].send(b"ack")
        assert sockets[2].recv() == b"ack"
    finally:
        for sock in sockets:
            sock.close()
        context.term()
        for handle in handles:
            Path(handle.removeprefix("ipc://")).unlink(missing_ok=True)
