"""Shared IPC naming for the actor sender and spawned vLLM receivers."""

import hashlib
import json
import re

WEIGHT_SYNC_NAMESPACE_ENV = "VERL_WEIGHT_SYNC_NAMESPACE"


def weight_sync_namespace(gcs_address: str, ray_job_id: str) -> str:
    """Ray job IDs are only unique within a cluster; include its GCS endpoint."""
    if not gcs_address or not ray_job_id:
        raise ValueError("Weight sync requires both a Ray cluster address and job ID")
    identity = json.dumps([gcs_address, ray_job_id], separators=(",", ":"))
    # Keep Unix socket paths short, including when the GCS uses an IPv6 address.
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def weight_sync_ipc_handle(namespace: str, replica_rank: int, local_rank: int) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", namespace):
        raise ValueError("Missing or invalid VERL_WEIGHT_SYNC_NAMESPACE; update sender and receiver together")
    replica_rank, local_rank = int(replica_rank), int(local_rank)
    if replica_rank < 0 or local_rank < 0:
        raise ValueError("Weight sync ranks must be non-negative")
    return f"ipc:///tmp/rl-colocate-zmq-{namespace}-replica-{replica_rank}-rank-{local_rank}.sock"
