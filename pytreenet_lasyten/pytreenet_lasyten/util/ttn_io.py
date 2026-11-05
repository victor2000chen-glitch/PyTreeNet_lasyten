"""TTN checkpoints using LASyTen's native HDF5 tensor serialization.

Install as pytreenet_lasyten/util/ttn_io.py and delegate the core TTN save/load
methods to save_ttn/load_ttn. New checkpoints are single .h5 files. Legacy
NumPy .npz/.json checkpoints can also be loaded. No native tensor is densified.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np

FORMAT = "pytreenet.native_ttn"
FORMAT_VERSION = 1
METADATA_ATTRIBUTES = ("u1_info", "u1_operator_info")


def _hdf5_dependencies():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("TTN HDF5 checkpoints require h5py.") from exc
    try:
        hdf5_io = importlib.import_module("hdf5_io")
    except ImportError as exc:
        raise ImportError(
            "TTN HDF5 checkpoints require LASyTen's optional hdf5_io package. "
            "Install it from the external/hdf5_io submodule of your LASyTen "
            "source checkout if it is unavailable from your package index."
        ) from exc
    return h5py, hdf5_io


def _native_array(tensor):
    from . import lasyten_backend as lb

    if isinstance(tensor, lb.LasyTenArray):
        native = tensor.native
    elif isinstance(tensor, lb._lasyten().Tensor):
        native = tensor
    else:
        raise TypeError("TTN checkpoints support NumPy arrays and native LASyTen tensors only.")
    if not callable(getattr(native, "save_hdf5", None)):
        raise NotImplementedError(
            f"{type(native).__name__} has no native save_hdf5 method. "
            "Use a LASyTen build with native HDF5 I/O enabled."
        )
    return native


def _checkpoint_path(filepath):
    path = Path(filepath)
    if path.suffix.lower() not in (".h5", ".hdf5"):
        path = Path(str(path) + ".h5")
    return path


def _backend_policy(network):
    for name in ("backend", "_backend"):
        value = getattr(network, name, None)
        if isinstance(value, str) and value in ("auto", "numpy", "lasyten"):
            return value
    return None


def _payload(network):
    # TensorDict.__getitem__ resolves lazy permutations by modifying the Node.
    # Read its raw storage instead so saving leaves the live network untouched.
    storage = network.tensors.data
    if set(storage) != set(network.nodes):
        raise ValueError("The tensor dictionary and node dictionary disagree.")
    records = []
    for key, node in network.nodes.items():
        tensor = storage[key]
        if isinstance(tensor, np.ndarray):
            if tensor.dtype.hasobject:
                raise TypeError("Object arrays are not supported as TTN tensors.")
            kind, stored = "numpy", tensor
        else:
            kind, stored = "lasyten", _native_array(tensor)
        shape = tuple(int(d) for d in tensor.shape)
        permutation = list(node.leg_permutation)
        if tuple(node.shape) != tuple(shape[i] for i in permutation):
            raise ValueError(f"Node/tensor shapes disagree at {key!r}.")
        records.append(dict(
            identifier=key,
            parent=node.parent,
            children=list(node.children),
            shape=shape,
            permutation=permutation,
            tensor_kind=kind,
            tensor=stored,
        ))
        if node.identifier != key:
            raise ValueError(f"Node identifier disagrees with its key: {key!r}.")
    payload = dict(
        format=FORMAT,
        format_version=FORMAT_VERSION,
        root_id=network.root_id,
        orthogonality_center_id=network.orthogonality_center_id,
        network_class=f"{type(network).__module__}.{type(network).__qualname__}",
        backend_policy=_backend_policy(network),
        nodes=records,
        metadata={name: getattr(network, name) for name in METADATA_ATTRIBUTES
                  if hasattr(network, name)},
    )
    _validate_payload(payload)
    return payload


def _validate_payload(payload):
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("This is not a PyTreeNet native TTN checkpoint.")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported TTN checkpoint version: {payload.get('format_version')!r}.")
    records = payload.get("nodes")
    if not isinstance(records, list):
        raise ValueError("Checkpoint nodes must be an ordered list.")
    nodes = {}
    for record in records:
        key = record["identifier"]
        if not isinstance(key, str) or key in nodes:
            raise ValueError("Node identifiers must be unique strings.")
        shape = tuple(record["shape"])
        permutation = list(record["permutation"])
        if sorted(permutation) != list(range(len(shape))):
            raise ValueError(f"Invalid leg permutation at {key!r}.")
        if tuple(int(d) for d in record["tensor"].shape) != shape:
            raise ValueError(f"Stored tensor shape disagrees at {key!r}.")
        if record["tensor_kind"] not in ("numpy", "lasyten"):
            raise ValueError(f"Unknown tensor storage kind at {key!r}.")
        children = record["children"]
        if not isinstance(children, list) or len(set(children)) != len(children):
            raise ValueError(f"Invalid child list at {key!r}.")
        if len(children) + int(record["parent"] is not None) > len(shape):
            raise ValueError(f"Too many virtual legs at {key!r}.")
        nodes[key] = record
    root = payload["root_id"]
    center = payload["orthogonality_center_id"]
    if not nodes:
        if root is not None or center is not None:
            raise ValueError("An empty network cannot have a root or a centre.")
        return
    if root not in nodes or nodes[root]["parent"] is not None:
        raise ValueError("Invalid checkpoint root.")
    if center is not None and center not in nodes:
        raise ValueError("The orthogonality centre is absent from the network.")
    seen, pending = set(), [root]
    while pending:
        key = pending.pop()
        if key in seen:
            raise ValueError("The checkpoint topology contains a cycle or a repeated child.")
        seen.add(key)
        record = nodes[key]
        shape = tuple(record["shape"][i] for i in record["permutation"])
        for offset, child in enumerate(record["children"], int(key != root)):
            if child not in nodes or nodes[child]["parent"] != key:
                raise ValueError(f"Inconsistent parent/child link at {key!r}.")
            child_record = nodes[child]
            child_dim = child_record["shape"][child_record["permutation"][0]]
            if shape[offset] != child_dim:
                raise ValueError(f"Bond dimensions disagree on {key!r}--{child!r}.")
            pending.append(child)
    if seen != set(nodes):
        raise ValueError("The checkpoint topology is disconnected.")


def save_ttn(network, filepath):
    """Save a TTN/TTNS/TTNO to one .h5 file, without changing the network.

    File extensions .h5 and .hdf5 are accepted; otherwise .h5 is appended.
    Native tensors are handed to LASyTen's save_hdf5 implementation, including
    their sparse blocks, spaces, LegPipes, labels, and hidden charge state.
    A temporary file is atomically renamed after a successful write.
    """
    h5py, hdf5_io = _hdf5_dependencies()
    payload = _payload(network)
    path = _checkpoint_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["pytreenet_format"] = FORMAT
            handle.attrs["pytreenet_format_version"] = FORMAT_VERSION
            hdf5_io.save_to_hdf5(handle, payload)
            handle.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _restore(cls, payload):
    from ..core.node import Node

    _validate_payload(payload)
    policy = payload.get("backend_policy")
    if policy in ("auto", "numpy", "lasyten") and "backend" in inspect.signature(cls).parameters:
        network = cls(backend=policy)
    else:
        network = cls()
    for record in payload["nodes"]:
        tensor = record["tensor"]
        if record["tensor_kind"] == "lasyten":
            from . import lasyten_backend as lb

            if not isinstance(tensor, lb._lasyten().Tensor):
                raise TypeError("The HDF5 loader did not restore a native LASyTen Tensor.")
            tensor = lb.LasyTenArray(tensor)
        elif not isinstance(tensor, np.ndarray):
            raise TypeError("The HDF5 loader did not restore a NumPy tensor.")
        key = record["identifier"]
        node = Node(identifier=key, tensor=tensor)
        node.parent = record["parent"]
        node.children = list(record["children"])
        network._nodes[key] = node
        network._tensors[key] = tensor
        node._leg_permutation = list(record["permutation"])
    network._root_id = payload["root_id"]
    network.orthogonality_center_id = payload["orthogonality_center_id"]
    metadata = payload.get("metadata", {})
    for name in METADATA_ATTRIBUTES:
        if name in metadata:
            setattr(network, name, metadata[name])
    return network


def _load_legacy(cls, base):
    npz_path, json_path = Path(str(base) + ".npz"), Path(str(base) + ".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Both {npz_path} and {json_path} are required.")
    with json_path.open(encoding="utf-8") as handle:
        structure = json.load(handle)
    records = []
    with np.load(npz_path, allow_pickle=False) as tensors:
        if set(tensors.files) != set(structure["nodes"]):
            raise ValueError("Legacy tensor and node dictionaries disagree.")
        for key, node in structure["nodes"].items():
            tensor = tensors[key].copy()
            if node["identifier"] != key or tuple(node["shape"]) != tensor.shape:
                raise ValueError(f"Legacy node/tensor metadata disagrees at {key!r}.")
            records.append(dict(
                identifier=key, parent=node["parent"], children=list(node["children"]),
                shape=tensor.shape, permutation=list(range(tensor.ndim)),
                tensor_kind="numpy", tensor=tensor,
            ))
    payload = dict(
        format=FORMAT, format_version=FORMAT_VERSION, nodes=records,
        root_id=structure["root_id"],
        orthogonality_center_id=structure["orthogonality_center_id"], metadata={},
    )
    return _restore(cls, payload)


def load_ttn(cls, filepath):
    """Load a native .h5 checkpoint, or a legacy NumPy .npz/.json pair.

    Use TTNS.load(...) for a state and TTNO.load(...) for an operator. The
    class supplied by the caller is retained; the checkpoint never selects
    a different network class. Loading does not canonicalize or reroot.
    """
    requested = Path(filepath)
    suffix = requested.suffix.lower()
    if suffix in (".npz", ".json"):
        return _load_legacy(cls, requested.with_suffix(""))
    if suffix in (".h5", ".hdf5"):
        path = requested
    else:
        path = _checkpoint_path(requested)
        if not path.is_file():
            alternative = Path(str(requested) + ".hdf5")
            if alternative.is_file():
                path = alternative
            else:
                return _load_legacy(cls, requested)
    if not path.is_file():
        raise FileNotFoundError(path)
    h5py, hdf5_io = _hdf5_dependencies()
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("pytreenet_format") != FORMAT:
            raise ValueError("This is not a PyTreeNet native TTN checkpoint.")
        if handle.attrs.get("pytreenet_format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported TTN checkpoint format version.")
        payload = hdf5_io.load_from_hdf5(handle, ignore_unknown=False)
    return _restore(cls, payload)
