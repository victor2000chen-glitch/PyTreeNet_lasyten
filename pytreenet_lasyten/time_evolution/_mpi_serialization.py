"""Frozen task messages for NumPy and native LASyTen BUG workers.

Native tensors and their representation spaces go through LASyTen's HDF5
serializer, never pickle or a dense array. The outer MPI message contains
only bytes. NumPy-only jobs retain the ordinary pickle path. All messages
are trusted, same-user worker messages; they are not an interchange format
for untrusted inputs.
"""
from __future__ import annotations

import pickle

import numpy as np

from ..util import ttn_io
from ..util.tensor_dispatch import is_lasyten_tensor

WIRE_VERSION = 1


def _network_has_native(network):
    return any(is_lasyten_tensor(a) for a in network.tensors.data.values())


def _hdf5_dump(payload):
    h5py, hdf5_io = ttn_io._hdf5_dependencies()
    if not hasattr(h5py.File, "in_memory"):
        raise ImportError("Native MPI transport requires h5py >= 3.13.")
    with h5py.File.in_memory() as handle:
        hdf5_io.save_to_hdf5(handle, payload)
        handle.flush()
        return handle.id.get_file_image()


def _hdf5_load(blob):
    h5py, hdf5_io = ttn_io._hdf5_dependencies()
    if not hasattr(h5py.File, "in_memory"):
        raise ImportError("Native MPI transport requires h5py >= 3.13.")
    with h5py.File.in_memory(blob) as handle:
        return hdf5_io.load_from_hdf5(handle, ignore_unknown=False)


def _array_record(array):
    if isinstance(array, np.ndarray):
        return dict(kind="numpy", tensor=array)
    if is_lasyten_tensor(array):
        return dict(kind="lasyten", tensor=ttn_io._native_array(array))
    raise TypeError("MPI BUG environments must be NumPy or native LASyTen tensors.")


def _restore_array(record):
    if record["kind"] == "numpy" and isinstance(record["tensor"], np.ndarray):
        return record["tensor"]
    if record["kind"] == "lasyten":
        from ..util import lasyten_backend as lb
        if not isinstance(record["tensor"], lb._lasyten().Tensor):
            raise TypeError("MPI HDF5 loader did not restore a native tensor.")
        return lb.LasyTenArray(record["tensor"])
    raise ValueError("Invalid MPI tensor record.")


def _envelope(codec, blob, metadata=None):
    return dict(version=WIRE_VERSION, codec=codec, payload=blob, metadata=metadata)


def _check_envelope(message):
    if (not isinstance(message, dict) or message.get("version") != WIRE_VERSION
            or message.get("codec") not in ("pickle", "hdf5")
            or not isinstance(message.get("payload"), bytes)):
        raise ValueError("Unsupported MPI BUG message.")


def encode_task(node_id, new_state, parent_state, environments, hamiltonian,
                time_step_size, bug_config, solver_options):
    """Freeze a task before asynchronous submission can overlap other updates."""
    native = (any(_network_has_native(t) for t in (new_state, parent_state, hamiltonian))
              or any(is_lasyten_tensor(a) for a in environments.values()))
    if not native:
        job = (node_id, new_state, parent_state, dict(environments), hamiltonian,
               time_step_size, bug_config, solver_options)
        return _envelope("pickle", pickle.dumps(job, protocol=5))
    # Config/solver options are small Python objects. Native coefficient data
    # and Node space caches are deliberately absent from their pickle.
    metadata = pickle.dumps((node_id, time_step_size, bug_config, solver_options), protocol=5)
    body = dict(new_state=ttn_io._payload(new_state),
                parent_state=ttn_io._payload(parent_state),
                hamiltonian=ttn_io._payload(hamiltonian),
                environments=[dict(source=a, target=b, array=_array_record(tensor))
                              for (a, b), tensor in environments.items()])
    return _envelope("hdf5", _hdf5_dump(body), metadata)


def decode_task(message):
    _check_envelope(message)
    if message["codec"] == "pickle":
        return pickle.loads(message["payload"])
    from ..ttns.ttns import TreeTensorNetworkState
    from ..ttno.ttno_class import TreeTensorNetworkOperator
    node_id, time_step_size, bug_config, solver_options = pickle.loads(message["metadata"])
    body = _hdf5_load(message["payload"])
    new_state = ttn_io._restore(TreeTensorNetworkState, body["new_state"])
    parent_state = ttn_io._restore(TreeTensorNetworkState, body["parent_state"])
    hamiltonian = ttn_io._restore(TreeTensorNetworkOperator, body["hamiltonian"])
    cache = {(r["source"], r["target"]): _restore_array(r["array"])
             for r in body["environments"]}
    return (node_id, new_state, parent_state, cache, hamiltonian,
            time_step_size, bug_config, solver_options)


def encode_result(result):
    native = (any(is_lasyten_tensor(t) for _, t in result.node_tensors.values())
              or is_lasyten_tensor(result.environment)
              or is_lasyten_tensor(result.basis_change))
    if not native:
        return _envelope("pickle", pickle.dumps(result, protocol=5))
    records = []
    for key, (node, tensor) in result.node_tensors.items():
        records.append(dict(identifier=key, parent=node.parent,
                            children=list(node.children),
                            permutation=list(node.leg_permutation),
                            array=_array_record(tensor)))
    body = dict(child_id=result.child_id, parent_id=result.parent_id, nodes=records,
                environment=_array_record(result.environment),
                basis_change=_array_record(result.basis_change))
    return _envelope("hdf5", _hdf5_dump(body))


def decode_result(message):
    _check_envelope(message)
    if message["codec"] == "pickle":
        return pickle.loads(message["payload"])
    from ..core.node import Node
    from .mpi_bug import _SubtreeUpdate
    body = _hdf5_load(message["payload"])
    nodes = {}
    for record in body["nodes"]:
        tensor = _restore_array(record["array"])
        node = Node(identifier=record["identifier"], tensor=tensor)
        node.parent = record["parent"]
        node.children = list(record["children"])
        node._leg_permutation = list(record["permutation"])
        nodes[node.identifier] = (node, tensor)
    return _SubtreeUpdate(body["child_id"], body["parent_id"], nodes,
                          _restore_array(body["environment"]),
                          _restore_array(body["basis_change"]))
