"""Sparse block handoff from the existing HK driver, extended to U1xU1.

No HDF5, native pickle, or dense many-body conversion is used in the run.
The tiny startup test alone explicitly inspects dense local tensors.
"""

def _plain_metadata(value):
    """Only ordinary Python values/NumPy arrays may enter an MPI message."""
    import numpy as np
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays cannot be transferred as HK metadata.')
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_plain_metadata(k): _plain_metadata(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_plain_metadata(v) for v in value)
    raise TypeError(f'Unsupported HK metadata type: {type(value).__name__}')


def _pack_space(space):
    """Describe U(1) spaces explicitly, including nested fused-leg factors."""
    import numpy as np
    import lasyten as ct
    from pytreenet_lasyten.util import lasyten_backend as lb
    from pytreenet_lasyten.util.abelian_charges import symmetry_rank
    rank = symmetry_rank(space.symmetry)
    result = dict(charge_rank=rank, is_dual=bool(space.is_dual),
                  basis_perm=np.array(space.basis_perm, dtype=np.int64, copy=True))
    if isinstance(space, ct.AbelianLegPipe):
        result.update(kind='pipe', legs=[_pack_space(s) for s in space.legs],
                      combine_cstyle=bool(space.combine_cstyle))
    elif isinstance(space, ct.DirectSumSpace):
        result.update(kind='sum', spaces=[_pack_space(s) for s in space.spaces],
                      summand_labels=_plain_metadata(space.summand_labels))
    elif type(space) is ct.ElementarySpace:
        result.update(kind='elementary',
                      defining_sectors=np.array(space.defining_sectors, dtype=np.int64, copy=True),
                      multiplicities=np.array(space.multiplicities, dtype=np.int64, copy=True),
                      custom_basis_perm=(None if space._basis_perm is None else result['basis_perm']))
    else:
        raise TypeError(f'Unsupported HK leg type: {type(space).__name__}')
    return result


def _unpack_space(record):
    import numpy as np
    import lasyten as ct
    from pytreenet_lasyten.util import lasyten_backend as lb
    if record['kind'] == 'pipe':
        space = ct.AbelianLegPipe([_unpack_space(s) for s in record['legs']],
                                  is_dual=record['is_dual'], combine_cstyle=record['combine_cstyle'])
    elif record['kind'] == 'sum':
        space = ct.DirectSumSpace([_unpack_space(s) for s in record['spaces']],
                                  is_dual=record['is_dual'], summand_labels=record['summand_labels'])
    elif record['kind'] == 'elementary':
        from pytreenet_lasyten.util.abelian_charges import symmetry_for
        space = ct.ElementarySpace(symmetry_for(record.get('charge_rank', 1)), record['defining_sectors'],
                                   multiplicities=record['multiplicities'],
                                   is_dual=record['is_dual'], basis_perm=record['custom_basis_perm'])
    else:
        raise ValueError(f"Unknown transferred leg kind: {record['kind']}")
    np.testing.assert_array_equal(space.basis_perm, record['basis_perm'])
    return space


def _pack_native(tensor):
    """Copy allowed numerical blocks, never a full dense local/many-body tensor."""
    import numpy as np
    import lasyten as ct
    if isinstance(tensor, ct.ChargedTensor):
        bb = tensor.backend.block_backend
        return dict(kind='charged', invariant=_pack_native(tensor.invariant_part),
                    charged_state=np.array(bb.to_numpy(tensor.charged_state), copy=True))
    if type(tensor) is not ct.SymmetricTensor:
        raise TypeError(f'Unsupported HK native tensor: {type(tensor).__name__}')
    from lasyten.backends import AbelianBackendData
    if not isinstance(tensor.data, AbelianBackendData):
        raise TypeError('HK transport expects LASyTen AbelianBackendData.')
    bb = tensor.backend.block_backend
    from pytreenet_lasyten.util.abelian_charges import symmetry_rank
    return dict(kind='symmetric', charge_rank=symmetry_rank(tensor.symmetry),
                codomain=[_pack_space(s) for s in tensor.codomain.factors],
                domain=[_pack_space(s) for s in tensor.domain.factors],
                labels=list(tensor.labels),
                dtype=np.dtype(tensor.dtype.to_numpy_dtype()).str,
                block_inds=np.array(tensor.data.block_inds, dtype=np.int64, copy=True),
                blocks=[np.array(bb.to_numpy(b), copy=True) for b in tensor.data.blocks])


def _unpack_native(record):
    import numpy as np
    import lasyten as ct
    from lasyten.backends import AbelianBackendData, BlockInds
    from pytreenet_lasyten.util import lasyten_backend as lb
    if record['kind'] == 'charged':
        invariant = _unpack_native(record['invariant'])
        tensor = ct.ChargedTensor(invariant, record['charged_state'])
    elif record['kind'] == 'symmetric':
        from pytreenet_lasyten.util.abelian_charges import symmetry_for
        backend = ct.get_backend(symmetry_for(record.get('charge_rank', 1)), block_backend='numpy')
        dtype = ct.Dtype.from_numpy_dtype(np.dtype(record['dtype']))
        data = AbelianBackendData(
            dtype=dtype, device='cpu',
            blocks=[backend.block_backend.as_block(b, dtype=dtype, device='cpu') for b in record['blocks']],
            block_inds=BlockInds(record['block_inds']), is_sorted=False)
        tensor = ct.SymmetricTensor(data,
            codomain=[_unpack_space(s) for s in record['codomain']],
            domain=[_unpack_space(s) for s in record['domain']],
            backend=backend, labels=record['labels'])
    else:
        raise ValueError(f"Unknown transferred tensor kind: {record['kind']}")
    tensor.test_sanity()
    return tensor


def _pack_network(network):
    """Freeze raw tensor storage and lazy Node permutations without changing them."""
    records = []
    for identifier, node in network.nodes.items():
        tensor = network.tensors.data[identifier]
        records.append(dict(identifier=identifier, parent=node.parent,
                            children=list(node.children), permutation=list(node.leg_permutation),
                            tensor=_pack_native(tensor.native)))
    return dict(root_id=network.root_id, center=network.orthogonality_center_id,
                nodes=records, metadata={k: _plain_metadata(getattr(network, k))
                    for k in ('u1_info', 'u1_operator_info') if hasattr(network, k)})


def _unpack_network(cls, record):
    from pytreenet_lasyten.core.node import Node
    from pytreenet_lasyten.util import lasyten_backend as lb
    network = cls(backend='lasyten')
    for entry in record['nodes']:
        identifier = entry['identifier']
        tensor = lb.LasyTenArray(_unpack_native(entry['tensor']))
        node = Node(identifier=identifier, tensor=tensor)
        node.parent, node.children = entry['parent'], list(entry['children'])
        network._nodes[identifier] = node
        network._tensors[identifier] = tensor
        node._leg_permutation = list(entry['permutation'])
    network._root_id = record['root_id']
    network.orthogonality_center_id = record['center']
    for key, value in record['metadata'].items():
        setattr(network, key, value)
    return network


def _memory_handoff_smoke_test(comm, charge_symmetry='u1'):
    """Before DMRG, check the installed native block API and actual MPI handoff."""
    import numpy as np
    import lasyten as ct
    from pytreenet_lasyten.core.node import Node
    from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
    from pytreenet_lasyten.util import lasyten_backend as lb
    from pytreenet_lasyten.util.u1_excitations import create_impurity_excitations
    rank, size = comm.Get_rank(), comm.Get_size()
    if rank == 0:
        state = TTNS(backend='lasyten')
        if charge_symmetry == 'u1xu1':
            physical = lb.u1xu1_space([(0, 0), (1, 0)])
            down = lb.u1xu1_space([(0, 0), (0, 1)])
            occupied_down = lb.u1xu1_space([(0, 1)])
            data = np.zeros((2, 1, 2), dtype=complex)
            data[:, 0, :] = [[0., .8], [.6, 0.]]
            state.add_root(Node(identifier='r'), lb.from_numpy(data,
                spaces=[physical, occupied_down, physical], charge=lb.u1xu1_space([(1, 1)])))
            state.add_child_to_parent(Node(identifier='c'), lb.from_numpy(
                np.eye(2, dtype=complex), spaces=[physical.dual, physical]), 0, 'r', 0)
            state.add_child_to_parent(Node(identifier='d'), lb.from_numpy(
                np.array([[0., 1.]], dtype=complex), spaces=[occupied_down.dual, down]), 0, 'r', 1)
        else:
            physical = lb.u1_space([0, 1])
            state.add_root(Node(identifier='r'), lb.from_numpy(
                np.array([[0., .8], [.6, 0.]], dtype=complex), spaces=[physical, physical],
                charge=lb.u1_space([1])))
            state.add_child_to_parent(Node(identifier='c'), lb.from_numpy(
                np.eye(2, dtype=complex), spaces=[physical.dual, physical]), 0, 'r', 0)
        removal, addition, zr, za = create_impurity_excitations(
            state, impurity_node='c', parity_nodes=('r',), normalise=False)
        np.testing.assert_allclose([zr, za], [.36, .64], atol=1e-12, rtol=1e-12)
        for label, before in [('ground', state), ('removal', removal), ('addition', addition)]:
            if label != 'ground' and not any(isinstance(s, ct.AbelianLegPipe)
                    for node in before.nodes for s in before.tensors[node].legs):
                raise RuntimeError('The direct-handoff test must include a fused leg.')
            record = _pack_network(before)
            after = _unpack_network(TTNS, record)
            for identifier in before.nodes:
                left, right = before.tensors[identifier], after.tensors[identifier]
                np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), atol=1e-12, rtol=1e-12)
                if any(type(a) is not type(b) or a != b for a, b in zip(left.legs, right.legs)):
                    raise RuntimeError('Direct handoff changed native leg spaces.')
            norm = before.scalar_product(use_orthogonal_center=False)
            np.testing.assert_allclose(after.scalar_product(use_orthogonal_center=False), norm,
                                       atol=1e-12, rtol=1e-12)
            if size == 2:
                comm.send((record, norm), dest=1, tag=1700)
                if comm.recv(source=1, tag=1701) != label:
                    raise RuntimeError('Direct MPI smoke-test acknowledgement mismatch.')
        print('Direct native block/MPI handoff checks passed; no HDF5 used.', flush=True)
    elif size == 2:
        for label in ('ground', 'removal', 'addition'):
            record, norm = comm.recv(source=0, tag=1700)
            after = _unpack_network(TTNS, record)
            np.testing.assert_allclose(after.scalar_product(use_orthogonal_center=False), norm,
                                       atol=1e-12, rtol=1e-12)
            comm.send(label, dest=0, tag=1701)


def _handoff_to_particle_rank(comm, rank, cls, network, name):
    """Send one network at a time; discard temporary buffers before the next."""
    if rank == 0:
        record = _pack_network(network)
        comm.send((name, record), dest=1, tag=1710)
        del record
        if comm.recv(source=1, tag=1711) != name:
            raise RuntimeError(f'Direct MPI acknowledgement mismatch for {name}.')
        return network
    received_name, record = comm.recv(source=0, tag=1710)
    if received_name != name:
        raise RuntimeError(f'Direct MPI message mismatch: {received_name} != {name}.')
    result = _unpack_network(cls, record)
    del record
    comm.send(name, dest=0, tag=1711)
    return result
