"""CPU NumPy/U1/U1xU1 block records for trusted subtree worker messages."""
from functools import lru_cache

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
    rank = _symmetry_rank(space.symmetry)
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
        space = ct.ElementarySpace(_symmetry_for(record.get('charge_rank', 1)), record['defining_sectors'],
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
    return dict(kind='symmetric', charge_rank=_symmetry_rank(tensor.symmetry),
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
        backend = ct.get_backend(_symmetry_for(record.get('charge_rank', 1)), block_backend='numpy')
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



def _symmetry_rank(symmetry):
    for rank in (1,2):
        if symmetry.is_equivalent_to(_symmetry_for(rank)):
            return rank
    raise TypeError('Native subtree transport requires bosonic U1 or U1xU1.')


@lru_cache(maxsize=2)
def _symmetry_for(rank):
    import lasyten as ct
    from ..util import lasyten_backend as lb
    if rank not in (1, 2):
        raise ValueError('Unsupported charge rank.')
    return lb._symmetry(ct.U1() if rank == 1 else ct.U1()*ct.U1())


def pack_array(array):
    import numpy as np
    from ..util.tensor_dispatch import is_lasyten_tensor
    if isinstance(array, np.ndarray):
        if array.dtype.hasobject:
            raise TypeError('Object arrays are not supported by the worker codec.')
        return dict(kind='numpy', data=np.array(array, copy=True))
    if is_lasyten_tensor(array):
        if str(array.native.device) not in ('cpu', 'cpu:0'):
            raise ValueError('This worker codec supports CPU tensors only.')
        return dict(kind='native', data=_pack_native(array.native))
    raise TypeError(f'Unsupported worker array: {type(array).__name__}')


def unpack_array(record):
    if record['kind']=='numpy':
        return record['data']
    if record['kind']=='native':
        from ..util.lasyten_backend import LasyTenArray
        return LasyTenArray(_unpack_native(record['data']))
    raise ValueError('Invalid worker array record.')


def node_record(node, tensor=None):
    # Never pickle cached native leg-space objects on a Node.
    result=dict(identifier=node.identifier,parent=node.parent,children=list(node.children),
                permutation=list(node.leg_permutation),shape=tuple(node._shape))
    if tensor is not None:
        result['tensor']=pack_array(tensor)
    else:
        result['spaces']=(None if node._leg_spaces is None else
                          [_pack_space(s) for s in node._leg_spaces])
    return result


def restore_node(record,tensor_cache=None,cache_token=None):
    from ..core.node import Node
    tensor=None
    if 'tensor' in record:
        key=(cache_token,record['identifier'],tuple(record['permutation']),tuple(record['shape']))
        tensor=None if tensor_cache is None else tensor_cache.get(key)
        if tensor is None:
            tensor=unpack_array(record['tensor'])
            if tensor_cache is not None:
                try:tensor_cache[key]=tensor
                except TypeError:pass  # A custom array may not support weakrefs.
    node=Node(identifier=record['identifier'],tensor=tensor)
    if tensor is None:
        node._shape=tuple(record['shape'])
        node._leg_spaces=(None if record['spaces'] is None else
                          tuple(_unpack_space(s) for s in record['spaces']))
    node.parent=record['parent']
    node.children=list(record['children'])
    node._leg_permutation=list(record['permutation'])
    return node,tensor


def pack_view(network, tensor_ids, metadata_ids):
    """Private worker view: keep the subtree and its gauge boundary, not a TTN checkpoint."""
    raw=network.tensors.data
    records=[node_record(node,raw[key] if key in tensor_ids else None)
             for key,node in network.nodes.items() if key in metadata_ids]
    return dict(root_id=network.root_id,center=network.orthogonality_center_id,
                records=records,backend=('lasyten' if any(r.get('tensor',{}).get('kind')=='native'
                    for r in records) else 'numpy'))


def restore_view(cls, record,tensor_cache=None,cache_token=None):
    network=cls(backend=record['backend'])
    for item in record['records']:
        node,tensor=restore_node(item,tensor_cache,cache_token)
        network._nodes[node.identifier]=node
        if tensor is not None:
            network._tensors.data[node.identifier]=tensor
    network._root_id=record['root_id']
    network.orthogonality_center_id=record['center']
    return network

