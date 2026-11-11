# Charge symmetries in PyTreeNet-LASyTen

LASyTen labels each tensor leg with an additive charge. It stores only the
blocks allowed by the charge rule, so contractions and decompositions can work
on those blocks directly. The
[`lasyten_backend` adapter](../pytreenet_lasyten/util/lasyten_backend.py)
exposes this through the usual PyTreeNet tensor and tree interfaces.

| Setting | Conserved charge | Empty / occupied spin-up mode | Empty / occupied spin-down mode |
| --- | --- | --- | --- |
| `symmetry="u1"` | Total particle number $N$ | $0,1$ | $0,1$ |
| `symmetry="u1xu1"` | $(N_\uparrow,N_\downarrow)$ | $(0,0),(1,0)$ | $(0,0),(0,1)$ |

Use `"u1xu1"` when the Hamiltonian conserves both spin populations, as in the
Hubbard–Kanamori (HK) model with the spin-conserving bath coupling used here.
A term that flips one spin can conserve $N$ while changing the two spin
populations; that Hamiltonian is not compatible with `"u1xu1"`. The charges
use bosonic representations. **Fermionic signs come from the Jordan–Wigner
strings in the model**, which depend on the ordering of the physical modes.

## The charge rule on a tree

A leg space lists the charge of every basis vector. Repeating a label gives
that sector more basis vectors. A bond's dimension is the **sum** of these
sector multiplicities, so a bond cap applies to the total dimension.

A dual leg contributes the opposite charge in the block selection rule.
Contract two symmetric legs only when their spaces are dual to each other:
the same dimension alone is insufficient. Use `space.dual` to get the
opposite representation.

In the occupation-tree constructors, a virtual bond labels the total charge
in the descendant subtree. The parent leg of a non-root state tensor is
dual, while its child legs and physical leg are nondual. Its allowed blocks
therefore satisfy

$$
q_{\mathrm{parent}} = q_{\mathrm{phys}} + \sum_c q_c.
$$

With the orthogonality center at the root, the root carries the fixed total
charge in a LASyTen `ChargedTensor`; the other state tensors are invariant
`SymmetricTensor` objects. This is how the network represents a fixed-$N$
state, or a fixed-$(N_\uparrow,N_\downarrow)$ state, without storing
coefficients from the other sectors.

## Create and contract individual tensors

Use one leg space per array axis, with charges in the input basis order. An
operator has a nondual output leg and a dual input leg. Omitting `charge`
creates a neutral tensor with the supplied spaces. For a nonzero net charge,
pass a one-dimensional charge space explicitly:

```python
import numpy as np
from pytreenet_lasyten.util import lasyten_backend as lb

# A U(1)-invariant number operator in the basis |0>, |1>.
p = lb.u1_space([0, 1])
n = lb.from_numpy(np.diag([0.0, 1.0]), spaces=[p, p.dual])
n_squared = lb.tensordot(n, n, axes=([1], [0]))
np.testing.assert_allclose(n_squared.to_numpy(), np.diag([0.0, 1.0]))

# |1_up, 0_down> with fixed (N_up, N_down) = (1, 0).
up = lb.u1xu1_space([(0, 0), (1, 0)])
down = lb.u1xu1_space([(0, 0), (0, 1)])
pair = lb.from_numpy(
    np.array([[0.0, 0.0], [1.0, 0.0]]),
    spaces=[up, down],
    charge=lb.u1xu1_space([(1, 0)]),
)
print(pair.shape, type(pair.native).__name__)  # (2, 2) ChargedTensor
```

`lb.from_numpy` without `spaces` produces a tensor with no charge symmetry.
With explicit spaces, it checks that the input respects the declared charge
rule. Use adapter operations such as `lb.tensordot`, `lb.transpose`, and
`lb.vdot` for native calculations. `tensor.to_numpy()` explicitly expands a
tensor to a dense array; reserve it for small checks.

## Build a fixed-charge tree and access its tensors

The [HK constructor](../pytreenet_lasyten/util/hk_u1_model.py) builds a random
state in the requested sector. It uses one occupation mode per physical site
and charge-zero, dimension-one legs at the branching nodes.

```python
from pytreenet_lasyten.util.abelian_charges import fixed_charge
from pytreenet_lasyten.util.hk_u1_model import build_u1_t3ns

state = build_u1_t3ns(
    norb=3, nb=3, virtual_bond_dimension=8, bath_bond_dimension=8,
    symmetry="u1xu1", total_particles_by_spin=(6, 6), seed=11,
)
assert fixed_charge(state) == (6, 6)

site = "Impurity_1_up"
node, tensor = state[site]                 # Or just state.tensors[site].
print(tensor.shape)
print(tensor.legs[-1].sectors_of_basis)    # Physical charges: (0,0), (1,0).

parent, parent_tensor = state[node.parent]
axis_on_parent = parent.neighbour_index(site)
axis_on_child = node.neighbour_index(parent.identifier)
assert parent_tensor.legs[axis_on_parent] == tensor.legs[axis_on_child].dual
```

For one charge, change the constructor arguments to `symmetry="u1",
total_particles=12` and omit `total_particles_by_spin`. `fixed_charge(state)`
returns `(12,)`. If the charged center has moved, first call
`state.ensure_root_orth_center(mode=SplitMode.REDUCED)` with `SplitMode`
imported from `pytreenet_lasyten.util.tensor_splitting`.

The tensor accessor returns a `LasyTenArray` in the public axis order:
parent if present, children, then physical leg. For a TTNO, the last two axes
are physical output and input. Use `tensor.legs` for the current spaces;
initialization metadata can differ from the bond spaces after truncation.
For a custom occupation tree, use
[`random_u1_state_like`](../pytreenet_lasyten/util/u1_t3ns.py) or
[`random_u1u1_state_like`](../pytreenet_lasyten/util/u1u1_t3ns.py);
the latter accepts `spin_by_node` when node names do not encode the spin.

## Build and apply a charge-conserving operator

The public TTNO constructor dispatches to the appropriate symmetry from the
reference tree. The operator has net charge zero; it does not inherit the
state's fixed particle number.

```python
from pytreenet_lasyten.operators import Hamiltonian, TensorProduct
from pytreenet_lasyten.ttns.ttns_ttno.application import (
    ApplicationMethod, apply_ttno_to_ttns,
)
from pytreenet_lasyten.util.abelian_charges import check_native_charges
from pytreenet_lasyten.util.u1_ttno import u1_ttno_from_hamiltonian

# Example: measure occupation of the site selected above.
symbolic = Hamiltonian(
    [TensorProduct({site: "n"})],
    conversion_dictionary={"n": np.diag([0.0, 1.0])},
)
operator = u1_ttno_from_hamiltonian(symbolic, state)
assert check_native_charges(state, operator) == (6, 6)
n_state = apply_ttno_to_ttns(state, operator, ApplicationMethod.DIRECT)
print(state.operator_expectation_value(operator))
```

`check_native_charges` returns the charge tuple and checks the actual bond
and physical spaces. The older `check_native_u1` helper returns their
particle-number sum. To create particle or hole states from a ground state,
use [`create_impurity_excitations`](../pytreenet_lasyten/util/u1_excitations.py)
with the preceding modes in your Jordan–Wigner ordering as `parity_nodes`.
For a spin-up site the two returned states carry $(N_\uparrow-1,N_\downarrow)$
and $(N_\uparrow+1,N_\downarrow)$; the Hamiltonian can evolve them separately.

Run `python -m examples.native_symmetry_check` from the backend project
directory in a LASyTen environment to check the native operations against a
small exact reference. See [MPI BUG](mpi.md) for parallel evolution.
