"""Independent four-mode number-conserving model for patch validation."""
from fractions import Fraction
from functools import reduce
import numpy as np

ORDER = ('Impurity_1_up', 'Impurity_1_down', 'Impurity_2_up', 'Impurity_2_down')
ANN = np.array([[0., 1.], [0., 0.]], dtype=complex)
ID = np.eye(2, dtype=complex)
Z = np.diag([1., -1.]).astype(complex)


def make_template():
    from pytreenet_lasyten.ttns.ttns import TreeTensorNetworkState as TTNS
    from pytreenet_lasyten.core.node import Node
    tree=TTNS(backend='numpy')
    tree.add_root(Node(identifier='node1'),np.ones((1,1,1),dtype=complex))
    tree.add_child_to_parent(Node(identifier=ORDER[0]),np.ones((1,1,2),dtype=complex),0,'node1',0)
    tree.add_child_to_parent(Node(identifier=ORDER[2]),np.ones((1,2),dtype=complex),0,ORDER[0],1)
    tree.add_child_to_parent(Node(identifier=ORDER[1]),np.ones((1,1,2),dtype=complex),0,'node1',1)
    tree.add_child_to_parent(Node(identifier=ORDER[3]),np.ones((1,2),dtype=complex),0,ORDER[1],1)
    return tree


def fermion_matrix(index, dagger=False):
    factors=[Z if j<index else (ANN.conj().T if dagger else ANN) if j==index else ID for j in range(4)]
    return reduce(np.kron,factors)


def make_hamiltonian():
    from pytreenet_lasyten.operators import Hamiltonian,TensorProduct
    # Spin-conserving hopping, density interactions, spin exchange and pair hopping.
    specifications=[(-.31,[(0,True),(2,False)]),(-.31,[(2,True),(0,False)]),
                    (.17j,[(1,True),(3,False)]),(-.17j,[(3,True),(1,False)]),
                    (.2,[(0,True),(0,False)]),(-.13,[(1,True),(1,False)]),
                    (1.1,[(0,True),(0,False),(1,True),(1,False)]),
                    (.8,[(2,True),(2,False),(3,True),(3,False)]),
                    (.37,[(0,True),(0,False),(3,True),(3,False)]),
                    (-.19,[(0,True),(1,False),(3,True),(2,False)]),
                    (-.19,[(2,True),(3,False),(1,True),(0,False)]),
                    (.23,[(0,True),(1,True),(3,False),(2,False)]),
                    (.23,[(2,True),(3,True),(1,False),(0,False)])]
    terms=[];conversion={'I1':np.eye(1,dtype=complex),'I2':ID};coefficients={};dense=np.zeros((16,16),complex)
    for k,(coefficient,sequence) in enumerate(specifications):
        local=[ID.copy() for _ in ORDER]
        exact=np.eye(16,dtype=complex)
        for index,dagger in sequence:
            exact=exact@fermion_matrix(index,dagger)
            for j in range(index):local[j]=local[j]@Z
            local[index]=local[index]@(ANN.conj().T if dagger else ANN)
        operators={}
        for j,matrix in enumerate(local):
            if np.array_equal(matrix,ID):continue
            label=f'op{k}_{j}';conversion[label]=matrix;operators[ORDER[j]]=label
        gamma=f'g{k}';coefficients[gamma]=coefficient
        terms.append((Fraction(1),gamma,TensorProduct(operators)))
        dense+=coefficient*exact
    np.testing.assert_allclose(dense,dense.conj().T,atol=1e-14)
    return Hamiltonian(terms,conversion_dictionary=conversion,coeffs_mapping=coefficients),dense


def native_vector(state):
    from pytreenet_lasyten.util.u1_t3ns import dense_state_for_checks
    data,order=dense_state_for_checks(state,max_physical_sites=4)
    return data.transpose([order.index(i) for i in ORDER]).reshape(-1)


def dense_operator(operator):
    """Contract plain NumPy TTNO site arrays; preserve physical output/input order."""
    order=operator.linearise()
    accumulated={}
    for i in order:
        node=operator.nodes[i];a=np.asarray(operator.tensors[i])
        labels=([('bond',node.parent)] if node.parent is not None else [])
        labels += [('bond',c) for c in node.children]+[('out',i),('in',i)]
        for child in node.children:
            b,other=accumulated.pop(child)
            axis=labels.index(('bond',child))
            a=np.tensordot(a,b,axes=([axis],[0]))
            labels=labels[:axis]+labels[axis+1:]+other[1:]
        accumulated[i]=a,labels
    a,labels=accumulated[operator.root_id]
    wanted=[('out',i) for i in ORDER]+[('in',i) for i in ORDER]
    # Dummy physical legs have dimension one and can be placed last.
    axes=[labels.index(label) for label in wanted]+[j for j,label in enumerate(labels) if label not in wanted]
    return a.transpose(axes).reshape(16,16)
