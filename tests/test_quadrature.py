from deferred_correction.DeC import get_nodes
import numpy as np

def test_quadrature():
    nodes, weights = get_nodes(3, "gaussLegendre" )
    assert np.isclose(np.sum(nodes**4*weights), 1./5. )

if __name__=="__main__":
    test_quadrature()
