"""Test the masked extension"""

import mdp


def get_data():
    data = mdp.numx.ma.array(mdp.numx.random.random((30, 10)), mask=False)
    for i,j in [(2,1), (3,3), (12,0), (20,0)]:
        data.mask[i,j] = True
    return data


def test_masked_extension():
    """Test that the masked extension is working at the global level"""
    data = get_data()

    # no extension -- what are we expecting here?
    node = mdp.nodes.PCANode()
    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.CovarianceMatrix)
    factors = node(data)

    node = mdp.nodes.PCANode()
    # activate the extension
    with mdp.extension('masked'):
        assert isinstance(node._new_covariance_matrix(),
                          mdp.utils.MaskedCovarianceMatrix)
        factors = node(data)

