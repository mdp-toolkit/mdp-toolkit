"""Test the masked extension"""

import mdp


def get_data():
    data = mdp.numx.ma.array(mdp.numx.random.random((30, 10)), mask=False)
    for i,j in [(2,1), (3,3), (12,0), (20,0)]:
        data.mask[i,j] = True
    return data


def test_masked_extension():
    """Test that the masked extension is working at the global level"""

    node = mdp.nodes.PCANode()
    data = get_data()

    # activate the extension
    mdp.masked.activate_masked()
    assert mdp.get_active_extensions() == ['masked']

    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.MaskedCovarianceMatrix)
    factors = node(data)

    # after deactivation
    mdp.masked.deactivate_masked()
    assert mdp.get_active_extensions() == []
    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.CovarianceMatrix)


def test_class_masked():
    """Test the masked extension for individual classes"""
    masked = mdp.nodes.PCANode()
    notmasked = mdp.nodes.SFANode()
    mdp.masked.activate_masked(masked_classes=[mdp.nodes.PCANode])
    assert masked.is_masked()
    assert not notmasked.is_masked()
    mdp.masked.deactivate_masked()


def test_class_masked_functionality():
    """Test that masked extension classes really handle masked data"""
    data = get_data()

    # the node is not masked
    node = mdp.nodes.PCANode()
    mdp.masked.activate_masked(masked_classes=[mdp.nodes.SFANode])
    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.CovarianceMatrix)
    factors = node(data)
    mdp.masked.deactivate_masked()

    # the node is masked
    node = mdp.nodes.PCANode()
    mdp.masked.activate_masked(masked_classes=[mdp.nodes.PCANode])
    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.MaskedCovarianceMatrix)
    factors = node(data)
    mdp.masked.deactivate_masked()


def test_instance_masked():
    """Test the masked extension for individual instances"""
    masked = mdp.nodes.PCANode()
    notmasked = mdp.nodes.PCANode()
    mdp.masked.activate_masked(masked_instances=[masked])
    assert masked.is_masked()
    assert not notmasked.is_masked()
    mdp.masked.deactivate_masked()


def test_instance_masked_functionality():
    """Test that masked extension instances really handle masked data"""
    data = get_data()

    # the node is not masked
    node = mdp.nodes.PCANode()
    othernode = mdp.nodes.PCANode()
    mdp.masked.activate_masked(masked_instances=[othernode])
    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.CovarianceMatrix)
    factors = node(data)
    mdp.masked.deactivate_masked()

    # the node is masked
    node = mdp.nodes.PCANode()
    mdp.masked.activate_masked(masked_instances=[node])
    assert isinstance(node._new_covariance_matrix(),
                      mdp.utils.MaskedCovarianceMatrix)
    factors = node(data)
    mdp.masked.deactivate_masked()
