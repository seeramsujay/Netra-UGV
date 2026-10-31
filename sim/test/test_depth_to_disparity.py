import numpy as np

from netra_sim.depth_to_disparity import depth_to_disparity


def test_matches_stereo_geometry_with_stack_defaults():
    # fx = 500 px, baseline = 0.075 m  ->  1 m away is 37.5 px of disparity
    d = depth_to_disparity([[1.0, 5.0]], fx=500.0, baseline=0.075, min_depth=0.3, max_depth=20.0)
    np.testing.assert_allclose(d, [[37.5, 7.5]], rtol=1e-6)


def test_invalid_pixels_become_zero_holes():
    depth = [[np.inf, -np.inf, np.nan, 0.0, 0.1, 25.0, 2.0]]
    d = depth_to_disparity(depth, 500.0, 0.075, 0.3, 20.0)
    np.testing.assert_array_equal(d[0, :6], 0.0)
    assert d[0, 6] > 0
    assert d.dtype == np.float32
