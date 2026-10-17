import numpy as np

from dreamteam import optim


def test_serpentine_is_a_permutation(rng):
    pts = rng.uniform(size=(500, 2))
    order = optim.serpentine_order(pts)
    assert sorted(order) == list(range(500))


def test_two_opt_never_worse_and_keeps_ends(rng):
    pts = rng.uniform(size=(80, 2))
    start = np.arange(80)
    improved = optim.two_opt(pts, start, fixed_ends=True)
    assert optim.route_length(pts, improved) <= optim.route_length(pts, start)
    assert improved[0] == 0 and improved[-1] == 79


def test_balanced_kmeans_respects_capacity(rng):
    pts = np.vstack([rng.normal(0, 0.1, (90, 2)), rng.normal(5, 0.1, (10, 2))])  # clúster gran i petit
    labels = optim.balanced_kmeans_labels(pts, k=4)
    assert np.bincount(labels).max() <= 25


def test_nearest_neighbour_visits_all(rng):
    pts = rng.uniform(size=(50, 2))
    assert sorted(optim.nearest_neighbour(pts)) == list(range(50))
