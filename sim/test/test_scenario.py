import math

from netra_sim import scenario as sc


def test_grid_dimensions_and_extent():
    assert len(sc.height_grid()) == sc.NX * sc.NY
    assert (sc.X1 - sc.X0, sc.Y1 - sc.Y0) == (50.0, 36.0)


def test_rocks_are_deterministic_and_clear_of_hazards():
    rocks = sc.build_rocks()
    assert rocks == sc.build_rocks()
    assert len(rocks) == sc.ROCK_COUNT
    for r in rocks:
        assert not sc.in_ditch(r.x, r.y, pad=1.0)
        assert math.hypot(r.x - sc.START[0], r.y - sc.START[1]) >= 5
        assert math.hypot(r.x - sc.GOAL[0], r.y - sc.GOAL[1]) >= 3


def test_ditch_floor_is_a_real_drop():
    for cx, cy, _, _ in sc.DITCHES:
        inside = sc.terrain_height(cx, cy)
        outside_on_flat = sc.terrain_height(cx, cy, ditches=False)
        assert math.isclose(outside_on_flat - inside, sc.DITCH_DEPTH)


def test_start_area_is_flat_and_goal_reachable():
    assert abs(sc.terrain_height(*sc.START[:2])) < 0.05
    assert not sc.in_ditch(*sc.GOAL)


def test_waypoints_avoid_ditches_and_are_integers():
    for wx, wy in sc.WAYPOINTS:
        assert isinstance(wx, int) and isinstance(wy, int)  # Webots HMAC-signs "%d,%d"
        assert not sc.in_ditch(wx, wy, pad=0.5)
    assert tuple(map(float, sc.WAYPOINTS[-1])) == sc.GOAL


def test_ground_z_matches_grid_at_vertices_and_interpolates():
    grid = sc.height_grid()
    i, j = 30, 40
    x, y = sc.X0 + i * sc.SP, sc.Y0 + j * sc.SP
    assert math.isclose(sc.ground_z(x, y, grid), grid[j * sc.NX + i])
    mid = sc.ground_z(x + sc.SP / 2, y, grid)
    lo, hi = sorted((grid[j * sc.NX + i], grid[j * sc.NX + i + 1]))
    assert lo - 1e-9 <= mid <= hi + 1e-9
