"""The committed Gazebo world/terrain must be exactly what scenario.py generates."""
import os

from netra_sim import gen_gazebo_assets as gen
from netra_sim import scenario as sc

SIM = gen.SIM_DIR


def test_world_is_in_sync_with_scenario():
    grid, rocks = sc.height_grid(), sc.build_rocks()
    assert open(os.path.join(SIM, 'worlds', 'tactical_field.sdf')).read() == gen.build_world(grid, rocks)


def test_terrain_mesh_is_in_sync_with_scenario(tmp_path):
    fresh = tmp_path / 'terrain.obj'
    gen.write_terrain_obj(str(fresh), sc.height_grid())
    assert fresh.read_text() == open(os.path.join(SIM, 'models', 'netra_terrain', 'meshes', 'terrain.obj')).read()
