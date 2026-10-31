import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'netra_sim'


def tree(src, dst):
    """(install_dir, [files]) pairs that keep the directory layout of `src`."""
    out = {}
    for path in glob(os.path.join(src, '**', '*'), recursive=True):
        if os.path.isfile(path):
            out.setdefault(os.path.join('share', package_name, dst, os.path.relpath(os.path.dirname(path), src)), []).append(path)
    return sorted(out.items())


setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ] + tree('models', 'models'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NETRA-UGV Team',
    maintainer_email='team@netra-ugv.in',
    description='Gazebo Harmonic and CARLA simulation backends for NETRA-UGV',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'depth_to_disparity = netra_sim.depth_to_disparity:main',
            'mission_manager = netra_sim.mission_manager:main',
            'carla_bridge = netra_sim.carla_bridge:main',
            'carla_sidecar = netra_sim.carla_sidecar:main',
            'odom_relay = netra_sim.odom_relay:main',
            'smoke_test = netra_sim.smoke_test:main',
            'gen_gazebo_assets = netra_sim.gen_gazebo_assets:main',
        ],
    },
)
