import os
from glob import glob

from setuptools import find_packages, setup

package_name = "netra_planning"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NETRA-UGV Team",
    maintainer_email="team@netra-ugv.in",
    description="Kinodynamic TEB trajectory planner and safety guards for NETRA-UGV",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "planner_node = netra_planning.planner_node:main",
            "failsafe_manager = netra_planning.failsafe_manager:main",
        ],
    },
)
