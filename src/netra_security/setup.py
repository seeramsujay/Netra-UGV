import os
from glob import glob

from setuptools import find_packages, setup

package_name = "netra_security"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "sros2_policies"), glob("sros2_policies/*.xml")),
        (
            os.path.join("share", package_name, "sros2_policies/enclaves/netra_perception"),
            glob("sros2_policies/enclaves/netra_perception/*.xml"),
        ),
        (
            os.path.join("share", package_name, "sros2_policies/enclaves/netra_localization"),
            glob("sros2_policies/enclaves/netra_localization/*.xml"),
        ),
        (
            os.path.join("share", package_name, "sros2_policies/enclaves/netra_planning"),
            glob("sros2_policies/enclaves/netra_planning/*.xml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NETRA-UGV Team",
    maintainer_email="team@netra-ugv.in",
    description="S-ROS 2 security policies, SecOC CAN-FD driver & tamper zeroization",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "secoc_can_driver = netra_security.secoc_can_driver:main",
            "tamper_monitor = netra_security.tamper_monitor:main",
        ],
    },
)
