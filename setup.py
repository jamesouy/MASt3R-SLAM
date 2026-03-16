from pathlib import Path
from setuptools import setup

backends = Path(__file__).parent / "mast3r_slam" / "backend" / "src"

setup(
    install_requires=[
        "numpy>=1.26.4",
        "einops",
        "pyrealsense2",
        "evo",
        "natsort",
        "torchcodec",
        "lietorch @ git+https://github.com/princeton-vl/lietorch.git",
        "plyfile",
		f"mast3r_slam_backends @ {backends.as_uri()}"
    ],
)
