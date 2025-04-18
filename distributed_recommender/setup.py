# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3

import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent.resolve()


def _get_version():
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(cmd, cwd=str(ROOT_DIR)).decode("ascii").strip()
    except Exception:
        sha = None

    if "BUILD_VERSION" in os.environ:
        version = os.environ["BUILD_VERSION"]
    else:
        with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
            version = f.readline().strip()
        if sha is not None and "OFFICIAL_RELEASE" not in os.environ:
            version += "+" + sha[:7]

    if sha is None:
        sha = "Unknown"
    return version, sha


def _export_version(version, sha):
    version_path = ROOT_DIR / "distributed_recommender" / "version.py"
    with open(version_path, "w") as fileobj:
        fileobj.write("__version__ = '{}'\n".format(version))
        fileobj.write("git_version = {}\n".format(repr(sha)))


def main() -> None:
    with open(
        os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf8"
    ) as f:
        readme = f.read()
    with open(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        encoding="utf8",
    ) as f:
        reqs = f.read()
        install_requires = reqs.strip().split("\n")

    version, sha = _get_version()
    _export_version(version, sha)

    print(f"-- distributed_recommender building version: {version}")

    packages = find_packages(
        exclude=(
            "*tests",
            "examples",
            "*examples.*",
            "*build",
            "*rfc",
        )
    )

    setup(
        # Metadata
        name="distributed_recommender",
        version=version,
        author="NVIDIA Corporation.",
        maintainer="zehuanw",
        maintainer_email="zehuanw@nvidia.com",
        description="distributed-recommender: Pytorch library for large scale generative recommendation systems",
        long_description=readme,
        license="BSD-3",
        keywords=[
            "pytorch",
            "recommendation systems",
            "distributed training",
        ],
        python_requires=">=3.9",
        install_requires=install_requires,
        packages=packages,
        zip_safe=False,
    )


if __name__ == "__main__":
    main() 