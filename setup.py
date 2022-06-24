###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from setuptools import find_packages, setup

setup(
    name="dsg_rl",
    description="Utilities to train RL policies using DSGs",
    packages=find_packages("src"),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    python_requires=">=3.7",
    package_dir={"": "src"},
    install_requires=[
        "ray[tune,rllib]>=1.6,<=1.10",
        "opencv-python",
        "numba",
        "pandas",
        "torch <= 1.10",
        "torchvision <= 0.11.2",
        "tesse_gym@git+ssh://git@github.com:MIT-TESSE/tesse-gym.git@master#egg=tesse_gym",
        "rllib_policies@git+ssh://git@github.com:MIT-TESSE/rllib-policies.git@master#egg=rllib_policies",
        "gym<=0.21",
    ],
    dependency_links=[
        "git+ssh://git@github.com:MIT-TESSE/tesse-gym.git@master#egg=tesse_gym",
        "git+ssh://git@github.com:MIT-TESSE/rllib-policies.git@master#egg=rllib_policies",
    ],
)
