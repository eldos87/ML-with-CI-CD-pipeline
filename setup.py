from setuptools import setup, find_packages
from typing import List


def get_requirements_list() -> List[str]:
    """
    Description : This function is going to return all req in requirements.txt file
    return: A List which contains name of libraries mentioned in requirements.txt file
    """
    with open("requirements.txt") as file:
        reqt_list = file.readlines()
        reqt_list = [reqt.replace("\n", "") for reqt in reqt_list]
        return reqt_list


setup(
    name="housing project",
    version="0.0.3",         # need to increment version whenever update
    author="Eldos Thomas",
    description="house value prediction in CI/CD mode",
    packages=find_packages(),   # local packages created. eg: housing
    install_requires=get_requirements_list()    # external libraries required
)
