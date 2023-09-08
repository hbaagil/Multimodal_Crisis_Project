from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='crisis_helper',
      version="0.0.1",
      description="crisis_helper (api_pred)",
      license="MIT",
      author="Le Wagon",
      author_email="contact@lewagon.org",
      #url="https://github.com/hbaagil/00_Multimodal_Crisis_Project",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
