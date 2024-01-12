from setuptools import find_packages, setup

requirements = [
    "numpy",
    "pytest",
    "xarray",
    "scikit-learn",
    "scipy",
"matplotlib",
"pandas",
"numpy",
"seaborn",
#"torch",
"transformers",
"tqdm",
"wandb",
"accelerate",
"datasets"
]

setup(
    name='sent_sampling',
    version='0.0.1',
    packages=find_packages(),
    install_requires=requirements,
    description="sent_sampling",
    author="Eghbal Hosseini",
    author_email='ehoseini@mit.edu',
    license="",
)