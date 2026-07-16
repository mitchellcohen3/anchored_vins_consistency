from setuptools import setup, find_packages

setup(
    name="pyvins",
    version="0.0.1",
    description="A collection of tools for launching visual-inertial navigation algorithms with various configurations.",
    packages=find_packages(),
    extras_require={"test": ["pytest"]},
    install_requires=[
        "pandas",
        "evo==1.31.0",
        "Jinja2",
        "navlie @ git+https://github.com/decargroup/navlie.git@79c46466d09a69adfb4f879eb8f6ac531a059ec9",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
