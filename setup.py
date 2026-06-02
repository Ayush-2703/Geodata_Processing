from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name             = "geodata-processing-ai",
    version          = "1.0.0",
    author           = "Ayush Kumar Singh",
    author_email     = "ab49ayush@gmail.com",
    description      = "End-to-end AI pipeline for geospatial data processing using ISRO satellite datasets",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/Ayush-2703/Geodata_Processing",
    packages         = find_packages(where="src"),
    package_dir      = {"": "src"},
    python_requires  = ">=3.8",
    install_requires = requirements,
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "geodata-pipeline=run_pipeline:main",
        ],
    },
)
