from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SEED",
    version="1.0.0",
    author="Juhyeon Park, Peter Yongho Kim",
    author_email="peterkim98@snu.ac.kr", 
    description="SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Concarne2/SEED",
    packages=find_packages(include=["seed", "seed.*", "mmdet", "mmdet.*"]),
    package_data={
        "mmdet": [
            "configs/**/*.py",
            "configs/**/*.yml",
            "configs/**/*.yaml",
            "configs/**/*.md",
            "configs/**/*.json",
            "configs/**/*.png",
            "configs/**/*.jpg",
            ".mim/**/*",
            "*.yml",
            "*.yaml",
            "**/*.json",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fairscale",
        "transformers",
        "sentence-transformers",
        "numpy",
        "matplotlib",
        "pycocotools",
        "scipy",
        "shapely",
        "terminaltables",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
)
