from setuptools import setup, find_packages

setup(
    name="LLM_understanding",
    version="0.1.0",
    author="Pegah Golchin",
    author_email="golchinpg@gmail.com",
    description="Understanding the codebase of a tabular transformer model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/golchinpg/LLM",  # Update with your repository URL
    packages=find_packages(exclude=["tests*", "scripts*"]),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",  # Update to the version you need
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "flake8>=3.9.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-tabular=tabular_transformer.scripts.train:main",
            "evaluate-tabular=tabular_transformer.scripts.evaluate:main",
        ],
    },
)
