from setuptools import setup, find_packages

setup(
    name="infraagent",
    version="1.0.0",
    description="InfraAgent: Empirical study of RAG vs self-correction for IaC generation",
    author="Anonymous",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "openai>=1.30.0",
        "anthropic>=0.30.0",
        "jsonlines>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "infraagent-run=scripts.run_experiments:main",
            "infraagent-plot=scripts.plot_figures:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
