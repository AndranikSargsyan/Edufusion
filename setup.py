from setuptools import setup, find_packages

setup(
    name="edufusion",
    version="0.1.0",
    author="Andranik Sargsyan",
    author_email="and.sargsyan@yahoo.com",
    description="DIY implementation of Stable Diffusion 1.5 with minimal dependencies.",
    packages=find_packages("edufusion"),
    install_requires=[
        "ftfy==6.3.0",
        "regex==2024.9.11",
        "torch>=2.0.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)