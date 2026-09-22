from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements_path = Path(__file__).with_name("requirements.txt")
    return [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name='ExpertFlow',
    version='0.1.0',
    description='Efficient Mixture-of-Experts inference via predictive expert caching and token scheduling',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
