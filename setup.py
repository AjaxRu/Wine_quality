from setuptools import setup, find_packages

setup(
    name="Wine_quality_model",
    version="0.1.0",
    author="Nastia",
    author_email="stupinaaa99@gmail.com",
    description="A package for wine quality prediction using machine learning",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.1.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.3.0",
        "pydantic>=2.0.0",
        "pytest>=6.2.3"
    ],
    extras_require={
        "dev": [
            "black>=20.8b1",
            "flake8>=3.9.0",
            "isort>=5.8.0",
            "mypy>=0.812",
            "pytest>=6.2.3"
        ]
    }
)
