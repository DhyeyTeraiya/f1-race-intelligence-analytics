from setuptools import setup, find_packages

setup(
    name="f1-race-intelligence-analytics",
    version="1.0.1",
    author="Dhyey Teraiya",
    author_email="dhyeyteraiya@gmail.com",
    description="Formula 1 Race Intelligence, Strategy Undercut Simulator & Predictive Modeling Platform",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
