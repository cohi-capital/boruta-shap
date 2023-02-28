from setuptools import setup


def readme():
    with open('README.md') as f:
        README = f.read()
    return README


setup(
    name="BorutaShap",
    version="1.0.17",
    description="A feature selection algorithm.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/cohi-capital/boruta-shap",
    author="Eoghan Keany, Ali Zia",
    author_email="ali@cohi.capital",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    py_modules=["BorutaShap"],
    package_dir={"": "src"},
    install_requires=[
        "catboost",
        "fasttreeshap",
        "lightgbm",
        "matplotlib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "seaborn",
        "shap",
        "tqdm",
        "xgboost"
    ],
)
