import setuptools

setuptools.setup(
    name="Triglav",
    version="1.0.0.dev",
    author="Josip Rudar",
    author_email="rudarj@uoguelph.ca",
    description="Supervised Selection of Features Using Iterative Refinement",
    url="https://github.com/jrudar/",
    license="MIT",
    keywords="ecology multivariate statistics",
    packages=["triglav"],
    python_requires=">=3.8",
    install_requires=[
        "numpy >= 1.22.2",
        "sklearn >= 1.0.1",
        "statsmodels >= 0.12.0",
        "shap >= 0.40.0",
        "sage",
        "scipy >= 1.7.3",
        "joblib >= 1.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8+",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Ecology :: Multivariate Statistics",
    ],
)
