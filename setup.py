from setuptools import find_packages, setup


root_packages = find_packages(where=".")
src_packages = find_packages(where="src")

setup(
    name="cans-svd",
    version="0.1.0",
    description="CANS-based SVD wrappers for Torch and JAX backends.",
    packages=root_packages + src_packages,
    package_dir={"cans_svd": "src/cans_svd"},
    python_requires=">=3.10",
    install_requires=[
        "jax",
        "numpy",
        "torch",
    ],
    extras_require={
        "dev": ["pytest"],
    },
)
