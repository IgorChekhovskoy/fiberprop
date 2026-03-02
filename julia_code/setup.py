from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description=fh.read()

setup(
    name="julia_compute",
    version="0.1",
    description="Пакет является обёрткой для вычислительно сложных функций, реализованных на языке Julia",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True
)
