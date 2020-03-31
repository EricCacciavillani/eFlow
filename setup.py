
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

MAIN_PACKAGES = [i.strip() for i in open("Installation/main_packages.txt").readlines()]

setuptools.setup(
     name='eflow',
     version='0.1.99',
     scripts=['init_env'],
     author="Eric Cacciavillani",
     author_email="eric.cacciavillani@gmail.com",
     description="",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/EricCacciavillani/eFlow",
     packages=setuptools.find_packages(),
     install_requires=MAIN_PACKAGES,
     classifiers=[
         "Programming Language :: Python :: 3.7",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
