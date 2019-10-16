
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='eflow',
     version='0.1',
     scripts=['init_env'] ,
     author="Eric Cacciavillani",
     author_email="eric.cacciavillani@gmail.com",
     description="",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/EricCacciavillani/eFlow",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.7.4",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
