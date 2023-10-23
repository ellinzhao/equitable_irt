import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='equitable_irt',
    version='0.0.1',
    author='ellin',
    author_email='ellinz@ucla.edu',
    description='Code for the equitable IRT project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages()
)
