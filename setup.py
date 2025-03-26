from setuptools import setup, find_packages

setup(
    name='portfolio_optimization',
    version='0.1.0',
    description='A portfolio optimization project using Python',
    author='Gustav Lantz',
    author_email='lantzgustav@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
    ],
)
