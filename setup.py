from setuptools import setup, find_packages

exec(open('rwtools/__version__.py').read())
setup(
    name='plantseg',
    version=__version__,
    packages=find_packages(exclude=["tests", "benchmarks"]),
    include_package_data=False,
    description='random walker algorithm implementation.',
    author='Lorenzo Cerrone',
    url='https://github.com/lorenzocerrone/random-walker-tools',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)