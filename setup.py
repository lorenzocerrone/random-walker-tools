from setuptools import setup, find_packages

exec(open('randomwalkertools/__version__.py').read())
setup(
    name='plantseg',
    version=__version__,
    packages=find_packages(exclude=["tests", "benchmark"]),
    include_package_data=False,
    description='random walker algorithm implementation.',
    author='Lorenzo Cerrone',
    url='https://github.com/lorenzocerrone/random-walker-tools',
    author_email='lorenzo.cerrone@iwr.uni-heidelberg.de',
)