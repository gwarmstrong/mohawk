from setuptools import setup, find_packages

# https://click.palletsprojects.com/en/7.x/setuptools/
setup(
    name='mohawk',
    version='0.1dev',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
    ],
    author="George Armstrong",
    license='BSD-3-Clause',
    author_email="garmstro@eng.ucsd.edu",
    url="https://github.com/gwarmstrong/mohawk",
    description="Taxonomic Classification and Metagenomic Experiment "
                "simulator",
    entry_points='''
        [console_scripts]
        mohawk=mohawk.scripts.mohawk:mohawk
    '''
)
