'''HAllA setup

To install: python setup.py install
'''

import sys

try:
	import setuptools
except ImportError:
	sys.exit('Please install setuptools.')

VERSION = '0.0.4'
AUTHOR  = 'HAllA Development Team'
MAINTAINER_EMAIL = 'kathleen_sucipto@hms.harvard.edu'

with open('readme.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='HAllA',
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    version=VERSION,
    license='MIT',
    description='HAllA: Hierarchical All-against All Association Testing',
    long_description=long_description,
    url='https://github.com/biobakery/halla_revised',
    keywords=['halla', 'association testing'],
    platforms=['Linux','MacOS'],
    classifiers=[
        'Programming Language :: Python',
        'Environment :: Console',
		'Operating System :: MacOS',
		'Operating System :: Unix',
		'Programming Language :: Python :: 3.7',
		'Topic :: Scientific/Engineering :: Bio-Informatics'
      ],
    packages=setuptools.find_packages(),
    package_data={
        'halla': ['config.yaml']
    },
    entry_points={
        'console_scripts': [
            'halla = scripts.halla:main',
            # TODO: for generating synthetic data
            # TODO: for generating hallagram / diagnostic plot / clustermap separately
        ]
    },
    test_suite= 'tests',
 )
