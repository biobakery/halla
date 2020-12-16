'''HAllA setup

To install: python setup.py install
'''

import sys

try:
    import setuptools
    from setuptools.command.install import install
except ImportError:
    sys.exit('Please install setuptools.')

VERSION = '0.8.18'
AUTHOR  = 'HAllA Development Team'
MAINTAINER_EMAIL = 'halla-users@googlegroups.com'

with open('readme.md', 'r') as fh:
    long_description = fh.read()

class PostInstallCommand(install):
    '''Post-installation for installation mode'''
    def run(self):
        install.run(self)
        # post-install script
        from rpy2.robjects.packages import importr
        try:
            eva = importr('eva')
        except:
            utils = importr('utils')
            utils.chooseCRANmirror(ind=1)
            utils.install_packages('EnvStats')
            utils.install_packages('https://cran.r-project.org/src/contrib/Archive/eva/eva_0.2.5.tar.gz')
            # check if eva has been successfully installed
            eva = importr('eva')
        try:
            XICOR = importr('XICOR')
        except:
            utils = importr('utils')
            utils.chooseCRANmirror(ind=1)
            utils.install_packages("XICOR")
            XICOR = importr('XICOR')


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
            'halladata = scripts.synthetic_data:main',
            'hallagram = scripts.hallagram:main',
            'hallagnostic = scripts.diagnostic_plot:main',
        ]
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    test_suite= 'tests',
 )
