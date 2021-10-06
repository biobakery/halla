'''HAllA setup

To install: python setup.py install
'''

import sys

try:
    import setuptools
    from setuptools.command.install import install
except ImportError:
    sys.exit('Please install setuptools.')

VERSION = '0.8.20'
AUTHOR  = 'HAllA Development Team'
MAINTAINER_EMAIL = 'halla-users@googlegroups.com'

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

# Installing requirements.txt dependencies
dependencies=[]
requirements = open('requirements.txt', 'r')
for dependency in requirements:
    dependencies.append(str(dependency))
    
setuptools.setup(
    name='HAllA',
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    version=VERSION,
    license='MIT',
    description='HAllA: Hierarchical All-against All Association Testing',
    long_description="Given two high-dimensional 'omics datasets X and Y (continuous and/or categorical features) from the same n biosamples, HAllA (Hierarchical All-against-All Association Testing) discovers densely-associated blocks of features in the X vs. Y association matrix where: 1) each block is defined as all associations between features in a subtree of X hierarchy and features in a subtree of Y hierarchy and 2) a block is densely associated if (1 - FNR)% of pairwise associations are FDR significant (FNR is the pre-defined expected false negative rate)",
    url='https://github.com/biobakery/halla',
    keywords=['halla', 'association testing'],
    platforms=['Linux','MacOS'],
    install_requires=dependencies,
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
