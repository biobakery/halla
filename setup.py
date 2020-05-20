from setuptools import setup

# Include the readme file in the build
def readme():
    with open('README.rst') as f:
        return f.read()

# TODO: Edit the function parameter with the relevant information on this package
setuptools.setup(
    name="HAllA",
    author="AUTHOR",
    author_email="AUTHOR_EMAIL",
    version="0.0.0",
    license="",
    description="Tool Description2",
    long_description="Detailed tool description",
    url="",
    keywords=[],
    platforms=['Linux','MacOS'],
    classifiers=[
        'Development Status :: Alpha',
        'License :: MIT License',
        'Programming Language :: Python ',
        'Topic :: Python tools :: Public',
      ],
    packages=setuptools.find_packages(),
    test_suite= 'sample_test.TestStringMethods',
    tests_require=['TestStringMethods'],
    zip_safe = False,
    install_requires=[
          'markdown',
          'numpy'
      ]
 )
