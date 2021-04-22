#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Thomas Holz",
    author_email='t.holz@campus.tu-berlin.de',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Calculates head-realated impulse responses (HRIRs) of a spherical head model with offset ears using the formulation from according to [1]. HRIRs are calculated by dividing the pressure on the sphere by the free field pressure of a point source in the origin of coordinates.",
    entry_points={
        'console_scripts': [
            'spheadpy=spheadpy.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='spheadpy',
    name='spheadpy',
    packages=find_packages(include=['spheadpy', 'spheadpy.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thomaschhh/spheadpy',
    version='0.1.0',
    zip_safe=False,
)
