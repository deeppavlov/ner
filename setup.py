from setuptools import (
    setup,
    find_packages,
)


def get_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


setup(
    name='ner',
    version='0.0.1',
    description='Named-entity recognition for russian language',
    url='https://github.com/deepmipt/ner',
    author='???',
    author_email='???',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='natural language processing, russian morphology, named entity recognition',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
