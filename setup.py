from setuptools import setup, find_packages

setup(
    name='lenses_experiments',
    version='0.0',
    packages=find_packages(),
    install_requires=[ 
        'numpy',
        'transformers',
        'tuned_lens',
        'torch',
        'datasets',
        'sklearn',
    ],
    author='Nicole Nobili',
    author_email='nnobili@ethz.ch',
    license='MIT',
)
