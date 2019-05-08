from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='xklearn',
    version='0.0.1',
    description='Handy machine learning tools in the spirit of scikit-learn.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    author='Simon Larsson',
    author_email='simonlarsson0@gmail.com',
    url='https://github.com/simon-larsson/extrakit-learn',
    license='MIT',
    install_requires=['numpy', 'sklearn'],
    classifiers=['License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering'
        ]
)