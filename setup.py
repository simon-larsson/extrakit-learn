from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='extrakit-learn',
    version='0.1.0',
    description='.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['xklearn'],
    author='Simon Larsson',
    author_email='simonlarsson0@gmail.com',
    url='https://github.com/simon-larsson/extrakit-learn',
    license='MIT',
    install_requires=['numpy', 'sklearn'],
    classifiers=['License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        ]
)