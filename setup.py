from setuptools import setup, find_packages

setup(
    name='statool-edu',
    version='0.0.1',
    description='A package for statistical education tools',
    author='surish',
    author_email='sbhath17@gmail.com',
    url='https://github.com/Surihub/statool-edu',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'koreanize-matplotlib==0.1.1',
        'matplotlib==3.7.1',
        'numpy==1.23.5',
        'seaborn==0.12.2',
        'plotly==5.16.1',
        'scipy==1.10.1'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
