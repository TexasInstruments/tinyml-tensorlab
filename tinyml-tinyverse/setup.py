import os
import subprocess
from setuptools import setup, Extension, find_packages


def git_hash():
    git_path = '.' if os.path.exists('.git') else ('..' if os.path.exists(os.path.join('..', '.git')) else None)
    if git_path:
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return hash[:7] if (hash is not None) else None
    else:
        return None


def get_version():
    # from version import __version__
    __version__ = '0.8.0'
    hash = git_hash()
    version_str = __version__ + '+' + hash.strip().decode('ascii') if (hash is not None) else __version__
    return version_str


if __name__ == '__main__':
    version_str = get_version()

    long_description = ''
    with open('README.md',  encoding="utf8") as readme:
        long_description = readme.read()

    setup(
        name='tinyml_tinyverse',
        version=get_version(),
        description='Tiny ML TinyVerse',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://bitbucket.itg.ti.com/projects/TINYML-ALGO/repos/tinyml-tinyverse',
        author='Adithya Thonse',
        author_email='thonse@ti.com',
        classifiers=[
            'Development Status :: 1 - Beta'
            'Programming Language :: Python :: 3.12'
        ],
        keywords = 'artifical intelligence, deep learning, image classification, visual wake words, time series classsification, audio_classification keyword spotting',
        python_requires='>=3.12',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[],
        project_urls={
            'Source': 'https://bitbucket.itg.ti.com/projects/TINYML-ALGO/repos/tinyml-tinyverse',
            'Bug Reports': 'https://e2e.ti.com/support/processors/',
        },
    )
