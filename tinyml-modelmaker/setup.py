from setuptools import setup

if __name__ == '__main__':
    setup()
# import os
# import subprocess
#
# from setuptools import find_packages, setup
#
#
# def git_hash():
#     git_path = '.' if os.path.exists('.git') else ('..' if os.path.exists(os.path.join('..', '.git')) else None)
#     if git_path:
#         hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
#         return hash[:7] if (hash is not None) else None
#     else:
#         return None
#
#
# def get_version():
#     # from version import __version__
#     __version__ = '1.0.0'
#     hash = git_hash()
#     version_str = __version__ + '+' + hash.strip().decode('ascii') if (hash is not None) else __version__
#     return version_str
#
#
# if __name__ == '__main__':
#     version_str = get_version()
#
#     long_description = ''
#     with open('README.md',  encoding="utf8") as readme:
#         long_description = readme.read()
#
#     setup(
#         name='tinyml_modelmaker',
#         version=get_version(),
#         description='Tiny ML ModelMaker',
#         long_description=long_description,
#         long_description_content_type='text/markdown',
#         url='https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelmaker',
#         author='Adithya Thonse',
#         author_email='thonse@ti.com',
#         classifiers=[
#             'Development Status :: 1 - Beta'
#             'Programming Language :: Python :: 3.10'
#         ],
#         keywords = 'artifical intelligence, deep learning, image classification, visual wake words, time series classsification, audio keyword spotting',
#         python_requires='>=3.10',
#         packages=find_packages(),
#         include_package_data=True,
#         install_requires=[],
#         project_urls={
#             'Source': 'https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelmaker',
#             'Bug Reports': 'https://e2e.ti.com/support/processors/',
#         },
#     )

