from setuptools import setup, find_packages

pkg_details = {}
with open("pandora/_pkgdetails.py") as fp:
    exec(fp.read(), pkg_details)

setup(
    name=pkg_details['name'],
    version=pkg_details['version'],
    url=pkg_details['url'],
    author=pkg_details['author'],
    author_email=pkg_details['author_email'],
    description=pkg_details['description'],
    license=pkg_details['license'],
    packages=find_packages(), install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'opencv-python',
        'scikit-learn',
        'tensorflow',
        'joblib'
    ]
)
