import os.path as op
import setuptools

thisdir = op.dirname(__file__)

with open(op.join(thisdir, 'pyfix', 'VERSION'), 'rt') as f:
    version = f.read().strip()

with open(op.join(thisdir, 'requirements.txt'), 'rt') as f:
    install_requires = [l.strip() for l in f.readlines()]

setuptools.setup(
    name="pyfix",
    version=version,
    author="Sean Fitzgibbon",
    author_email="sean.fitzgibbon@ndcn.ox.ac.uk",
    description="Python implementation of FSL's FIX tool for denoising",
    url="https://git.fmrib.ox.ac.uk/seanf/pyfix",
    install_requires=install_requires,
    scripts=['pyfix/fix'],
    packages=['pyfix'],
    package_data={'pyfix': ['VERSION', 'resources/*', 'resources/masks/*']},
    include_package_data=True,
    python_requires='>=3.8',
)
