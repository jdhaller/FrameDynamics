from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(name='FrameDynamics',
      version='0.1.2',
      description='Simulations of the average Hamiltonian.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Physics'],
      keywords='NMR interaction average Hamiltonian theory toggling frame',
      url='http://github.com/jdhaller/FrameDynamics',
      author='Jens Daniel Haller',
      author_email='jhaller@gmx.de',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
            install_requires = [
          'numpy',
          'scipy>=1.8.0',
          'matplotlib'],
      zip_safe=False)


