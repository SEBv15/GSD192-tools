from setuptools import setup, find_packages

setup(
    name='gsd192_tools',
    version='0.1.2',
    description='Tools for the Germanium Strip Detector',
    author='Sebastian Strempfer',
    author_email='sebastian@strempfer.com',
    url='https://github.com/SEBv15/GSD192-tools',
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: IPython",
        "Framework :: Matplotlib",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Utilities",
    ],
    platforms=["any"],
    python_requires='>=3.4',
    install_requires=[
        "matplotlib",
        "numpy",
        "zmq", # zclient
        "lmfit", # calibration
        "dtw-python==1.1.5", # calibration
        "pandas" # Reading in mca files
    ],
    entry_points={
        'console_scripts': [
            'gsd192-configure=gsd192_tools.configure:main', 
            'gsd192-monitor=gsd192_tools.gsd192_monitor:main',
            'gsd192-time-mca=gsd192_tools.zclient_time_mca:main'
        ],
    }
)
