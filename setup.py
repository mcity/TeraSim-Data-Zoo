from setuptools import setup, find_packages

setup(
    name='scenparse',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # 在这里列出你的项目依赖
        # 'some_package>=1.0',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'your_script=your_package.your_module:main_function',
    #     ],
    # },
)