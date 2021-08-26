from setuptools import setup, find_packages

setup(
  name = 'MPPMCTS',
  packages=find_packages(),
  include_package_data=True,
  version = '0.1',
  description = 'Multi-Player-Parallel MCTS for ILs design',
  author = 'Kexin Zhang',
  author_email = 'zhangkx2@shanghaitech.edu.cn',
  license='MIT',
  keywords = ['computational chemistry', 'AI',"ML","COSMO","MCTS"],
  classifiers = [
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        ],
  install_requires=['torch',"pandas","scipy","numpy","matplotlib","argparse"],
  entry_points={
        'console_scripts': [
            'mppmcts=MPPMCTS.run:run',
        ],
    },

)
