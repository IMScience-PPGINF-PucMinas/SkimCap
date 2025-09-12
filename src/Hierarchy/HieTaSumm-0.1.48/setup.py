from setuptools import setup, find_packages

setup(
    name='HieTaSumm',
    version='0.1.48',    
    license='MIT Licence',
    description='A hierarchical approach for video summarization',
    keywords='summarization hierarchy graph',
    author='Leonardo Vilela',
    author_email='leonardo.cardoso.794229@sga.pucminas.br',
    packages=find_packages(),
    package_data={'HieTaSumm': ['options.json','mat-files/*', 'eccv16_dataset_summe_google_pool5.h5']},
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'scikit-learn',
        'torch',
        'Pillow',
        'higra',
        'tensorflow',
        'keras',
        'scipy',
        'networkx',
        'pathlib',
        'h5py'
    ],
)