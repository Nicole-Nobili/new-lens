from setuptools import setup, find_packages

setup(
    name='lenses_experiments',
    version='0.0',
    packages=find_packages(),
    install_requires=[ 
        'numpy==1.26.4',                   
        'transformers==4.39.1',            
        'tuned-lens==0.2.0',               
        'torch==2.1.2+cu121',              
        'datasets==2.18.0',                
        'scikit-learn==1.3.2',             
        'tqdm==4.66.1',                    
    ],
    author='Nicole Nobili',
    author_email='nnobili@ethz.ch',
    license='MIT',
)

