from setuptools import setup, find_packages

setup(
    name='MIRAGE',  # Replace with your package name
    version='0.1.0',           # Version of your package
    author='r.kulik, ev.sorokin',        # Your name
    author_email='r.kulik@innopolis.university',  # Your email
    description='Multidomain Intelligent Retrieval Augmented Generation Engine',  # Short description
    long_description=open('README.md').read(),  # Long description from README
    long_description_content_type='text/markdown',  # Type of long description
    url='https://github.com/r-kulik/MIRAGE',  # Project URL
    install_requires=[         # List of dependencies
        'numpy==1.26.4',
        'pdfplumber>=0.11.5', 
        'python-docx>=1.1.2',
        'tqdm>=4.67.1',
        'faiss-cpu==1.10.0',
        'natasha>=1.6.0',
        'nltk>=3.9.1',
        'scikit-learn>=1.6.1',
        'sentence-transformers>=3.4.1',
        'chardet>=5.2.0',
        'pytest>=8.3.4',
        'fpdf>=1.7.2',
        'pydantic>=2.10.6',
        'whoosh>=2.7.4',
        'pymorphy3>=2.0.3',
        'gensim>=4.3.3',
        'loguru>=0.7.3'
    ],
    classifiers=[              # Classifiers for PyPI (optional)
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',   # Python version requirement
    py_modules=['mirage']
)