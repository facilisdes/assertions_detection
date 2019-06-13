from setuptools import setup

setup(
    name='reviewsSpeechActAnalyzer',
    version='0.7',
    packages=['common', 'features', 'classification', 'clusterization'],
    url='https://github.com/facilisdes/assertions_detection',
    license='',
    author='facilisdes',
    author_email='facilisdes@gmail.com',
    description='',
    install_requires=['redis', 'emoji', 'pymystem3', 'PyYAML', 'scikit-learn', 'sklearn'],
)
