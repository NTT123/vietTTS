from setuptools import setup

__version__ = '0.1'
url = 'https://github.com/ntt123/vietTTS'

install_requires = ['tabulate', 'optax', 'jax', 'jaxlib', 'einops', 'librosa',
                    'textgrid', 'dm-haiku @ git+https://github.com/deepmind/dm-haiku', 'tqdm', 'gdown']
setup_requires = []
tests_require = []

setup(
    name='vietTTS',
    version=__version__,
    description='A vietnamese text-to-speech system.',
    author='ntt123',
    author_email='xcodevn@gmail.com',
    url=url,
    keywords=['text-to-speech', 'tts', 'deep-learning', 'dm-haiku', 'jax', 'vietnamese', 'speech-synthesis'],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=['vietTTS'],
    python_requires='>=3.6',
)
