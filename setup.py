from setuptools import setup

__version__ = "0.4.1"
url = "https://github.com/ntt123/vietTTS"

install_requires = [
    "dm-haiku",
    "einops",
    "fire",
    "gdown",
    "jax",
    "jaxlib",
    "librosa",
    "optax",
    "tabulate",
    "textgrid @ git+https://github.com/kylebgorman/textgrid.git",
    "tqdm",
    "matplotlib",
]
setup_requires = []
tests_require = []

setup(
    name="vietTTS",
    version=__version__,
    description="A vietnamese text-to-speech library.",
    author="ntt123",
    url=url,
    keywords=[
        "text-to-speech",
        "tts",
        "deep-learning",
        "dm-haiku",
        "jax",
        "vietnamese",
        "speech-synthesis",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=["vietTTS"],
    python_requires=">=3.7",
)
