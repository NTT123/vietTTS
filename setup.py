from setuptools import setup

__version__ = "0.4.0b"
url = "https://github.com/ntt123/vietTTS"

install_requires = [
    "dm-haiku",
    "einops",
    "gdown",
    "jax",
    "jaxlib",
    "librosa",
    "optax",
    "tabulate",
    "textgrid",
    "tqdm",
]
setup_requires = []
tests_require = []

setup(
    name="vietTTS",
    version=__version__,
    description="A vietnamese text-to-speech engine.",
    author="Thông Nguyễn",
    url=url,
    keywords=[
        "deep-learning",
        "dm-haiku",
        "jax",
        "speech-synthesis",
        "text-to-speech",
        "tpu",
        "tts",
        "vietnamese",
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=["vietTTS"],
    python_requires=">=3.6",
)
