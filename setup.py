from setuptools import setup, find_packages

setup(
    name='model_navigator',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        "loguru",
        "py-cpuinfo",
        "psutil",
        'pyyaml',
        'importlib_metadata',
        'pyee',
        'rich',
        'pynvml',
        'dacite',
        'tabulate',
        'wrapt',
        'tritonclient[grpc]',
        'polygraphy',
        'tritonclient==2.45.0',
        'onnx_graphsurgeon',
        'colored',
        'onnxruntime',
    ],
)
