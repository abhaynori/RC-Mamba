from setuptools import setup, find_packages

setup(
    name="rc_mamba",
    version="0.1.0",
    packages=find_packages(),
    description="Retrieval‑Conditioned Mamba with FiLM, dynamic retrieval, and dual‑codebook quantization",
    author="Anonymous",
    author_email="",
    install_requires=[
        # Note: These dependencies are illustrative.  You may need to install mamba‑ssm, sentence‑transformers, faiss‑cpu, accelerate, and gradio separately.
        "torch",
    ],
)