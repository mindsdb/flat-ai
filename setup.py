from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flat-ai",
    version="0.1.0",
    author="Yours truly Jorge Torres and an LLM",
    author_email="your.email@example.com",
    description="F.L.A.T. (Frameworkless LLM Agent Thing) for building AI Agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/flat-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers Who Are Too Cool for Frameworks",
        "Intended Audience :: People Who Think AI Should Have More Personality",
        "Intended Audience :: Anyone Who's Ever Said 'I Could Build That Better'",
        "Intended Audience :: Coffee-Powered Code Ninjas",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "pydantic>=2.0.0",
    ],
) 