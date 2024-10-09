from pathlib import Path
from setuptools import setup

CWD = Path(__file__).absolute().parent


def get_version():
    # Gets the version
    path = CWD / "neural_dnf" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="neural_dnf",
    version=get_version(),
    description="Neural DNF-based models",
)
