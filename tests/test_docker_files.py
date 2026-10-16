from pipeline.config import PROJECT_ROOT


def read(name):
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_is_cpu_only_and_ships_no_data():
    d = read("Dockerfile")
    assert d.startswith("FROM python:3.14-slim") and "uvicorn" in d and "serving.app:app" in d
    assert "COPY data" not in d and "cu130" not in d and "whl/cpu" in d


def test_dockerignore_keeps_data_and_models_out_of_the_image():
    ignore = read(".dockerignore").split()
    assert {"data", "models/*.pt", "outputs", "notebooks", "tests", ".git"} <= set(ignore)   # models/ is also a package


def test_serving_requirements_are_minimal():
    req = read("requirements-serving.txt")
    for needed in ("fastapi", "uvicorn", "torch-geometric", "numpy", "pandas"):
        assert needed in req
    for banned in ("pytest", "jupyter", "ipykernel", "scikit-learn", "matplotlib", "cu130"):
        assert banned not in req
