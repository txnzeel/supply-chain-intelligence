import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dashboard_layers() -> dict[str, list[str]]:
    tree = ast.parse(
        (PROJECT_ROOT / "dashboard" / "Dashboard.py").read_text(encoding="utf-8")
    )
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "LAYERS" for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError("LAYERS configuration was not found")


def test_dashboard_has_five_layers_and_twenty_four_sections():
    layers = dashboard_layers()
    assert len(layers) == 5
    assert sum(len(sections) for sections in layers.values()) == 24
    assert len({section for sections in layers.values() for section in sections}) == 24


def test_deployment_assets_exist():
    for relative_path in [
        "Dockerfile",
        ".dockerignore",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "docs/DEPLOYMENT.md",
    ]:
        assert (PROJECT_ROOT / relative_path).is_file()
