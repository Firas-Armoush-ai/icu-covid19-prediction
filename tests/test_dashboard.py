"""The dashboard pages run without errors and every chart they show exists."""

import re
import sys

import pytest
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route
from starlette.testclient import TestClient
from streamlit.testing.v1 import AppTest
from streamlit.web.server.starlette.starlette_gzip_middleware import MediaAwareGZipMiddleware

from src.config import PROJECT_ROOT

FRONTEND = PROJECT_ROOT / "app" / "frontend"
FIGURES = PROJECT_ROOT / "reports" / "figures"


@pytest.fixture
def frontend_on_path(monkeypatch):
    # `streamlit run` puts the entry script's folder on sys.path, so the pages
    # import their helpers as top-level `config` and `utils` modules.
    monkeypatch.syspath_prepend(str(FRONTEND))
    for name in ("config", "utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def run_page(path):
    return AppTest.from_file(str(path), default_timeout=30).run()


def test_prediction_page_loads(frontend_on_path):
    # Only the form is built on load; the API is called when it is submitted.
    at = run_page(FRONTEND / "predict.py")
    assert not at.exception
    assert len(at.slider) == 5


def test_evaluation_page_shows_every_chart(frontend_on_path):
    at = run_page(FRONTEND / "evaluation.py")
    assert not at.exception
    assert not at.warning  # a missing figure or metadata file shows a warning
    assert len(at.get("imgs")) == 4
    assert len(at.get("graphviz_chart")) == 1


def test_navigation_reaches_both_pages(frontend_on_path):
    at = run_page(FRONTEND / "streamlit_app.py")
    assert not at.exception
    assert len(at.slider) == 5  # the prediction page is the default

    at.switch_page("evaluation.py").run()
    assert not at.exception
    assert any("evaluated" in title.value for title in at.title)


def test_figures_in_readme_exist():
    readme = (PROJECT_ROOT / "README.md").read_text()
    figures = set(re.findall(r"reports/figures/[\w.-]+\.png", readme))

    assert len(figures) >= 4
    for figure in figures:
        assert (PROJECT_ROOT / figure).is_file(), figure


@pytest.mark.parametrize("encoding", ["gzip", "identity"])
def test_streamlit_can_serve_png_images(encoding):
    # Streamlit serves st.image files through this middleware. With
    # Starlette 1.8.0 it sends the body before the headers and every PNG
    # fails with HTTP 500, which is why pyproject.toml pins Starlette.
    async def png(request):
        return Response(b"\x89PNG" + b"0" * 2000, media_type="image/png")

    app = MediaAwareGZipMiddleware(Starlette(routes=[Route("/figure.png", png)]))

    with TestClient(app) as client:
        response = client.get("/figure.png", headers={"Accept-Encoding": encoding})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
