"""
tests/test_scraper.py
─────────────────────
Tests básicos del scraper y pipeline.
"""
import pytest
from unittest.mock import patch, MagicMock


def test_extract_text_basic():
    """Test extracción de texto HTML."""
    from src.scraper import _extract_text

    html = """
    <html><head><title>Test Recamier</title></head>
    <body>
        <nav>Menu de navegación</nav>
        <main>
            <h1>Keratina Recamier</h1>
            <p>Producto de alta calidad para el cabello seco y dañado.</p>
        </main>
        <footer>Footer del sitio</footer>
    </body></html>
    """
    result = _extract_text(html, "https://recamier.com/test")
    assert "Keratina Recamier" in result
    assert "recamier.com" in result


def test_extract_links_same_domain():
    """Test extracción de enlaces internos."""
    from src.scraper import _extract_links

    html = """
    <html><body>
        <a href="/productos">Productos</a>
        <a href="https://recamier.com/marcas">Marcas</a>
        <a href="https://otro.com/externo">Externo</a>
        <a href="mailto:info@recamier.com">Email</a>
    </body></html>
    """
    links = _extract_links(html, "https://recamier.com/", ["recamier.com"])
    assert any("recamier.com" in l for l in links)
    assert not any("otro.com" in l for l in links)
    assert not any("mailto:" in l for l in links)


def test_random_ua():
    """Test que el User-Agent se genera correctamente."""
    from src.scraper import _random_ua
    ua = _random_ua()
    assert isinstance(ua, str)
    assert len(ua) > 20
    assert "Mozilla" in ua


def test_build_headers():
    """Test que las cabeceras HTTP son completas."""
    from src.scraper import _build_headers
    headers = _build_headers()
    assert "User-Agent" in headers
    assert "Accept" in headers
    assert "Accept-Language" in headers
    assert "es" in headers["Accept-Language"]
