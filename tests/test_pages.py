"""Smoke test: every page renders without an exception, and the example buttons work."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/predict.py", "views/portfolio.py", "views/performance.py", "views/method.py"]


def _page(path: str) -> AppTest:
    module = path.removesuffix(".py").replace("/", ".")
    script = f"""
import streamlit as st
from lib.ui import inject_css
import {module} as page
inject_css()
page.render()
"""
    return AppTest.from_string(script, default_timeout=60)


@pytest.mark.parametrize("path", PAGES)
def test_page_renders(path):
    at = _page(path).run()
    assert not at.exception, at.exception


def test_example_button_loads_a_test_customer():
    at = _page("views/predict.py").run()
    at.button[1].click().run()  # "At-risk example"
    assert not at.exception
    assert at.session_state["example_note"].startswith("Loaded test customer")
