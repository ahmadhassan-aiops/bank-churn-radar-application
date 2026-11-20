"""Smoke test: every page renders without an exception, and the example buttons work."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["views/predict.py", "views/portfolio.py", "views/performance.py", "views/method.py"]


def _page(path: str) -> AppTest:
    module = path.removesuffix(".py").replace("/", ".")
    script = f"""
import streamlit as st
from scope.ui import inject_css
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


def test_strip_board_call_line_moves():
    at = _page("views/portfolio.py").run()
    at.slider[0].set_value(0.5).run()
    assert not at.exception
    at.toggle(key="calls_only").set_value(False).run()
    assert not at.exception


def test_calibration_slider_moves():
    at = _page("views/performance.py").run()
    at.slider[0].set_value(0.3).run()
    assert not at.exception
