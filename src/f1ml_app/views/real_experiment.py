"""Dedicated, read-only real-data experiment dashboard."""

from f1ml_app import ui
from f1ml_app.real_dashboard import render

ui.guard(render)
