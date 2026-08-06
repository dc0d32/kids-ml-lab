"""Kids ML Lab — the front door and the chapter list.

Streamlit derives a sidebar label from the filename and strips the leading number, so
``00_guessing_machine.py`` would show up as "guessing machine". The chapter numbers matter
— they are how the course refers to itself — so the navigation is declared explicitly here,
straight from ``CHAPTERS``.

Run with:  ./run.sh app
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))

from kidsml.lesson import apply_style  # noqa: E402
from kidsml.ui import CHAPTERS, page_filename  # noqa: E402

st.set_page_config(page_title="Kids ML Lab", page_icon="🧪", layout="centered")
apply_style()

pages = [
    st.Page(str(APP / "welcome.py"), title="The course map", icon="🧪", default=True)
]
for number, slug, title, _idea, _part in CHAPTERS:
    pages.append(
        st.Page(
            str(APP / "pages" / page_filename(number)),
            title=f"{number:02d} · {title}",
            url_path=f"{number:02d}_{slug}",
        )
    )

st.navigation(pages).run()
