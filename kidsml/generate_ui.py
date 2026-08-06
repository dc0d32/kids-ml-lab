"""A regenerate button for steps that roll dice.

Anything that samples — babbled words, generated names, generated text, a sampled
picture — needs a way to ask for another go. Streamlit only re-runs the script when
*something* changes, so a step that samples once looks frozen: you change the prompt and the
same words come back, because the sampling was pinned to the old roll.

:func:`regenerate` renders a button and hands back a whole number that grows by one every
time the button is pressed. Feed that number in as a seed (or as part of a cache key) and a
press produces genuinely different output::

    from kidsml import generate_ui

    roll = generate_ui.regenerate(key="babble")
    words = make_words(prompt, seed=roll)

The ``key`` must be unique on the page, the way any Streamlit widget key must be.
"""

from __future__ import annotations

import streamlit as st


def regenerate(label: str = "🎲 Generate again", key: str = "", help: str | None = None) -> int:
    """A button that hands back a bigger number every time it is pressed.

    Returns 0 on first render, then 1, 2, 3, ... on each press. The count lives in
    ``st.session_state`` so it survives re-runs triggered by other widgets.
    """
    slot = f"kml_roll_{key or label}"
    if slot not in st.session_state:
        st.session_state[slot] = 0

    def bump() -> None:
        st.session_state[slot] += 1

    st.button(label, key=slot + "_btn", on_click=bump, help=help, type="primary")
    return int(st.session_state[slot])
