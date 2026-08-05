"""Kids ML Lab — the front door.

Run with:  ./run.sh app
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make ``import kidsml`` work no matter where streamlit was started from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kidsml.plots import use_house_style  # noqa: E402
from kidsml.ui import CHAPTERS  # noqa: E402

st.set_page_config(page_title="Kids ML Lab", page_icon="🧪", layout="wide")
use_house_style()

st.title("🧪 Kids ML Lab")
st.markdown(
    "#### Machine learning, taken apart until it stops being scary.\n"
    "26 chapters. Every idea starts with numbers small enough to work out on scrap paper, "
    "then becomes a picture you can poke with a slider, and only *then* becomes code."
)

st.divider()

left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("How every chapter works")
    st.markdown(
        """
A chapter is a run of small screens. **One idea per screen**: something to read,
something to look at, and something to move. Ten screens or so, then you're done.

Along the way a chapter will stop and **ask you what you think will happen** before it
shows you. Guessing wrong is useful — being surprised is how the idea sticks.

Each chapter walks the same six beats:

| | Beat | What happens |
|---|---|---|
| 🎣 | **Start here** | A question or a game. Plain English. No math, no code. |
| ✏️ | **Work it out** | A few rows of tiny numbers, small enough for scrap paper. |
| 👀 | **Take a look** | A picture of the exact thing you just worked out. |
| 🎛️ | **Your turn** | Sliders. Move one knob, watch the picture change instantly. |
| 💻 | **In real code** | 10–25 lines of actual code on actual data. |
| 🏆 | **Go further** | Beat the machine, or break it on purpose. |

Every chapter also has a **🧸 Little Kid Corner** — the same idea with no algebra in it.
"""
    )

with right:
    st.subheader("Two ways to take the course")
    st.markdown(
        """
**This app** is for playing. Drag sliders, break things, see what happens.
No code to read unless you want to.

**The notebooks** are the same chapters with the code showing, so you can change it
and re-run it:

```bash
./run.sh lab
```

Do a chapter here first, then open the notebook if you want to look under the hood.
"""
    )
    st.info(
        "👈 Pick a chapter from the sidebar to begin.\n\n"
        "If you have never done any of this before, start at **Chapter 00**. "
        "It takes about half an hour and there is no maths in it."
    )

st.divider()
st.subheader("The whole course")

current_part = None
for number, slug, title, idea, part in CHAPTERS:
    if part != current_part:
        current_part = part
        st.markdown(f"##### {part}")
    st.markdown(f"**{number:02d} · {title}** — {idea}")

st.divider()
st.caption(
    "House rules: simple Python, small data, and every chapter trains on a plain laptop "
    "in seconds to a couple of minutes. If something gets slow, the dataset shrinks — "
    "never the explanation."
)
