"""Sliders in a notebook, without hanging the test runner.

`ipywidgets.interact` expects a live browser on the other end. Under automated notebook
execution there isn't one, and the kernel can sit waiting for a frontend that will never
answer — intermittently, which is the worst kind of bug to chase.

So: in JupyterLab you get real sliders, and under `./run.sh test` the same function is
called once with the slider defaults. The code still runs, the figure is still drawn, and
nothing waits for a browser.
"""

from __future__ import annotations

import os


def is_headless() -> bool:
    """True when a notebook is being executed by the test runner rather than a person."""
    return os.environ.get("KIDSML_HEADLESS") == "1"


def interact(func, **controls):
    """Show sliders for ``func``'s arguments, or run it once if nobody is watching.

    Use exactly like ``ipywidgets.interact``::

        interactive.interact(
            draw,
            angle=IntSlider(value=20, min=-180, max=180, step=5),
        )
    """
    if not is_headless():
        from ipywidgets import interact as _interact

        return _interact(func, **controls)

    defaults = {}
    for name, control in controls.items():
        # A widget carries its starting value; a plain list or tuple is a range of options.
        if hasattr(control, "value"):
            defaults[name] = control.value
        elif isinstance(control, (list, tuple)) and control:
            defaults[name] = control[0]
        else:
            defaults[name] = control
    return func(**defaults)
