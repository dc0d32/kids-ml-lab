"""Chapter 09 · Real Data, Real Mess."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kidsml import datasets, realdata, ui
from kidsml.plots import ACCENT, confusion_grid

ui.page_setup(9)


TABLES = list(realdata.REAL_TABLES)


def dataset_label(name: str) -> str:
    return f"{name} — {datasets.blurb_of(name)}"


@st.cache_data(show_spinner=False)
def cached_overview(name: str):
    return realdata.table_overview(name)


@st.cache_data(show_spinner=False)
def cached_scores():
    return realdata.all_dataset_scores()


@st.cache_data(show_spinner=False)
def cached_missing_scores():
    return realdata.penguin_missing_scores()


@st.cache_data(show_spinner=False)
def cached_train(name: str, features: tuple[str, ...]):
    return realdata.train_table_model(name, features=list(features))


@st.cache_data(show_spinner=False)
def cached_penguins():
    return realdata.penguin_confusion()


@st.cache_data(show_spinner=False)
def cached_bikes():
    return realdata.bike_regression()


# ---------------------------------------------------------------------------
ui.beat("hook", "Goodbye, Flatland.")

st.markdown(
    """
Every dataset so far had **exactly two columns**, so we could draw the whole thing.

Real data has twenty columns. Some columns are numbers. Some are words. Some cells are
blank. You cannot draw the whole table on one neat graph.

So the question changes. Instead of *can I draw the whole world?*, you ask:
what kind of column is this, what is missing, and what would a boring guess score before
my model learns anything?
"""
)

penguins = datasets.load_table("penguins")
st.markdown("Here is a real table. Let it look messy for a second.")
st.dataframe(penguins.head(8), hide_index=True, use_container_width=True)
st.caption(f"penguins has {len(penguins)} rows and {len(penguins.columns)} columns. That is already too many for Flatland.")

ui.little_kid_corner(
    "Imagine sorting a huge box of trading cards. You cannot hold every card in the air at once. "
    "You look at one clue at a time: colour, power, height, team, missing sticker. Data works the same way."
)

ui.mermaid(
    """
graph LR
    A[Raw table] --> B[Encode words]
    B --> C[Fill or drop gaps]
    C --> D[Split rows]
    D --> E[Train model]
    E --> F[Score against baseline]
""",
    height=250,
)

st.markdown(
    """
Notice the order. We do not train first and clean later. The model can only learn from
the table we hand it, so every cleanup choice becomes part of the experiment.
"""
)

# ---------------------------------------------------------------------------
ui.beat("byhand", "Three chores every real table gives you.")

st.subheader("1. Word columns have to become yes/no columns")
before, after = realdata.weather_one_hot_demo()
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Before**")
    st.dataframe(before, hide_index=True, use_container_width=True)
with right:
    st.markdown("**After**")
    st.dataframe(after, hide_index=True, use_container_width=True)

st.markdown(
    """
A model does arithmetic. It can compare `4 > 1`, multiply by 3, and split a tree at
`weather < 2.5`. It cannot multiply by the word `storm`.

Numbering clear=1, misty=2, rain=3, storm=4 sneaks in a fake ruler. A straight-line model
may treat storm as four times clear. A tree may ask `weather <= 2.5`, which groups clear
and misty against rain and storm. That only makes sense if the order is real.

The yes/no columns avoid the fake ruler. `weather = storm` is either 0 or 1, and it is not
larger, warmer, or halfway between anything.
"""
)
ui.jargon("one-hot encoding", "Turn each possible word into its own 0-or-1 column.")

st.subheader("2. Blanks are not free")
missing = realdata.penguin_missing_rows()
st.dataframe(missing.head(12), hide_index=True, use_container_width=True)
st.caption(f"Penguins has {len(missing)} rows with at least one blank cell.")
st.dataframe(cached_missing_scores(), hide_index=True, use_container_width=True)
ui.careful(
    "Dropping blank rows says, 'these penguins never existed.' That can erase the exact "
    "kind of penguin your measuring tools had trouble with.\n\n"
    "Filling blanks with an average tells a different lie: 'this missing beak was perfectly "
    "ordinary.' That keeps the row, but it hides the weirdness. Pick the lie you understand."
)

st.subheader("3. A boring guess comes first")
st.markdown(
    """
Before you are impressed by 82%, ask what the boring guess gets for free.

If 80 out of 100 mushrooms are safe, a model that says **safe every time** scores 80%.
A fancy model at 82% only bought two extra correct answers. A fancy model at 96% bought
sixteen. Same scorecard, very different story.
"""
)
st.dataframe(cached_scores(), hide_index=True, use_container_width=True)
ui.jargon("baseline", "A boring score from a model that does not learn. For classes, it says the most common answer every time.")

# ---------------------------------------------------------------------------
ui.beat("seeit", "A real table explorer.")

choice = st.selectbox("Pick a bundled table", TABLES, format_func=dataset_label, key="ch09_explorer")
overview = cached_overview(choice)
st.markdown(f"**Target:** `{overview['target']}` — the column we try to predict.")

cols = st.columns(3)
cols[0].metric("rows", overview["rows"])
cols[1].metric("columns", overview["columns"])
cols[2].metric("target", overview["target"])

st.markdown("**First rows**")
st.dataframe(overview["head"], hide_index=True, use_container_width=True)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Column kinds**")
    st.dataframe(overview["dtypes"], hide_index=True, use_container_width=True)
with right:
    st.markdown("**Missing cells**")
    st.dataframe(overview["missing"], hide_index=True, use_container_width=True)

st.markdown("**How lopsided is the target?**")
st.dataframe(overview["target_counts"], hide_index=True, use_container_width=True)
st.markdown(
    "Look first for giant piles. A lopsided target is where accuracy starts lying, because "
    "the most common answer may already score well."
)

# ---------------------------------------------------------------------------
ui.beat("play", "The Feature Draft.")

st.markdown(
    """
Pick the columns your model is allowed to use. This is not a guessing contest where the
computer is always right; it is a draft.

You choose a team of clues, train, then compare your hunch with what the model leaned on.
If one column dominates the bar chart, ask whether it is a real clue or a sneaky shortcut.
"""
)

draft_table = st.selectbox("Draft dataset", TABLES, index=2, format_func=dataset_label, key="ch09_draft_table")
df = datasets.load_table(draft_table)
target = datasets.target_of(draft_table)
options = [c for c in df.columns if c != target]
default_features = realdata.suggested_features(draft_table)
selected = st.multiselect("Your feature team", options, default=default_features, key="ch09_features")

if not selected:
    st.warning("Pick at least one column so the model has something to look at.")
else:
    result = cached_train(draft_table, tuple(selected))
    c1, c2, c3 = st.columns(3)
    c1.metric("baseline", f"{result['baseline_score']:.1%}")
    c2.metric("your model", f"{result['model_score']:.1%}")
    c3.metric("rows used", result["rows_used"])

    if "ch09_board" not in st.session_state:
        st.session_state.ch09_board = []
    if st.button("Add this attempt to my leaderboard", type="primary"):
        st.session_state.ch09_board.append(
            {
                "dataset": draft_table,
                "columns": ", ".join(selected),
                "score": result["model_score"],
                "beats baseline by": result["model_score"] - result["baseline_score"],
            }
        )
    if st.session_state.ch09_board:
        board = pd.DataFrame(st.session_state.ch09_board).sort_values("score", ascending=False)
        st.dataframe(board, hide_index=True, use_container_width=True)

    st.markdown("**What mattered most?**")
    st.bar_chart(result["importances"].set_index("column"))
    st.markdown(
        "Look for one tall bar. That column carried most of the model's decision, so it "
        "deserves a human sanity check."
    )

    if draft_table == "monsters":
        with st.expander("Reveal the monster world's secret rule"):
            st.markdown(f"`{datasets.MONSTER_SECRET_RULE}`")
            st.markdown("Did your column team find `attack`, `magic`, and `speed`?")

# ---------------------------------------------------------------------------
ui.beat("forreal", "Two real checks you will use again.")

st.markdown("**Classification check: what gets mixed up with what?**")
penguin_info = cached_penguins()
fig, ax = ui.figure(5.5, 4.5)
confusion_grid(penguin_info["cm"], labels=penguin_info["labels"], ax=ax, title="Penguin species confusion")
ui.show(fig)
st.caption(f"Test accuracy: {penguin_info['score']:.1%}. The off-diagonal cells are the mix-ups.")

xcol, ycol = penguin_info["top"]
fig, ax = ui.figure(6, 4.6)
for species, part in penguin_info["data"].groupby("species"):
    ax.scatter(part[xcol], part[ycol], label=species, s=45, edgecolors="white", linewidths=0.8)
ax.set_xlabel(xcol)
ax.set_ylabel(ycol)
ax.set_title("The two most important measurements")
ax.legend(fontsize=9)
ui.show(fig)
st.markdown(
    "Look for species whose dots overlap. The confusion matrix above and this scatter plot "
    "are telling the same story: mistakes usually live where the measurements overlap."
)

st.markdown("**Regression check: predicted versus actual.**")
bike = cached_bikes()
rows = bike["rows"]
fig, ax = ui.figure(5.8, 5.0)
ax.scatter(rows["rentals"], rows["predicted"], s=35, alpha=0.85, edgecolors="white", linewidths=0.7)
lo = min(rows["rentals"].min(), rows["predicted"].min())
hi = max(rows["rentals"].max(), rows["predicted"].max())
ax.plot([lo, hi], [lo, hi], color=ACCENT, linestyle="--", label="perfect")
ax.set_xlabel("actual rentals")
ax.set_ylabel("predicted rentals")
ax.set_title("The regression plot you should always draw")
ax.legend()
ui.show(fig)
st.metric("bike model R²", f"{bike['result']['model_score']:.2f}", f"baseline {bike['result']['baseline_score']:.2f}")
st.markdown(
    "Look for points far from the dashed perfect line. Those are not random embarrassments; "
    "they are clues. Here, the dates matter because bike rentals grew during the years this table covers."
)
st.dataframe(bike["worst"], hide_index=True, use_container_width=True)

st.code(
    """
X = pandas.get_dummies(table[feature_columns])
y = table[target_column]
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
""".strip(),
    language="python",
)

# ---------------------------------------------------------------------------
ui.beat("challenge")

st.markdown(
    """
1. Find the **smallest set of columns** that still beats 95% of the full model's score.
2. Find a column that makes the model worse. Weird columns are allowed.
3. On mushrooms, find the single question that gets you furthest.
4. On bikes, find one wrong prediction and explain it using the date or weather.
5. 🧸 **Little Kid Corner:** Sort real cards or toys using three clues. Then hide one clue. Which clue did you miss most?
"""
)

ui.worksheet_link(9)
