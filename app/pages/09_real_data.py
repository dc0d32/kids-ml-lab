"""Chapter 09 · Real Data, Real Mess."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kidsml import datasets, lesson, realdata
from kidsml.plots import ACCENT, confusion_grid

lesson.begin(9)


TABLES = list(realdata.REAL_TABLES)


def dataset_label(name: str) -> str:
    return f"{name} — {datasets.blurb_of(name)}"


def representative_preview(name: str) -> pd.DataFrame:
    df = datasets.load_table(name)
    target = datasets.target_of(name)
    if pd.api.types.is_numeric_dtype(df[target]):
        return df.head(5)
    return df.groupby(target, group_keys=False).head(2).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_overview(name: str):
    overview = realdata.table_overview(name)
    overview["head"] = representative_preview(name)
    return overview


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


@lesson.step("Goodbye, Flatland", beat="hook")
def _():
    lesson.say(
        """
Every dataset so far had **exactly two columns**. Lovely Flatland! We could draw the whole world on one plot.

Real data kicks the door open: twenty columns, numbers, words, blank cells. The table is a
cabinet full of drawers, not one neat graph.
"""
    )

    penguins = datasets.load_table("penguins")
    # Take a few of each species rather than the first eight rows. The file is sorted by
    # species, so head(8) is eight Adelie — a misleading first look at a three-way problem.
    sample = penguins.groupby("species", group_keys=False).head(3)
    st.dataframe(sample, hide_index=True, width="stretch")
    st.caption(f"penguins has {len(penguins)} rows and {len(penguins.columns)} columns. Flatland has failed a full audit, no cap.")
    lesson.look_for("word columns, number columns, and blank-looking cells — and three different species, which is what we are being asked to tell apart.")
    lesson.kid_corner(
        "Imagine sorting a huge box of trading cards. You cannot hold every card in the air at once. "
        "You look at one clue at a time: colour, power, height, team, missing sticker."
    )


@lesson.step("The real-data pipeline", beat="hook")
def _():
    lesson.say("The question changes: inspect the column, hunt the blanks, and ask what a boring guess scores before the model wakes up.")
    lesson.mermaid(
        """
graph TD
    A[Raw table] --> B[Encode words]
    B --> C[Fill or drop gaps]
    C --> D[Split rows]
    D --> E[Train model]
    E --> F[Score against baseline]
""",
    )
    lesson.look_for("the order. Clean first, train after. No mud in the engine.")
    lesson.say("The model eats the table we hand it. Every cleanup choice becomes part of the experiment.")


@lesson.step("Words need yes-or-no columns", beat="byhand")
def _():
    lesson.say(
        """
A model does arithmetic. It can compare `4 > 1`, multiply by 3, and split a tree at
`weather < 2.5`.

It cannot multiply by the word `storm`, and numbering the words sneaks in a fake ruler.
"""
    )

    before, after = realdata.weather_one_hot_demo()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Before**")
        st.dataframe(before, hide_index=True, width="stretch")
    with right:
        st.markdown("**After**")
        st.dataframe(after, hide_index=True, width="stretch")
    lesson.look_for("how `storm` becomes its own yes/no column instead of a bigger number than `clear`.")
    lesson.jargon("one-hot encoding", "Turn each possible word into its own 0-or-1 column.")


@lesson.step("Blanks are not free", beat="byhand")
def _():
    lesson.say("Dropping blank rows makes those penguins vanish. Filling blanks with an average puts a cardboard penguin in the gap.")
    missing = realdata.penguin_missing_rows()
    st.dataframe(missing.head(12), hide_index=True, width="stretch")
    st.caption(f"Penguins has {len(missing)} rows with at least one blank cell.")
    st.dataframe(cached_missing_scores(), hide_index=True, width="stretch")
    lesson.look_for("whether the score changes when blanks are dropped or filled. Cleanup choices are model choices.")
    lesson.careful(
        "Dropping can erase the exact kind of penguin your measuring tools had trouble with. "
        "Filling keeps the row, but it hides the weirdness. Pick the lie you understand."
    )


@lesson.step("A boring guess comes first", beat="byhand")
def _():
    guess = lesson.predict(
        "If 80 out of 100 mushrooms are safe, what does the boring `always safe` guess score?",
        ["20%", "50%", "80%"],
        correct=2,
        why="It nails the big safe pile and learns zero about dangerous mushrooms. Shiny score; hollow mushroom.",
        key="ch09_baseline_guess",
    )
    if guess is None:
        return

    lesson.say(
        """
Before 82% gets a parade, ask what the boring guess gets for free.

If 80 out of 100 mushrooms are safe, a model that says **safe every time** scores 80%.
A fancy model at 82% moved only two mushrooms. Baseline first!

The score table below uses four bundled datasets: **penguins** are birds with island,
beak, flipper, weight, and sex columns, predicting species. **Mushrooms** are mushroom
descriptions such as cap shape, smell, and gill clues, predicting edible or poisonous.
**Monsters** are trading-card creatures with element, home, and battle stats, predicting
whether each one is a boss. **Bikes** are daily weather rows, predicting the number of rentals.
"""
    )
    st.dataframe(cached_scores(), hide_index=True, width="stretch")
    lesson.look_for("datasets where the baseline already towers. Accuracy starts on that platform, not on the floor.")
    lesson.jargon("baseline", "A boring score from a model that does not learn. For classes, it says the most common answer every time.")


@lesson.step("A real table explorer", beat="seeit")
def _():
    choice = st.selectbox("Pick a bundled table", TABLES, format_func=dataset_label, key="ch09_explorer_table")
    overview = cached_overview(choice)
    lesson.say(f"Target: `{overview['target']}` — the column we try to predict.")

    cols = st.columns(3)
    cols[0].metric("rows", overview["rows"])
    cols[1].metric("columns", overview["columns"])
    cols[2].metric("target", overview["target"])
    st.dataframe(overview["head"], hide_index=True, width="stretch")
    lesson.look_for("how wide the table is. For class targets, this preview grabs rows from each answer pile instead of trusting the file order.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Column kinds**")
        st.dataframe(overview["dtypes"], hide_index=True, width="stretch")
    with right:
        st.markdown("**Missing cells**")
        st.dataframe(overview["missing"], hide_index=True, width="stretch")
    st.markdown("**How lopsided is the target?**")
    st.dataframe(overview["target_counts"], hide_index=True, width="stretch")
    lesson.look_for("giant target piles. A lopsided target is where accuracy starts lying.")


@lesson.step("Draft your feature team", beat="play")
def _():
    lesson.say(
        """
Now meet the monsters table before the model touches it. Each row is a trading-card
creature. The columns include words like `element` and `home`, plus stats like `attack`,
`magic`, and `speed`. The target is `is_boss`: yes or no.
"""
    )
    guess = lesson.predict(
        "Before training, which clues do you expect the monsters model to lean on most?",
        ["colour and size", "attack, magic, and speed", "name and row number"],
        correct=1,
        why="The monster world's hidden rule lives in attack, magic, and speed. Now see whether your drafted clues sniff it out.",
        key="ch09_feature_hunch",
    )
    if guess is None:
        return

    lesson.say("Pick the columns your model is allowed to use. You draft a team of clues, then the bar chart shows who carried the ball.")
    draft_table = st.selectbox("Draft dataset", TABLES, index=2, format_func=dataset_label, key="ch09_draft_table")
    df = datasets.load_table(draft_table)
    target = datasets.target_of(draft_table)
    options = [c for c in df.columns if c != target]
    default_features = realdata.suggested_features(draft_table)
    selected = st.multiselect("Your feature team", options, default=default_features, key="ch09_feature_team")
    if not selected:
        st.warning("Pick at least one column so the model has something to look at.")
        return

    result = cached_train(draft_table, tuple(selected))
    c1, c2, c3 = st.columns(3)
    c1.metric("baseline", f"{result['baseline_score']:.1%}")
    c2.metric("your model", f"{result['model_score']:.1%}")
    c3.metric("rows used", result["rows_used"])
    lesson.look_for("whether your score beats the baseline by enough to matter.")


@lesson.step("Reveal what mattered", beat="play")
def _():
    lesson.say(
        """
Now pop the hood and look at feature importances. Each bar is how much the trained forest
leaned on that column while making splits.

If one column towers, ask whether it is a real clue or a sneaky shortcut.
"""
    )
    lesson.jargon("feature importance", "A score for how much a trained model leaned on each column.")
    draft_table = st.selectbox("Feature-importance dataset", TABLES, index=2, format_func=dataset_label, key="ch09_importance_table")
    df = datasets.load_table(draft_table)
    target = datasets.target_of(draft_table)
    options = [c for c in df.columns if c != target]
    default_features = realdata.suggested_features(draft_table)
    selected = st.multiselect("Feature team", options, default=default_features, key="ch09_importance_features")
    if not selected:
        st.warning("Pick at least one column so the model has something to look at.")
        return

    result = cached_train(draft_table, tuple(selected))
    st.bar_chart(result["importances"].set_index("column"))
    lesson.look_for("one tall bar. That column is doing the heavy lifting; check whether it is a real clue or a shortcut.")
    if draft_table == "monsters":
        st.markdown(f"Monster secret rule: `{datasets.MONSTER_SECRET_RULE}`")
        st.caption("Did your column team find `attack`, `magic`, and `speed`?")


@lesson.step("Keep your own leaderboard", beat="play")
def _():
    lesson.say("Try a column team, then save it. The leaderboard is for your experiments, not a universal truth.")
    draft_table = st.selectbox("Leaderboard dataset", TABLES, index=2, format_func=dataset_label, key="ch09_board_table")
    df = datasets.load_table(draft_table)
    target = datasets.target_of(draft_table)
    options = [c for c in df.columns if c != target]
    selected = st.multiselect("Leaderboard feature team", options, default=realdata.suggested_features(draft_table), key="ch09_board_features")
    if not selected:
        st.warning("Pick at least one column so the model has something to look at.")
        return

    result = cached_train(draft_table, tuple(selected))
    if "ch09_board" not in st.session_state:
        st.session_state.ch09_board = []
    if st.button("Add this attempt to my leaderboard", type="primary", key="ch09_add_board"):
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
        st.dataframe(board, hide_index=True, width="stretch")
        lesson.look_for("attempts that beat the baseline by the most, not only the highest raw score.")


@lesson.step("What gets mixed up?", beat="forreal")
def _():
    lesson.say(
        """
A classification check asks: what gets mixed up with what? This penguin model predicts
species from one row per bird, using island, beak, flipper, weight, and sex columns.

In the confusion matrix, rows are the real species and columns are what the model guessed.
The diagonal is correct; off-diagonal cells are mix-ups.
"""
    )
    lesson.jargon("confusion matrix", "A table where rows are real answers, columns are model guesses, and off-diagonal cells are mistakes.")
    penguin_info = cached_penguins()
    fig, ax = lesson.figure(5.5, 4.5)
    confusion_grid(penguin_info["cm"], labels=penguin_info["labels"], ax=ax, title="Penguin species confusion")
    lesson.show(fig)
    lesson.look_for("off-diagonal cells. Those are the mix-ups, not random decoration.")
    st.caption(f"Test accuracy: {penguin_info['score']:.1%}.")

    xcol, ycol = penguin_info["top"]
    fig, ax = lesson.figure(6, 4.6)
    for species, part in penguin_info["data"].groupby("species"):
        ax.scatter(part[xcol], part[ycol], label=species, s=45, edgecolors="white", linewidths=0.8)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("The two most important measurements")
    ax.legend(fontsize=9)
    lesson.show(fig)
    lesson.look_for("species whose dots overlap. The matrix and scatter plot are telling the same story.")


@lesson.step("Predicted versus actual", beat="forreal")
def _():
    lesson.say(
        """
A regression check slams predicted versus actual onto one picture. The bikes table has
one row per day, weather columns such as temperature, humidity, wind, and season, and a
number target: how many bikes were rented that day. The dashed line is perfection.
"""
    )
    bike = cached_bikes()
    rows = bike["rows"]
    fig, ax = lesson.figure(5.8, 5.0)
    ax.scatter(rows["rentals"], rows["predicted"], s=35, alpha=0.85, edgecolors="white", linewidths=0.7)
    lo = min(rows["rentals"].min(), rows["predicted"].min())
    hi = max(rows["rentals"].max(), rows["predicted"].max())
    ax.plot([lo, hi], [lo, hi], color=ACCENT, linestyle="--", label="perfect")
    ax.set_xlabel("actual rentals")
    ax.set_ylabel("predicted rentals")
    ax.set_title("The regression plot you should always draw")
    ax.legend()
    lesson.show(fig)
    lesson.look_for("points far from the dashed perfect line. Those dots are clues with boots on.")
    st.metric("bike model R²", f"{bike['result']['model_score']:.2f}", f"baseline {bike['result']['baseline_score']:.2f}")
    st.caption("R² is a number-prediction score: 1.0 is perfect, and 0.0 is about as useful as guessing the average.")
    st.dataframe(bike["worst"], hide_index=True, width="stretch")


@lesson.step("The code shape", beat="forreal")
def _():
    lesson.say("The real-data program is not magic. It picks columns, encodes them, trains, and scores on hidden rows.")
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


@lesson.step("Check yourself", beat="challenge")
def _():
    lesson.workbook()


@lesson.step("Go draft better clues", beat="challenge")
def _():
    lesson.say(
        """
1. Find the **smallest set of columns** that still beats 95% of the full model's score.
2. Find a column that makes the model worse. Weird columns are allowed.
3. On mushrooms, find the single question that gets you furthest.
4. On bikes, find one wrong prediction and explain it using the date or weather.
5. 🧸 **Little Kid Corner:** Sort real cards or toys using three clues. Then hide one clue. Which clue did you miss most?
"""
    )


lesson.finish()
