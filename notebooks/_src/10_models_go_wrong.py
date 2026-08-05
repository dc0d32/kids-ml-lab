# %% [markdown]
# # Chapter 10 · Where Models Go Wrong
#
# ### Bias, leakage, and being confidently wrong.
#
# *Part 2 · Escaping Flatland*
#
# ---
#
# This chapter is short, blunt, and important. It is about honesty, not fancy algorithms.
# Chapter 00 warned that a model always answers; now we learn when that confident answer
# should make your dashboard blink red.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from kidsml import realdata, workbook
from kidsml.plots import WARM, confusion_grid, scatter_2d, use_house_style

use_house_style()

# %% [markdown]
# ## 🎣 Start here
#
# A model that is right **99%** of the time can still be a trophy made of fog.
#
# A model that scores brilliantly can be cheating without anybody noticing.
#
# This chapter shows the trapdoor under the magic trick. Once you know the trick,
# suspicious scores start flashing like hazard lights. Good.
#
# ```mermaid
# graph TD
#     A[Shiny score] --> B{What feels wrong?}
#     B -->|rare answer| C[Useless accuracy]
#     B -->|too perfect| D[Leakage]
#     B -->|hurts one group| E[Copied unfair history]
#     B -->|far away| F[Outside its world]
# ```
#
# Use this flowchart like a detective card. A score is not the end of the investigation;
# it is the first muddy footprint.
#
# > 🧸 **Little Kid Corner** — If a smoke alarm never beeps, it is quiet almost all day.
# > That does not make it a good smoke alarm. The important question is what happens on
# > the one day with smoke.

# %% [markdown]
# ## ✏️ Work it out
#
# Imagine 1000 people. The model says **sick** for 10 of them.
#
# - 8 were really sick. Good catch.
# - 2 were healthy. Scary false alarm.
# - 40 sick people were missed.
# - 950 healthy people were left alone.

# %%
worked = pd.DataFrame(
    [[8, 40], [2, 950]],
    index=["really sick", "really healthy"],
    columns=["model said sick", "model said healthy"],
)
worked

# %% [markdown]
# This 2-by-2 box is a **confusion matrix**. It counts the four things that can happen:
# caught sick people, missed sick people, false alarms, and correctly ignored healthy people.
#
# Precision asks, "When the alarm rang, how often was it right?" Recall asks, "Of all the
# sick people, how many did the alarm catch?"

# %%
metrics = realdata.metrics_from_counts(tp=8, fp=2, fn=40, tn=950)
print("accuracy:", round(metrics["accuracy"] * 100, 1), "%")
print("precision:", round(metrics["precision"] * 100, 1), "%")
print("recall:", round(metrics["recall"] * 100, 1), "%")

# %% [markdown]
# Accuracy looks great. Recall is awful.
#
# Accuracy = `(8 + 950) / 1000 = 95.8%`.
#
# Precision = `8 / (8 + 2) = 80%`.
#
# Recall = `8 / (8 + 40) = 16.7%`.
#
# There are 48 sick people, but the model only caught 8 of them. Accuracy counts the 950
# healthy people it left alone, so the big healthy pile hides the medical failure under a rug.
#
# That is why rare problems need more than accuracy. If a disease appears in 1 out of 100
# people, a model can score 99% by saying **healthy** to everyone and helping nobody sick.
#
# > 📖 **Grown-ups call this:** **precision** means: of the ones it flagged, how many
# > really were?
#
# > 📖 **Grown-ups call this:** **recall** means: of the ones that really were, how many
# > did it catch?

# %% [markdown]
# ## 👀 Take a look
#
# Failure 1: the useless 99%.

# %%
print("Always say healthy accuracy:", f"{realdata.always_healthy_accuracy():.0%}")

# %%
THRESHOLD = 0.50
report = realdata.threshold_report(THRESHOLD)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
confusion_grid(report["cm"], labels=["sick", "healthy"], ax=ax, title="Threshold trade-off")
plt.show()
report["metrics"]

# %% [markdown]
# The model has a hidden worry score for each person. The threshold is the line that says,
# "above here, call it sick."
#
# Lower the line and more people cross it. Recall rises because you catch more sick people,
# but precision can fall because more healthy people get swept in too. Raise the line and
# you bother fewer healthy people, but you miss more sick ones.
#
# For a smoke alarm, you may accept more false alarms. For a spam filter, eating real mail
# is painful. Same slider, different stakes.

# %% [markdown]
# ## 🎛️ Your turn
#
# Failure 2: the model cheated.

# %%
realdata.leakage_scores()

# %% [markdown]
# First we celebrate. Then we ask why the score is suspiciously perfect.
#
# Perfect can happen on tiny toy worlds, but real messy data that hits 100% makes the
# leaked-answer alarm clang. A column like `was_approved_last_time` or `future_total` lets
# the model peek at the test.
#
# > 📖 **Grown-ups call this:** **leakage** means a column lets the answer sneak into the
# > features, so the model is not learning the real pattern.
#
# A hospital model once looked clever because it learned which scanner machine was used.
# The sickest patients used the portable scanner more often. The machine was the shortcut;
# pneumonia was the real target.

# %% [markdown]
# Failure 3: unfair copies make unfair models.

# %%
bias = realdata.bias_report()
bias["data"]

# %%
print("overall score against old labels:", f"{bias['overall']:.1%}")
bias["summary"]

# %%
bias["examples"]

# %% [markdown]
# The model does not know history is unfair. It only sees examples to copy.
#
# The model is not being mean. It is copying. That is all it can do. Copy from an unfair
# stack of cards, and the unfairness comes back at scale with a confident voice.
#
# Failure 4: confidently wrong, outside its world.

# %%
far = realdata.moons_out_of_world(span=8)
fig, ax = plt.subplots(figsize=(6, 5))
img = ax.contourf(far["xx"], far["yy"], far["confidence"], levels=20, cmap="Blues", alpha=0.85)
scatter_2d(far["X"], far["y"], ax=ax, size=25)
ax.scatter([far["far"][0]], [far["far"][1]], marker="*", s=220, color=WARM, edgecolors="white", label="far away")
ax.set_title("Confidence far outside the training data")
ax.legend(fontsize=9)
fig.colorbar(img, ax=ax, label="model confidence")
plt.show()
print("far-away guess:", far["far_guess"])
print("far-away confidence:", f"{far['far_confidence']:.0%}")

# %% [markdown]
# Look at how far the star is from the training moons. The model has no built-in
# new-planet alarm. Chapter 00 warned you: a model answers anyway.

# %% [markdown]
# ## 💻 In real code
#
# A blunt honesty checklist fits in a few lines.

# %%
checklist = [
    "What is the baseline?",
    "Does the score seem too good?",
    "Does it work for everyone, or only on average?",
    "What happens when it sees something new?",
]
for item in checklist:
    print("-", item)

# %% [markdown]
# If one answer feels awkward, stop and investigate. That is not negativity; that is doing
# machine learning with the flashlight on.

# %% [markdown]
# ## 🏆 Go further
#
# 1. Build a model that scores 95% and is useless. Hint: make the answer lopsided.
# 2. Find the threshold where a spam filter starts eating real mail.
# 3. Take the Chapter 09 monsters model and find a monster it is confidently wrong about.
# 4. Make a suspiciously perfect model, then find the leaked column.
# 5. 🧸 **Little Kid Corner:** Make a pretend alarm that never rings. Count how many quiet
#    minutes it gets right. Then ask what happens during toast smoke.

# %%
workbook.render(10)

# %% [markdown]
# ---
# **Next up:** Chapter 11 · *Arrows and Grids* — matrices become space movers before neurons use them.
