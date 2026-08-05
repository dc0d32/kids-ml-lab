"""Chapter 11 workbook · Arrows and Grids."""

from kidsml import linalg as la
from kidsml.workbook import Question, Workbook

length_6_8 = la.magnitude([6, 8])
dot_small = la.dot([2, 3], [4, 1])
dot_right_angle = la.dot([2, 0], [0, 5])
stretched = [[2, 0], [0, 3]]
stretched_result = [2 * 1 + 0 * 1, 0 * 1 + 3 * 1]
first_column = la.where_the_arrows_land([[3, -1], [2, 4]])["right arrow (1, 0)"]
area_stretch = la.area_change([[2, 0], [0, 3]])
area_flat = la.area_change([[1, 2], [2, 4]])
combined = [[2 * 0 + 1 * 1, 2 * 1 + 1 * 0], [0 * 0 + 3 * 1, 0 * 1 + 3 * 0]]

WORKBOOK = Workbook(
    chapter=11,
    title="Workbook · Arrows and grids",
    intro=(
        "Use small numbers and scrap paper. A grid means the graph-paper floor on screen. A matrix `[[a, b], [c, d]]` tells where the right and up starter arrows land, and each question matches one thing you dragged: length, agreement, grid moves, area, and chained matrices."
    ),
    questions=[
        Question(
            prompt="What is the length of the vector `[6, 8]`?",
            kind="number",
            answer=length_6_8,
            why="`[6, 8]` is the 3-4-5 triangle doubled. The legs make `sqrt(6² + 8²) = sqrt(36 + 64) = sqrt(100) = 10`.",
        ),
        Question(
            prompt="Compute the dot product `[2, 3] · [4, 1]`.",
            kind="number",
            answer=dot_small,
            why="Multiply matching parts, then add: `2×4 + 3×1 = 8 + 3 = 11`. Positive means the arrows are pushing partly the same way.",
        ),
        Question(
            prompt="Compute the dot product `[2, 0] · [0, 5]`.",
            kind="number",
            answer=dot_right_angle,
            why="`2×0 + 0×5 = 0`. These arrows meet at a right angle, so neither one casts any same-way shadow on the other.",
        ),
        Question(
            prompt="Apply `[[2, 0], [0, 3]]` to `[1, 1]`. What is the new x value?",
            kind="number",
            answer=stretched_result[0],
            why="The first row builds the new x value: `2×1 + 0×1 = 2`. Sideways movement gets doubled.",
        ),
        Question(
            prompt="Apply `[[2, 0], [0, 3]]` to `[1, 1]`. What is the new y value?",
            kind="number",
            answer=stretched_result[1],
            why="The second row builds the new y value: `0×1 + 3×1 = 3`. Up-down movement gets tripled.",
        ),
        Question(
            prompt="Under `[[3, -1], [2, 4]]`, where does `(1, 0)` land? Give the x value.",
            kind="number",
            answer=float(first_column[0]),
            why="The arrow `(1, 0)` lands on the first column of the matrix. Here that landing pad is `[3, 2]`, so the x value is 3.",
        ),
        Question(
            prompt="Under `[[3, -1], [2, 4]]`, where does `(1, 0)` land? Give the y value.",
            kind="number",
            answer=float(first_column[1]),
            why="Keep reading the first column: `[3, 2]`. Same landing pad, y value 2.",
        ),
        Question(
            prompt="What is the area multiplier of `[[2, 0], [0, 3]]`?",
            kind="number",
            answer=area_stretch,
            why="A 1-by-1 square gets stretched to 2 wide and 3 tall, so its area becomes `2×3 = 6`. The determinant says the same thing.",
        ),
        Question(
            prompt="What is the area multiplier of `[[1, 2], [2, 4]]`?",
            kind="number",
            answer=area_flat,
            why="The second column is twice the first, so both starter arrows land on the same line. The square pancakes into a line, and a line has area 0.",
        ),
        Question(
            prompt="Multiply `[[2, 1], [0, 3]] @ [[0, 1], [1, 0]]`. What is the top-left number?",
            kind="number",
            answer=combined[0][0],
            why="Top row with left column: `2×0 + 1×1 = 1`. Matrix multiplication fuses two grid moves into one ticket.",
        ),
        Question(
            prompt="For that same product, what is the top-right number?",
            kind="number",
            answer=combined[0][1],
            why="Top row with right column: `2×1 + 1×0 = 2`. That number helps pin down where the second starter arrow lands after both moves.",
        ),
        Question(
            prompt="For that same product, what is the bottom-left number?",
            kind="number",
            answer=combined[1][0],
            why="Bottom row with left column: `0×0 + 3×1 = 3`. That is the new y landing for the first starter arrow.",
        ),
        Question(
            prompt="For that same product, what is the bottom-right number?",
            kind="number",
            answer=combined[1][1],
            why="Bottom row with right column: `0×1 + 3×0 = 0`. Snap the four numbers together: `[[1, 2], [3, 0]]`.",
        ),
        Question(
            prompt="Why would stacking ten layers with no squish be a waste of time?",
            kind="open",
            why="Because all ten linear moves can be multiplied into one combined matrix. Without a squish, extra layers add no new bending power; they are a longer road to the same linear move.",
        ),
    ],
    kid_corner=(
        "Make shadows with a torch and your hand. Turn your hand until the shadow looks wide, skinny, or strange. Then stretch a drawing on a balloon. That is what this chapter's grids are doing on screen."
    ),
)
