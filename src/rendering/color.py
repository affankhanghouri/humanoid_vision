"""Colors for the demo renderer.

OpenCV colors are BGR.
"""


WHITE = (
    245,
    245,
    245,
)

LIGHT_TEXT = (
    215,
    225,
    235,
)

DARK_TEXT = (
    15,
    20,
    25,
)

CYAN = (
    255,
    225,
    0,
)

GREEN = (
    60,
    255,
    80,
)

YELLOW = (
    0,
    230,
    255,
)

ORANGE = (
    0,
    145,
    255,
)

MAGENTA = (
    255,
    0,
    255,
)

BLUE = (
    255,
    120,
    20,
)

RED = (
    70,
    70,
    255,
)


PALETTE = (
    CYAN,
    GREEN,
    YELLOW,
    ORANGE,
    MAGENTA,
    BLUE,
)


def color_for_entity(
    entity_id: int,
):

    return PALETTE[
        entity_id
        % len(PALETTE)
    ]