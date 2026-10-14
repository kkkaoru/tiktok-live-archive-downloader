"""Synthetic prerequisite diagnostics do not issue real calibration approvals."""

import copy

import pytest

from replay.caption_prerequisites import (
    observation_requests,
    require_same_geometry,
    template_differences,
)
from replay.core import ReplayError

TEMPLATE = (
    "<ozml><scene><sceneSettings><presetPath>old</presetPath></sceneSettings>"
    '<font size="12">face</font></scene></ozml>'
)


@pytest.fixture
def identity() -> dict[str, object]:
    return {
        "font": "font-sha",
        "variation": "wght=700",
        "native": "native-sha",
        "code": "code-sha",
        "model": "none",
        "policy": "policy-sha",
        "resource": "preset-sha",
    }


def test_resource_path_only_is_not_geometry(identity: dict[str, object]) -> None:
    selected = TEMPLATE.replace(">old<", ">new<")
    assert template_differences(TEMPLATE, selected) == []
    require_same_geometry(TEMPLATE, selected, old_identity=identity, new_identity=identity)


@pytest.mark.parametrize(
    "key", ["font", "variation", "native", "code", "model", "policy", "resource"]
)
def test_changed_identity_rejected(identity: dict[str, object], key: str) -> None:
    changed = {**identity, key: "changed"}
    with pytest.raises(ReplayError, match="identity changed"):
        require_same_geometry(TEMPLATE, TEMPLATE, old_identity=identity, new_identity=changed)


def test_missing_identity_rejected(identity: dict[str, object]) -> None:
    with pytest.raises(ReplayError, match="Incomplete"):
        require_same_geometry(TEMPLATE, TEMPLATE, old_identity={}, new_identity=identity)
    identity["variation"] = None
    with pytest.raises(ReplayError, match="unobserved"):
        require_same_geometry(TEMPLATE, TEMPLATE, old_identity=identity, new_identity=identity)


@pytest.mark.parametrize(
    "selected",
    [
        TEMPLATE.replace('size="12"', 'size="13"'),
        TEMPLATE.replace("face", "other"),
        TEMPLATE.replace("face", " face "),
        TEMPLATE.replace("</scene>", "<extra/></scene>"),
    ],
)
def test_geometry_not_normalized(identity: dict[str, object], selected: str) -> None:
    with pytest.raises(ReplayError, match="Template geometry differs"):
        require_same_geometry(TEMPLATE, selected, old_identity=identity, new_identity=identity)


def test_malformed_and_large_template_rejected() -> None:
    with pytest.raises(ReplayError, match="Invalid template"):
        template_differences("<", TEMPLATE)
    with pytest.raises(ReplayError, match="budget"):
        template_differences(" " * (8 * 1024 * 1024 + 1), TEMPLATE)
    assert template_differences("<ozml><a/><a/></ozml>", "<ozml><a/><b/></ozml>") == [
        "/ozml/a[1]/@structure"
    ]


@pytest.fixture
def metadata() -> dict[str, object]:
    return {
        "stateCount": 3,
        "visibleFrames": 3,
        "states": [
            {
                "part": "a",
                "stateIndex": 0,
                "globalStart": 1,
                "globalEnd": 2,
                "sourceSupport": [
                    {
                        "sourceStartSeconds": "1/10",
                        "sourceEndSeconds": "1/5",
                        "observedSourcePTSSeconds": ["9/100", "3/20"],
                    }
                ],
            },
            {
                "part": "a",
                "stateIndex": 1,
                "globalStart": 2,
                "globalEnd": 3,
                "sourceSupport": [
                    {
                        "sourceStartSeconds": "1/5",
                        "sourceEndSeconds": "3/10",
                        "observedSourcePTSSeconds": ["3/20", "1/4"],
                    }
                ],
            },
            {
                "part": "b",
                "stateIndex": 0,
                "globalStart": 4,
                "globalEnd": 5,
                "sourceSupport": [
                    {
                        "sourceStartSeconds": "1",
                        "sourceEndSeconds": "11/10",
                        "observedSourcePTSSeconds": ["1"],
                    }
                ],
            },
        ],
    }


def test_only_actual_support_requested(metadata: dict[str, object]) -> None:
    before = copy.deepcopy(metadata)
    result = observation_requests(metadata)
    assert result["windows"] == [
        {
            "startSeconds": "1/10",
            "endSecondsExclusive": "3/10",
            "actualPTSSeconds": ["9/100", "3/20", "1/4"],
        },
        {"startSeconds": "1", "endSecondsExclusive": "11/10", "actualPTSSeconds": ["1"]},
    ]
    assert result["constraintsGenerated"] is False
    assert result["sourceIdentity"] is None
    assert metadata == before


@pytest.mark.parametrize("patch", [{"states": []}, {"stateCount": 2}, {"visibleFrames": 8}])
def test_incomplete_coverage_rejected(
    metadata: dict[str, object], patch: dict[str, object]
) -> None:
    with pytest.raises(ReplayError):
        observation_requests({**metadata, **patch})


@pytest.mark.parametrize(
    "support",
    [
        [],
        [{"sourceStartSeconds": "1/10", "sourceEndSeconds": "1/5", "observedSourcePTSSeconds": []}],
        [
            {
                "sourceStartSeconds": "bad",
                "sourceEndSeconds": "1/5",
                "observedSourcePTSSeconds": ["1/10"],
            }
        ],
    ],
)
def test_missing_observations_not_invented(support: list[object]) -> None:
    with pytest.raises(ReplayError):
        observation_requests(
            {
                "stateCount": 1,
                "visibleFrames": 1,
                "states": [
                    {
                        "part": "a",
                        "stateIndex": 0,
                        "globalStart": 0,
                        "globalEnd": 1,
                        "sourceSupport": support,
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "points", [["1/5"], ["1/10", "1/10"], ["1/20", "9/100"], [None], ["1/0"], ["90000"]]
)
def test_invalid_actual_pts_rejected(points: list[object]) -> None:
    with pytest.raises(ReplayError):
        observation_requests(
            {
                "stateCount": 1,
                "visibleFrames": 1,
                "states": [
                    {
                        "part": "a",
                        "stateIndex": 0,
                        "globalStart": 0,
                        "globalEnd": 1,
                        "sourceSupport": [
                            {
                                "sourceStartSeconds": "1/10",
                                "sourceEndSeconds": "1/5",
                                "observedSourcePTSSeconds": points,
                            }
                        ],
                    }
                ],
            }
        )


def test_negative_clock_not_observed_head() -> None:
    with pytest.raises(ReplayError, match="output clock"):
        observation_requests(
            {"stateCount": 1, "visibleFrames": 0, "states": [{"globalStart": 1, "globalEnd": 1}]}
        )
