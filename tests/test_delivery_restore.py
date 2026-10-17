import json
from pathlib import Path

import pytest

from replay import delivery_restore, disk_budget
from replay.core import ReplayError
from replay.stages import Stage, run_stage


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def workflow(path: Path, inputs: list[str], outputs: list[str]) -> Path:
    return write(path, {"stages": [{"name": "make", "inputs": inputs, "outputs": outputs}]})


def test_a_workflow_plan_lists_its_paths_and_json_outputs(tmp_path: Path) -> None:
    movie = str(tmp_path / "part.mov")
    response = str(tmp_path / "resp.json")
    plan = delivery_restore.load_plan(
        workflow(tmp_path / "join.json", ["in.wav"], [movie, response])
    )
    assert plan.state == tmp_path / "join-state"
    assert plan.outputs == (tmp_path / "resp.json",)
    assert plan.media == frozenset({"in.wav", movie, response})


def test_a_job_plan_lists_its_single_output(tmp_path: Path) -> None:
    path = write(tmp_path / "final-job.json", {"jobs": [{"inputs": ["a.mov"], "output": "b.mov"}]})
    plan = delivery_restore.load_plan(path)
    assert plan.media == frozenset({"a.mov", "b.mov"})
    assert plan.outputs == ()


def test_a_render_response_names_its_movie(tmp_path: Path) -> None:
    response = write(tmp_path / "render.response.json", {"outputPath": "/m/part-0001-master.mov"})
    path = workflow(tmp_path / "render-0-0000.json", [], [str(response)])
    assert delivery_restore.load_plan(path).media == frozenset(
        {str(response), "/m/part-0001-master.mov"}
    )


def test_an_unreadable_render_response_names_no_movie(tmp_path: Path) -> None:
    missing = str(tmp_path / "render.response.json")
    path = workflow(tmp_path / "render-0-0000.json", [], [missing])
    assert delivery_restore.load_plan(path).media == frozenset({missing})


def test_a_response_without_an_output_path_names_no_movie(tmp_path: Path) -> None:
    response = write(tmp_path / "render.response.json", {"frameCount": 3})
    path = workflow(tmp_path / "render-0-0000.json", [], [str(response)])
    assert delivery_restore.load_plan(path).media == frozenset({str(response)})


def test_an_oversized_plan_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path / "join-workflow.json", {"stages": []})
    monkeypatch.setattr(delivery_restore, "MAX_JSON_BYTES", 1)
    with pytest.raises(ReplayError, match="size limit"):
        delivery_restore.load_plan(path)


def test_plans_find_named_workflows_and_render_plans(tmp_path: Path) -> None:
    work = tmp_path / "work"
    workflow(work / "join-workflow.json", [], [])
    workflow(work / "composition" / "render-0-0000.json", [], [])
    workflow(work / "composition" / "render-1-0000.json", [], [])
    found = delivery_restore.plans(work, work / "composition")
    assert [plan.path.name for plan in found] == [
        "join-workflow.json",
        "render-0-0000.json",
        "render-1-0000.json",
    ]


def test_prepare_selects_only_plans_that_touch_retired_files(tmp_path: Path) -> None:
    work = tmp_path / "work"
    movie = tmp_path / "joined.mp4"
    workflow(work / "join-workflow.json", ["a.mov"], [str(movie)])
    workflow(work / "audio-workflow.json", ["b.wav"], ["c.wav"])
    write(
        work / "join-workflow-state" / "make.json",
        {"status": "complete", "outputs": {str(movie): "f" * 64}},
    )
    recipe = delivery_restore.prepare(work, work / "composition", [movie])
    assert recipe["retired"] == [str(movie)]
    assert recipe["plans"] == [
        {
            "plan": str(work / "join-workflow.json"),
            "state": str(work / "join-workflow-state"),
            "outputs": [],
            "digests": {str(movie): "f" * 64},
        }
    ]


def test_prepare_ignores_a_corrupt_receipt(tmp_path: Path) -> None:
    work = tmp_path / "work"
    movie = tmp_path / "joined.mp4"
    workflow(work / "join-workflow.json", [], [str(movie)])
    (work / "join-workflow-state").mkdir()
    (work / "join-workflow-state" / "make.json").write_text("not json", encoding="utf-8")
    recipe = delivery_restore.prepare(work, work / "composition", [movie])
    assert recipe["plans"] == [
        {
            "plan": str(work / "join-workflow.json"),
            "state": str(work / "join-workflow-state"),
            "outputs": [],
            "digests": {},
        }
    ]


def test_archive_moves_checkpoints_and_json_outputs_without_deleting(tmp_path: Path) -> None:
    work = tmp_path / "work"
    response = write(work / "composition" / "render.response.json", {"outputPath": "/m/p.mov"})
    plan = workflow(work / "composition" / "render-0-0000.json", [], [str(response)])
    write(plan.with_name("render-0-0000-state") / "render-000000.json", {"status": "complete"})
    recipe = delivery_restore.prepare(work, work / "composition", [Path("/m/p.mov")])
    recipe_path = delivery_restore.archive(recipe, work / "retention")
    archived = work / "retention" / "archive" / "0000-render-0-0000"
    assert not response.exists()
    assert not plan.with_name("render-0-0000-state").exists()
    assert (archived / "000000-render.response.json").read_text(encoding="utf-8") == (
        '{"outputPath": "/m/p.mov"}'
    )
    assert (archived / "render-0-0000-state" / "render-000000.json").is_file()
    assert plan.is_file()
    assert len(json.loads(recipe_path.read_text(encoding="utf-8"))["moved"]) == 2


def test_archive_refuses_to_overwrite_an_archived_file(tmp_path: Path) -> None:
    work = tmp_path / "work"
    response = write(work / "composition" / "render.response.json", {"outputPath": "/m/p.mov"})
    workflow(work / "composition" / "render-0-0000.json", [], [str(response)])
    recipe = delivery_restore.prepare(work, work / "composition", [Path("/m/p.mov")])
    write(work / "retention" / "archive" / "0000-render-0-0000" / "000000-render.response.json", {})
    with pytest.raises(ReplayError, match="already exists"):
        delivery_restore.archive(recipe, work / "retention")


def test_restore_sets_the_receipt_aside_and_names_the_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(disk_budget, "require_headroom", lambda _paths: [])
    directory = tmp_path / "retention"
    write(directory / "restore-recipe.json", {"plans": [{"plan": "/w/join-workflow.json"}]})
    write(directory / "retention-receipt.json", {})
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    named = delivery_restore.restore(source, [tmp_path], directory, "retention-receipt.json")
    assert named == ["/w/join-workflow.json"]
    assert not (directory / "retention-receipt.json").exists()
    assert (directory / "archive" / "retention-receipt.json").is_file()


def test_restore_requires_a_recipe(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="restore recipe"):
        delivery_restore.restore(tmp_path / "s.mp4", [tmp_path], tmp_path, "retention-receipt.json")


def test_restore_requires_the_original_recording(tmp_path: Path) -> None:
    write(tmp_path / "restore-recipe.json", {"plans": []})
    write(tmp_path / "retention-receipt.json", {})
    with pytest.raises(ReplayError, match="original recording"):
        delivery_restore.restore(tmp_path / "s.mp4", [tmp_path], tmp_path, "retention-receipt.json")


def test_a_stage_rebuilds_after_its_retired_output_is_archived(tmp_path: Path) -> None:
    work = tmp_path / "work"
    source = tmp_path / "source.wav"
    source.write_bytes(b"input")
    movie = work / "part.mov"
    response = work / "resp.json"
    plan = workflow(work / "join-workflow.json", [str(source)], [str(movie), str(response)])
    state = work / "join-workflow-state"
    stage = Stage(
        name="make",
        command=("make",),
        inputs=(source,),
        outputs=(movie, response),
        timeout_seconds=5,
    )

    def produce(_command: tuple[str, ...], log: Path, _timeout: int) -> None:
        movie.write_bytes(b"movie")
        response.write_text("{}", encoding="utf-8")
        log.write_bytes(b"")

    assert run_stage(stage, state_directory=state, execute=produce) == "completed"
    movie.unlink()
    with pytest.raises(FileNotFoundError):
        run_stage(stage, state_directory=state, execute=produce)
    recipe = delivery_restore.prepare(work, work / "composition", [movie])
    delivery_restore.archive(recipe, work / "retention")
    assert run_stage(stage, state_directory=state, execute=produce) == "completed"
    assert plan.is_file()
    assert movie.read_bytes() == b"movie"
