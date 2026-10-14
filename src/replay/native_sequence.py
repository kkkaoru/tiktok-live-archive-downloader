"""Finite orchestration; adapters own original proof and durable storage."""

from dataclasses import dataclass
from typing import Literal, Protocol

Phase = Literal["prepared", "started", "published", "retired", "released", "uncertain"]


class ContractError(ValueError):
    """Immutable sequence or effect contract was violated."""


@dataclass(frozen=True)
class Config:
    identity: str
    tail: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class Checkpoint:
    config: Config
    phase: Phase
    proof: str = ""


@dataclass(frozen=True)
class Outcome:
    status: Literal["complete", "resource-refused", "uncertain"]
    part: int


class Journal(Protocol):
    """Production adapter must provide exclusive durable compare-and-swap."""

    def get(self, part: int) -> Checkpoint | None: ...

    def put(self, part: int, old: Checkpoint | None, new: Checkpoint) -> None: ...


class Steps(Protocol):
    """Callbacks must retain original API evidence; strings are opaque proof refs."""

    def materialize(self, tail: tuple[int, ...]) -> Config: ...

    def validate(self, checkpoint: Checkpoint) -> None: ...

    def capacity(self, config: Config) -> bool: ...

    def pump(self, config: Config) -> str | None: ...

    def retire(self, config: Config, publication: str) -> str: ...

    def close_observation_lease(self) -> None: ...

    def leased(self) -> bool: ...

    def release(self, current: Config, successor: Config, retirement: str) -> str: ...


class Recovery(Protocol):
    """Optional bounded recovery; returns DATA only, never retries the current pump."""

    def refused(self, checkpoint: Checkpoint) -> None: ...


def _refused(checkpoint: Checkpoint, steps: Steps, recovery: Recovery | None) -> Outcome:
    if recovery is not None and checkpoint.phase == "prepared" and not steps.leased():
        recovery.refused(checkpoint)
    return Outcome("resource-refused", checkpoint.config.tail[0])


def _config(steps: Steps, tail: tuple[int, ...]) -> Config:
    config = steps.materialize(tail)
    if config.tail != tail or not config.digest or not config.identity:
        raise ContractError("Materializer changed the ordered tail or omitted identity")
    return config


def _prepared(journal: Journal, config: Config) -> Checkpoint:
    part = config.tail[0]
    saved = journal.get(part)
    if saved is None:
        saved = Checkpoint(config, "prepared")
        journal.put(part, None, saved)
    if saved.config != config:
        raise ContractError("Existing next-part config is immutable")
    return saved


def run(
    parts: tuple[int, ...], journal: Journal, steps: Steps, *, recovery: Recovery | None = None
) -> Outcome:
    """Run a bounded sequence; never repeat an uncertain mutating effect.

    A retirement callback must resume its original idempotent workflow state.
    An interrupted pump needs external reconciliation, not another pump call.
    No callback is allowed to synthesize a live grant or freshness timestamp.
    """
    if not parts or len(parts) > 1024 or len(set(parts)) != len(parts):
        raise ContractError("Expected finite unique ordered parts")
    if any(type(part) is not int or part <= 0 for part in parts):
        raise ContractError("Expected positive part identities")
    for index, part in enumerate(parts):
        config = _config(steps, parts[index:])
        state = _prepared(journal, config)
        steps.validate(state)

        def advance(
            phase: Phase, proof: str = "", *, selected: Config = config, identity: int = part
        ) -> None:
            nonlocal state
            updated = Checkpoint(selected, phase, proof)
            journal.put(identity, state, updated)
            state = updated

        if state.phase in ("started", "uncertain"):
            return Outcome("uncertain", part)
        if state.phase == "prepared":
            if not steps.capacity(config):
                return _refused(state, steps, recovery)
            advance("started")
            proof = steps.pump(config)
            if not proof:
                advance("uncertain")
                return Outcome("uncertain", part)
            advance("published", proof)
        if state.phase == "published":
            proof = steps.retire(config, state.proof)
            if not proof:
                raise ContractError("Actual retirement/postverification proof is required")
            advance("retired", proof)
        if state.phase == "retired":
            if index + 1 < len(parts):
                successor = _config(steps, parts[index + 1 :])
                successor_state = _prepared(journal, successor)
                steps.close_observation_lease()
                if steps.leased():
                    raise ContractError("Release requires the observation lease to be closed")
                if not steps.capacity(successor):
                    return _refused(successor_state, steps, recovery)
                proof = steps.release(config, successor, state.proof)
                if not proof:
                    raise ContractError("Verified release proof is required")
                advance("released", proof)
            else:
                advance("released", state.proof)
    return Outcome("complete", parts[-1])
