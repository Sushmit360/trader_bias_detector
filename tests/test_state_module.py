from dataclasses import dataclass, field
from typing import Any, Dict

import app_modules.state as state_mod


@dataclass
class StubStreamlit:
    session_state: Dict[str, Any] = field(default_factory=dict)


def test_init_state_populates_defaults(monkeypatch):
    stub = StubStreamlit()
    stub.session_state["active_view"] = "dashboard"
    monkeypatch.setattr(state_mod, "st", stub)

    state_mod.init_state()

    defaults = state_mod.DEFAULT_SESSION_STATE
    for key, default_value in defaults.items():
        assert key in stub.session_state
        if key == "active_view":
            # Should respect existing value
            assert stub.session_state[key] == "dashboard"
        else:
            assert stub.session_state[key] == default_value
