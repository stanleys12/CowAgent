# encoding:utf-8
"""Group members who are not owners must not reach the owner's files or shell."""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.protocol.agent import Agent
from agent.protocol.agent_stream import AgentStreamExecutor
from bridge.agent_bridge import AgentBridge
from bridge.context import Context, ContextType
from config import conf


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "notes.md").write_text("hi")
    (tmp_path / ".env").write_text("KEY=secret")
    return tmp_path


def _agent(ws):
    agent = Agent.__new__(Agent)
    agent.model = object()
    agent.tools = []
    agent.max_steps = 5
    agent.messages = []
    agent.messages_lock = threading.Lock()
    agent.get_full_system_prompt = lambda skill_filter=None: "sys"
    agent._execute_post_process_tools = lambda: None
    agent.effective_permission_mode = lambda: "full-access"
    agent.effective_cwd = lambda: str(ws)
    agent.write_roots = lambda: []
    agent.protected_paths = lambda: [str(ws / ".env")]
    return agent


def _denial(agent, tool, args, guest=True):
    executor = AgentStreamExecutor(agent, agent.model, "sys", [], guest=guest)
    return executor._permission_denial(tool, args)


@pytest.mark.parametrize("sender,isgroup,owners,expected", [
    ("ou_owner", True, ["ou_owner"], False),
    ("ou_guest", True, ["ou_owner"], True),
    (None, True, ["ou_owner"], True),
    ("ou_guest", False, ["ou_owner"], False),
    ("ou_guest", True, [], False),
])
def test_requester_is_resolved_per_message(monkeypatch, sender, isgroup, owners, expected):
    monkeypatch.setitem(conf(), "agent_owner_ids", owners)
    msg = SimpleNamespace(actual_user_id=sender) if sender is not None else None
    context = Context(ContextType.TEXT, "hi", {"isgroup": isgroup, "msg": msg})
    assert AgentBridge._is_guest(context) is expected


@pytest.mark.parametrize("tool,args,reason", [
    ("bash", {"command": "cat ~/.ssh/id_rsa"}, "bash is restricted to the owner"),
    ("send", {"path": "notes.md"}, "send is restricted to the owner"),
    ("scheduler", {"action": "create"}, "scheduler is restricted to the owner"),
    ("search_files", {"pattern": "KEY", "path": "."}, "search_files is restricted to the owner"),
    ("github_create_issue", {}, "github_create_issue is restricted to the owner"),
    ("read", {"path": "/etc/passwd"}, "'/etc/passwd' is outside the workspace"),
    ("read", {"path": ".env"}, "'.env' holds config or credentials"),
])
def test_guest_is_refused(ws, tool, args, reason):
    denial = _denial(_agent(ws), tool, args)
    assert reason in denial
    assert "Only the owner" in denial


def test_scheduled_run_in_a_group_is_not_a_guest(monkeypatch):
    monkeypatch.setitem(conf(), "agent_owner_ids", ["ou_owner"])
    context = Context(ContextType.TEXT, "hi", {"isgroup": True, "msg": None, "is_scheduled_task": True})
    assert AgentBridge._is_guest(context) is False


@pytest.mark.parametrize("sender,expected", [("ou_guest", True), ("ou_owner", False)])
def test_agent_reply_runs_a_guest_message_as_guest(monkeypatch, sender, expected):
    monkeypatch.setitem(conf(), "agent_owner_ids", ["ou_owner"])
    seen = {}

    class _Agent:
        permission_mode = None
        tools = []
        captured_actions = []
        memory_manager = None
        skill_manager = None
        runtime_info = None
        workspace_dir = None
        max_steps = 8

        def run_stream(self, *args, **kwargs):
            seen["guest"] = kwargs.get("guest")
            return "done"

    bridge = AgentBridge.__new__(AgentBridge)
    bridge.agent_registry = SimpleNamespace(default_agent_id="default", list=lambda: [])
    context = Context(ContextType.TEXT, "hi", {
        "session_id": "g1",
        "isgroup": True,
        "msg": SimpleNamespace(actual_user_id=sender),
    })
    with patch.object(AgentBridge, "_has_runtime", return_value=True), \
         patch.object(AgentBridge, "get_agent", return_value=_Agent()), \
         patch.object(AgentBridge, "route_context", return_value="default"), \
         patch.object(AgentBridge, "_seed_team_members", return_value=None), \
         patch.object(AgentBridge, "_peer_speaker", return_value=None), \
         patch("bridge.agent_bridge.get_steer_registry"), \
         patch("bridge.agent_bridge.get_cancel_registry"), \
         patch("agent.evolution.trigger.mark_run_active", create=True):
        bridge.agent_reply("hi", context)

    assert seen["guest"] is expected


def test_guest_keeps_safe_tools(ws):
    assert _denial(_agent(ws), "web_search", {"query": "x"}) is None
    assert _denial(_agent(ws), "read", {"path": "notes.md"}) is None


def test_owner_is_untouched(ws):
    assert _denial(_agent(ws), "bash", {"command": "cat /etc/passwd"}, guest=False) is None


def test_a_failed_check_refuses_a_guest(ws):
    agent = _agent(ws)
    agent.effective_cwd = lambda: 1 / 0
    assert "did not run" in _denial(agent, "read", {"path": "notes.md"})


def test_owner_turn_started_mid_guest_turn_does_not_lift_the_guest_limits(ws, monkeypatch):
    agent = _agent(ws)
    denials = {}

    def run_stream(self, user_message):
        if self.guest:
            agent.run_stream("owner")
        denials[user_message] = self._permission_denial("bash", {"command": "id"})
        return "ok"

    monkeypatch.setattr(AgentStreamExecutor, "run_stream", run_stream)
    agent.run_stream("guest", guest=True)

    assert denials["owner"] is None
    assert "bash is restricted to the owner" in denials["guest"]
