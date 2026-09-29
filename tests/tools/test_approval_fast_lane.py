"""Tests for the approval fast lane (``tools.approval_fast_lane``) and its wiring
into ``tools.approval_smart._smart_verdict``.

The fast lane is a security boundary: a local typed-decision endpoint may
APPROVE its high-confidence clearly-safe class, and every other outcome must
fall through to the guardian LLM, which remains the only deny authority. These
tests pin exactly that contract:

  1. Disabled config -> the endpoint is never contacted.
  2. Approvable answer above the floor -> "approve".
  3. Low probability / low confidence -> None (guardian decides).
  4. Transport / parse / malformed-payload failure -> None, never an exception.
  5. ``_smart_verdict``: a fast-lane approve skips ``call_llm``; a fall-through
     still consults it (guardian is never bypassed).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from tools.approval_fast_lane import _fast_lane_verdict
from tools.approval_smart import _smart_verdict

_APPROVE_PAYLOAD = json.dumps({
    "answers": {"clearly_safe": {"type": "boolean", "probability": 0.99}},
    "confidence": {"clearly_safe": 0.99},
})


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _cfg(enabled=True, floor=0.6):
    return {"fast_lane": {"enabled": enabled, "base_url": "http://127.0.0.1:9/", "confidence_floor": floor, "timeout": 1}}


class TestFastLaneConfigGate(unittest.TestCase):
    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_disabled_never_contacts_endpoint(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg(enabled=False)
        assert _fast_lane_verdict("npm install -g x", "package install") is None
        mock_urlopen.assert_not_called()

    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_missing_base_url_falls_through(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = {"fast_lane": {"enabled": True}}
        assert _fast_lane_verdict("npm install -g x", "package install") is None
        mock_urlopen.assert_not_called()


class TestFastLaneVerdicts(unittest.TestCase):
    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_confident_approvable_answer(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg()
        mock_urlopen.return_value = _FakeResponse(_APPROVE_PAYLOAD.encode())
        assert _fast_lane_verdict("git status", "git inspection") == "approve"

    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_probability_below_floor_falls_through(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg(floor=0.99)
        body = json.dumps({"answers": {"clearly_safe": {"probability": 0.66}}, "confidence": {"clearly_safe": 0.66}})
        mock_urlopen.return_value = _FakeResponse(body.encode())
        assert _fast_lane_verdict("cp a b", "file copy") is None

    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_low_provider_confidence_falls_through_even_with_high_probability(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg(floor=0.6)
        body = json.dumps({"answers": {"clearly_safe": {"probability": 0.99}}, "confidence": {"clearly_safe": 0.3}})
        mock_urlopen.return_value = _FakeResponse(body.encode())
        assert _fast_lane_verdict("npm install -g x", "package install") is None

    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_transport_failure_is_none_not_raise(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg()
        mock_urlopen.side_effect = OSError("connection refused")
        assert _fast_lane_verdict("npm install -g x", "package install") is None

    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_malformed_payload_is_none(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg()
        mock_urlopen.return_value = _FakeResponse(b"not json at all")
        assert _fast_lane_verdict("npm install -g x", "package install") is None

    @patch("tools.approval_context._get_approval_config")
    @patch("urllib.request.urlopen")
    def test_missing_answer_is_none(self, mock_urlopen, mock_cfg):
        mock_cfg.return_value = _cfg()
        mock_urlopen.return_value = _FakeResponse(json.dumps({"answers": {}}).encode())
        assert _fast_lane_verdict("npm install -g x", "package install") is None


class TestSmartVerdictWiring(unittest.TestCase):
    @patch("tools.approval_context._get_approval_config")
    @patch("tools.approval_context._fire_approval_hook")
    @patch("agent.auxiliary_client.call_llm")
    @patch("urllib.request.urlopen")
    def test_fast_lane_approve_skips_guardian(self, mock_urlopen, mock_call_llm, mock_hook, mock_cfg):
        mock_cfg.return_value = _cfg()
        mock_urlopen.return_value = _FakeResponse(_APPROVE_PAYLOAD.encode())
        verdict = _smart_verdict("npm install -g x", "package install", "key", ["key"], "sess")
        assert verdict == "approve"
        mock_call_llm.assert_not_called()

    @patch("tools.approval_context._get_approval_config")
    @patch("tools.approval_context._fire_approval_hook")
    @patch("agent.auxiliary_client.call_llm")
    @patch("urllib.request.urlopen")
    def test_fall_through_still_consults_guardian(self, mock_urlopen, mock_call_llm, mock_hook, mock_cfg):
        mock_cfg.return_value = _cfg()
        mock_urlopen.return_value = _FakeResponse(
            json.dumps({"answers": {"clearly_safe": {"probability": 0.01}}}).encode())
        response = MagicMock()
        response.choices[0].message.content = "DENY"
        mock_call_llm.return_value = response
        verdict = _smart_verdict(
            "rm -rf ~/Library/Caches/com.example.app", "recursive delete", "key", ["key"], "sess")
        assert verdict == "deny"
        assert mock_call_llm.call_count == 1

    @patch("tools.approval_context._get_approval_config")
    @patch("tools.approval_context._fire_approval_hook")
    @patch("agent.auxiliary_client.call_llm")
    @patch("urllib.request.urlopen")
    def test_fast_lane_crash_never_blocks_the_gate(self, mock_urlopen, mock_call_llm, mock_hook, mock_cfg):
        """A fast-lane import/config bug must fall back to the guardian, not skip the gate."""
        mock_cfg.return_value = _cfg()
        mock_urlopen.side_effect = RuntimeError("boom")
        response = MagicMock()
        response.choices[0].message.content = "ESCALATE"
        mock_call_llm.return_value = response
        verdict = _smart_verdict("npm install -g x", "package install", "key", ["key"], "sess")
        assert verdict == "escalate"
        assert mock_call_llm.call_count == 1


if __name__ == "__main__":
    unittest.main()
