"""Real httpx/httpcore startup errors with fake sockets, process, input and clock; no network."""
import errno
import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpcore
import httpx
import e17_local as local
from tests import test_e17_local_deadline as deadline_tests


class FakeSocket:
    def __init__(self, malformed=False):
        self.data = b"invalid HTTP\r\n\r\n" if malformed else (
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")

    def settimeout(self, timeout):
        if timeout <= 0:
            raise ValueError("synthetic socket rejects expired timeout")

    def setsockopt(self, *args):
        pass

    def send(self, data):
        return len(data)

    def recv(self, size):
        data, self.data = self.data[:size], self.data[size:]
        return data

    def close(self):
        pass


class HTTPErrorTests(unittest.TestCase):
    def exercise_http_worker(self, mode, expected, system_errno=10061):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            clock = SimpleNamespace(now=100.0)
            order, chains, attempts = [], [], []
            process = mock.Mock(returncode=0)
            process.poll.return_value = None
            process.terminate.side_effect = lambda: order.append("terminate")
            process.wait.side_effect = lambda **k: order.append("wait")
            port_socket = mock.MagicMock()
            port_socket.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 1234)
            def connect(*args, **kwargs):
                attempts.append(clock.now)
                if mode == "always" or mode == "system" or mode == "once" and len(attempts) == 1:
                    if system_errno in (errno.ECONNREFUSED, 10061):
                        raise ConnectionRefusedError(system_errno, "synthetic startup refusal")
                    raise OSError(system_errno, "synthetic startup error")
                return FakeSocket(mode == "protocol")

            original_get = httpx.Client.get
            def get(client, *args, **kwargs):
                try:
                    return original_get(client, *args, **kwargs)
                except httpx.ConnectError as error:
                    nodes, visited = [], set()
                    current = error
                    while current is not None and id(current) not in visited and len(nodes) < 16:
                        visited.add(id(current))
                        nodes.append({"type": type(current).__name__, "errno": getattr(current, "errno", None),
                                      "cause": type(current.__cause__).__name__ if current.__cause__ else None,
                                      "context": type(current.__context__).__name__ if current.__context__ else None,
                                      "suppressed_context": current.__suppress_context__})
                        current = current.__cause__ if current.__cause__ is not None else current.__context__
                    chains.append(nodes)
                    raise

            session = "0" * 32
            values = [{"session": session, "models": [local.QWEN], "execute": True,
                       "binary": "fake", "models_dir": str(root), "output": str(root)},
                      {"session": session, "id": 1, "timeout": .4, "json": {"model": local.QWEN}},
                      {"session": session, "id": 2, "close": True}]
            pending = mock.Mock()
            pending.get.side_effect = lambda timeout: local.canonical(values.pop(0)) + b"\n"
            generation = mock.Mock(return_value=SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {"model": local.QWEN, "marker": "own-generation"}))
            def sleep(seconds):
                clock.now += seconds
            for patch in (mock.patch.object(local, "os", SimpleNamespace(name="nt")),
                    mock.patch.object(local, "owned_job", return_value=1),
                    mock.patch.object(local, "WorkerInput", return_value=pending),
                    mock.patch.object(local, "inspect_assets", return_value={"synthetic": True}),
                    mock.patch.object(local, "loopback_client", return_value=SimpleNamespace(post=generation)),
                    mock.patch.object(local.time, "monotonic", side_effect=lambda: clock.now),
                    mock.patch.object(local.time, "sleep", side_effect=sleep),
                    mock.patch.object(local.socket, "socket", return_value=port_socket),
                    mock.patch("httpcore._backends.sync.socket.create_connection", side_effect=connect),
                    mock.patch.object(httpx.Client, "get", get),
                    mock.patch.object(local.subprocess, "Popen", return_value=process),
                    mock.patch.object(local.subprocess, "CREATE_NO_WINDOW", 0, create=True)):
                stack.enter_context(patch)
            output = io.StringIO()
            original_print = print
            def record(*args, **kwargs):
                order.append("ack" + str(json.loads(args[0])["id"]))
                original_print(*args, **kwargs)
            stack.enter_context(mock.patch("builtins.print", side_effect=record))
            with redirect_stdout(output):
                local.worker()
            replies = [json.loads(line) for line in output.getvalue().splitlines()]
            evidence = {"httpx": httpx.__version__, "httpcore": httpcore.__version__, "network_used": False,
                        "mode": mode, "errno": system_errno, "chains": chains, "attempts": attempts,
                        "order": order, "replies": replies, "generation_calls": generation.call_count}
            if os.environ.get("E17_HTTP_PROOFS"):
                target = Path(os.environ["E17_HTTP_PROOFS"]) / (mode + "-" + str(system_errno) + ".json")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            self.assertEqual(replies[1]["outcome"], expected)
            self.assertEqual([r["id"] for r in replies], [0, 1, 2])
            self.assertTrue(all(r["session"] == session for r in replies))
            self.assertTrue(replies[2]["closed"])
            self.assertEqual(generation.call_count, 1 if expected == "response" else 0)
            if expected == "response":
                self.assertEqual(replies[1]["response"]["marker"], "own-generation")
                self.assertEqual(len(attempts), 2)
            else:
                self.assertLess(order.index("terminate"), order.index("wait"))
                self.assertLess(order.index("wait"), order.index("ack1"))
            if mode != "protocol":
                self.assertTrue(any(n["errno"] == system_errno and n["type"] == "ConnectionRefusedError"
                                    for n in chains[0]) if system_errno in (errno.ECONNREFUSED, 10061) else
                                any(n["errno"] == system_errno for n in chains[0]))
                self.assertTrue(any(n["cause"] is None and n["context"] is not None and n["suppressed_context"]
                                    for n in chains[0]), chains)
            if expected == "infrastructure":
                self.assertEqual(replies[1]["exception_type"], "RemoteProtocolError" if mode == "protocol" else "ConnectError")

    def test_real_httpcore_suppressed_refusal_then_health_ready_and_generation(self):
        for value in sorted({errno.ECONNREFUSED, 10061}):
            with self.subTest(errno=value):
                self.exercise_http_worker("once", "response", value)

    def test_real_httpcore_refusals_until_deadline_stop_before_timeout_ack(self):
        self.exercise_http_worker("always", "generation_timeout")

    def test_real_httpcore_other_system_errors_stay_infrastructure(self):
        for value in (errno.ENETUNREACH, errno.EACCES):
            with self.subTest(errno=value):
                self.exercise_http_worker("system", "infrastructure", value)

    def test_real_httpcore_protocol_error_stays_infrastructure(self):
        self.exercise_http_worker("protocol", "infrastructure")

    def test_explicit_non_refusal_cause_wins_over_refusal_context(self):
        error = httpx.ConnectError("synthetic ambiguity")
        error.__cause__ = PermissionError(errno.EACCES, "synthetic explicit cause")
        error.__context__ = ConnectionRefusedError(10061, "synthetic implicit context")
        error.__suppress_context__ = True
        deadline_tests.DeadlineTests().exercise_worker("health_failure", "infrastructure", error)

    def test_cycles_and_chain_beyond_bound_stay_infrastructure(self):
        cycle = httpx.ConnectError("synthetic cycle")
        cycle.__cause__ = cycle
        implicit = httpx.ConnectError("synthetic implicit cycle")
        implicit.__context__ = implicit
        deep = ConnectionRefusedError(10061, "synthetic beyond bound")
        for _ in range(16):
            wrapper = httpx.ConnectError("synthetic deep chain")
            wrapper.__cause__ = deep
            deep = wrapper
        for error in (cycle, implicit, deep):
            with self.subTest(error=str(error)):
                deadline_tests.DeadlineTests().exercise_worker("health_failure", "infrastructure", error)
