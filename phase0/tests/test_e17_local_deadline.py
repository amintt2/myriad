"""Deterministic startup/stdio boundaries with fake clocks, clients and processes only."""
import io
import errno
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import e17_local as local


class DeadlineTests(unittest.TestCase):
    def exercise_worker(self, boundary, expected, error=None):
        exception_type = type(error).__name__ if error else "ValueError" if boundary == "foreign_model" else "RuntimeError"
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            clock = SimpleNamespace(now=100.0)
            order, health_timeouts = [], []
            process = mock.Mock(returncode=0)
            process.poll.return_value = None
            process.terminate.side_effect = lambda: order.append("terminate")
            process.wait.side_effect = lambda **k: order.append("wait")
            sock = mock.MagicMock()
            sock.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 1234)
            response = SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                       json=lambda: {"model": local.QWEN, "marker": "second"})

            def enter():
                if boundary == "health_client":
                    clock.now += 2
                return health

            def get(*a, timeout):
                nonlocal boundary
                health_timeouts.append(timeout)
                if timeout <= 0:
                    raise ValueError("synthetic health transport rejects negative timeout")
                if boundary == "health_failure":
                    clock.now += 2
                    raise error
                if boundary == "connection_refused":
                    boundary = "success"
                    cause = ConnectionRefusedError(errno.ECONNREFUSED, "synthetic not listening")
                    raise httpx.ConnectError("synthetic not listening") from cause
                return response

            health = mock.MagicMock()
            health.__enter__.side_effect = enter
            health.get.side_effect = get

            def factory(*a):
                if boundary == "generation_client":
                    clock.now += 2
                def post(*a, **k):
                    if boundary == "generation_failure":
                        raise error
                    if boundary == "foreign_model":
                        return SimpleNamespace(raise_for_status=lambda: None,
                                               json=lambda: {"model": "foreign"})
                    return response
                return SimpleNamespace(post=post)

            session = "0" * 32
            config = {"session": session, "models": [local.QWEN], "execute": True,
                      "binary": "fake", "models_dir": str(root), "output": str(root)}
            requests = [config, {"session": session, "id": 1, "timeout": 1,
                                "json": {"model": local.QWEN}}]
            # Switch to a successful fixture only after the first ACK has been printed.
            def next_line(timeout):
                nonlocal boundary
                if requests:
                    return local.canonical(requests.pop(0)) + b"\n"
                if "next" not in order:
                    order.append("next")
                    boundary = "success"
                    process.poll.return_value = None
                    return local.canonical({"session": session, "id": 2, "timeout": 1,
                                            "json": {"model": local.QWEN}}) + b"\n"
                return local.canonical({"session": session, "id": 3, "close": True}) + b"\n"

            pending = mock.Mock()
            pending.get.side_effect = next_line
            if boundary == "process_exit":
                process.poll.return_value = 1
            real_client = local.WindowsClient
            clients = []
            def client(*a):
                value = real_client(*a)
                clients.append(value)
                return value
            for patch in (mock.patch.object(local, "os", SimpleNamespace(name="nt")),
                    mock.patch.object(local, "owned_job", return_value=1),
                    mock.patch.object(local, "WorkerInput", return_value=pending),
                    mock.patch.object(local, "inspect_assets", return_value={"synthetic": True}),
                    mock.patch.object(local, "WindowsClient", side_effect=client),
                    mock.patch.object(local, "loopback_client", side_effect=factory),
                    mock.patch.object(local.time, "monotonic", side_effect=lambda: clock.now),
                    mock.patch.object(local.time, "sleep"),
                    mock.patch.object(local.socket, "socket", return_value=sock),
                    mock.patch.object(local.subprocess, "Popen", return_value=process),
                    mock.patch.object(local.subprocess, "CREATE_NO_WINDOW", 0, create=True),
                    mock.patch("httpx.Client", return_value=health)):
                stack.enter_context(patch)
            output = io.StringIO()
            original_print = print
            def record(*a, **k):
                value = json.loads(a[0])
                order.append("ack" + str(value["id"]))
                original_print(*a, **k)
            stack.enter_context(mock.patch("builtins.print", side_effect=record))
            with redirect_stdout(output):
                local.worker()
            replies = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(replies[1]["outcome"], expected)
            if expected == "infrastructure":
                self.assertEqual(replies[1]["exception_type"], exception_type)
            self.assertEqual([r["id"] for r in replies], [0, 1, 2, 3])
            self.assertTrue(all(r["session"] == session for r in replies))
            self.assertEqual(replies[2]["response"]["marker"], "second")
            self.assertTrue(replies[3]["closed"])
            if expected != "response":
                self.assertLess(order.index("terminate"), order.index("wait"))
                self.assertLess(order.index("wait"), order.index("ack1"))
            self.assertLess(order.index("ack1"), order.index("next"))
            self.assertTrue(clients[0].events.closed)
            self.assertIsNone(clients[0].process)
            self.assertTrue(all(t > 0 for t in health_timeouts), health_timeouts)
            return replies

    def test_expiry_during_health_client_creation_stops_before_ack_and_reuses_session(self):
        self.exercise_worker("health_client", "generation_timeout")

    def test_expiry_during_generation_client_creation_stops_before_ack_and_reuses_session(self):
        self.exercise_worker("generation_client", "generation_timeout")

    def test_startup_process_exit_remains_infrastructure(self):
        self.exercise_worker("process_exit", "infrastructure")

    def test_non_timeout_failures_remain_infrastructure(self):
        for error in (OSError("synthetic disk failure"), httpx.RemoteProtocolError("synthetic protocol"),
                      ValueError("synthetic invalid configuration")):
            with self.subTest(error=type(error).__name__):
                self.exercise_worker("generation_failure", "infrastructure", error)

    def test_health_protocol_failure_remains_infrastructure(self):
        self.exercise_worker("health_failure", "infrastructure", httpx.RemoteProtocolError("synthetic protocol"))

    def test_health_non_refusal_connect_error_remains_infrastructure(self):
        self.exercise_worker("health_failure", "infrastructure", httpx.ConnectError("synthetic transport failure"))

    def test_health_connection_refused_retries_within_same_budget(self):
        self.exercise_worker("connection_refused", "response")

    def test_http_timeout_stops_before_ack_and_reuses_session(self):
        self.exercise_worker("generation_failure", "generation_timeout", httpx.ReadTimeout("synthetic timeout"))

    def test_foreign_model_remains_infrastructure(self):
        self.exercise_worker("foreign_model", "infrastructure")

    def test_success_remains_response(self):
        self.exercise_worker("success", "response")

    def test_late_timeout_drained_exactly_once_before_next_answer(self):
        client = object.__new__(local.PipeClient)
        client.broken, client.pending, client.sequence = False, 1, 1
        client.deadline = 200
        client.loading_seconds = client.request_seconds = 0
        def receive(timeout):
            self.assertIsNotNone(client.pending)
            value = replies.pop(0)
            self.assertEqual(value["id"], client.pending)
            client.pending = None
            return value
        def send(value):
            self.assertIsNone(client.pending)
            client.sequence += 1
            client.pending = client.sequence
        replies = [{"id": 1, "outcome": "generation_timeout", "loading_seconds": 2, "request_seconds": 0},
                   {"id": 2, "outcome": "response", "loading_seconds": 3, "request_seconds": 1,
                    "response": {"marker": "second"}}]
        with mock.patch.object(local.time, "monotonic", return_value=100), \
                mock.patch.object(client, "receive", side_effect=receive) as read, \
                mock.patch.object(client, "send", side_effect=send):
            response = client.post("chat/completions", {"model": local.QWEN}, 1)
            self.assertEqual(response.json(), {"marker": "second"})
            self.assertEqual(read.call_count, 2)
            self.assertEqual(client.sequence, 2)
            self.assertIsNone(client.pending)
            client.synchronize()
            self.assertEqual(read.call_count, 2)

    def test_late_infrastructure_includes_exception_type_and_blocks_next_send(self):
        client = object.__new__(local.PipeClient)
        client.broken, client.pending = False, 1
        client.loading_seconds = client.request_seconds = 0
        result = {"outcome": "infrastructure", "exception_type": "OSError",
                  "loading_seconds": 1, "request_seconds": 0}
        with mock.patch.object(client, "receive", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "infrastructure failure: OSError"):
                client.synchronize()
        self.assertTrue(client.broken)
        with self.assertRaises(RuntimeError):
            client.send({"json": {"model": local.QWEN}, "timeout": 1})
