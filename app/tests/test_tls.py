"""TLS trust roots setup (no network)."""
import sys
import types

import pytest

from myriad import tls


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(tls, "_mode", None)


def fake_truststore(monkeypatch, fail: Exception | None = None):
    calls = []
    mod = types.ModuleType("truststore")

    def inject_into_ssl():
        calls.append(1)
        if fail:
            raise fail

    mod.inject_into_ssl = inject_into_ssl
    monkeypatch.setitem(sys.modules, "truststore", mod)
    return calls


def test_truststore_injected_by_default(monkeypatch):
    calls = fake_truststore(monkeypatch)
    env = {}
    assert tls.setup_tls(env) == "truststore"
    assert calls == [1]
    assert tls.setup_tls(env) == "truststore" and calls == [1]  # idempotent
    if sys.platform in ("win32", "darwin"):
        assert "SSL_CERT_FILE" not in env


@pytest.mark.parametrize("var", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_user_variables_are_respected(monkeypatch, var):
    calls = fake_truststore(monkeypatch)
    env = {var: "/custom/ca"}
    assert tls.setup_tls(env) == "env"
    assert calls == [] and env == {var: "/custom/ca"}


def test_fallback_on_certifi_when_truststore_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)  # import raises ImportError
    env = {}
    assert tls.setup_tls(env) == "certifi"
    import certifi
    assert env["SSL_CERT_FILE"] == certifi.where()


def test_fallback_on_certifi_when_truststore_fails(monkeypatch):
    fake_truststore(monkeypatch, fail=OSError("no keychain"))
    env = {}
    assert tls.setup_tls(env) == "certifi"
    assert env["SSL_CERT_FILE"]


def test_fallback_does_not_override_user_dir(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)
    env = {}
    tls._use_certifi({"SSL_CERT_DIR": "/d"})  # user value present: nothing is added
    mode = tls._use_certifi(env)
    assert mode == "certifi" and "SSL_CERT_FILE" in env
    keep = {"SSL_CERT_DIR": "/d"}
    tls._use_certifi(keep)
    assert keep == {"SSL_CERT_DIR": "/d"}


def test_no_bundle_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "truststore", None)
    monkeypatch.setitem(sys.modules, "certifi", None)
    env = {}
    assert tls.setup_tls(env) == "none" and env == {}


def test_linux_without_system_paths_adds_certifi(monkeypatch):
    fake_truststore(monkeypatch)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(tls, "_default_paths_usable", lambda: False)
    env = {}
    assert tls.setup_tls(env) == "truststore" and "SSL_CERT_FILE" in env


def test_selftest_reports_failure(monkeypatch, capsys):
    import httpx
    fake_truststore(monkeypatch)

    def boom(*a, **k):
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")

    monkeypatch.setattr(httpx, "get", boom)
    assert tls.selftest("https://example.invalid/") == 1
    assert "CERTIFICATE_VERIFY_FAILED" in capsys.readouterr().err


def test_selftest_ok(monkeypatch, capsys):
    import httpx
    fake_truststore(monkeypatch)
    monkeypatch.setattr(httpx, "get", lambda url, **k: httpx.Response(200, request=httpx.Request("GET", url)))
    assert tls.selftest("https://example.invalid/") == 0
    assert "OK" in capsys.readouterr().out
