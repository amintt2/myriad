"""Tiny standard-library-only remote observation; invoked exclusively by the E17 supervisor payload."""
import ctypes
import hashlib
import importlib.util
import json
import platform
import subprocess
import tempfile
from pathlib import Path

FLAGS = ["--user", "--map-root-user", "--mount", "--net", "--pid", "--ipc", "--uts",
         "--fork", "--kill-child=KILL"]
NAMESPACES = ["user", "mount", "pid", "net", "ipc", "uts"]


def observe(request, source, ephemeral_source=None):
    result = {**request, "schema": 1, "origin": "colab-exec", "kernel": platform.release(),
              "system": platform.system(), "machine": platform.machine(), "landlock_abi": None,
              "namespace_flags": FLAGS, "namespaces": {}, "namespace_combined": None,
              "seccomp": None, "smoke": None, "verdict": "infrastructure-unknown"}
    if hashlib.sha256(source.encode()).hexdigest() != request["local_linux_sha256"]:
        return result
    if request.get("sandbox_backend", "strict") not in ("strict", "ephemeral-vm"):
        return result
    ephemeral = request.get("sandbox_backend", "strict") == "ephemeral-vm"
    if ephemeral and (not isinstance(ephemeral_source, str) or hashlib.sha256(ephemeral_source.encode()).hexdigest()
                      != request.get("ephemeral_vm_sha256")):
        return result
    if result["system"] != "Linux" or result["machine"] != "x86_64":
        result["verdict"] = "incompatible"
        return result
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result["landlock_abi"] = libc.syscall(444, 0, 0, 1)
    try:
        for name in NAMESPACES:
            flags = ["--user", "--map-root-user"]
            if name != "user":
                flags += ["--" + name]
            flags += ["--fork", "--kill-child=KILL"]
            result["namespaces"][name] = subprocess.run(
                ["/usr/bin/unshare", *flags, "/usr/bin/true"], capture_output=True, timeout=5).returncode == 0
        result["namespace_combined"] = subprocess.run(
            ["/usr/bin/unshare", *FLAGS, "/usr/bin/true"], capture_output=True, timeout=5).returncode == 0
        with tempfile.TemporaryDirectory(prefix="myriad-e17-") as directory:
            root = Path(directory)
            directory = root / "essaim" if ephemeral else root
            directory.mkdir(exist_ok=True)
            if ephemeral:
                (directory / "__init__.py").write_text("", encoding="utf-8")
            module = directory / "local_linux.py"
            module.write_text(source, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("e17_local_linux", module)
            local_linux = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(local_linux)
            runtime, work = root / "runtime", root / "work"
            (runtime / "bin").mkdir(parents=True)
            work.mkdir()
            (runtime / "bin/python").symlink_to("/usr/bin/python3")
            # Install the unchanged seccomp filter in a disposable process, never in the controller.
            script = ("import ctypes,importlib.util,socket\n"
                      f"s=importlib.util.spec_from_file_location('ll',{str(module)!r})\n"
                      "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
                      "assert ctypes.CDLL(None).prctl(38,1,0,0,0)==0\nm.seccomp()\n"
                      "try: socket.socket()\n"
                      "except PermissionError: print('SECCOMP-DENIED')\n")
            check = subprocess.run(["/usr/bin/python3", "-I", "-c", script], capture_output=True,
                                   text=True, timeout=5)
            result["seccomp"] = check.returncode == 0 and check.stdout == "SECCOMP-DENIED\n"
            environment = local_linux.LocalLinuxEnv
            if ephemeral:
                target = directory / "ephemeral_vm.py"
                target.write_text(ephemeral_source, encoding="utf-8")
                spec = importlib.util.spec_from_file_location("e17_ephemeral_vm", target)
                vm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vm)
                result["colab_markers"] = vm.colab_markers()
                try:
                    environment = vm.select_backend("ephemeral-vm")
                except RuntimeError:
                    result["verdict"] = "incompatible"
                    return result
            env = environment(work, runtime)
            try:
                code, output = env.run("python -I -c 'assert 2 + 2 == 4; print(\"E17-SYNTHETIC-OK\")'", 10)
                result["smoke"] = {"code": code, "output_ok": output == "E17-SYNTHETIC-OK\n",
                                   "isolation": env.isolation}
            except RuntimeError:
                result["smoke"] = {"code": None, "output_ok": False, "isolation": None}
            finally:
                env.close()
        if result["smoke"]["code"] == 124:
            return result
        ready = result["smoke"]["isolation"]
        compatible = ((ephemeral or result["landlock_abi"] >= 3) and all(result["namespaces"].values())
                      and result["namespace_combined"] and result["seccomp"]
                      and result["smoke"]["code"] == 0 and result["smoke"]["output_ok"]
                      and ready == {"isolation_ready": "ephemeral-vm-v1" if ephemeral else local_linux.VERSION,
                                    "landlock_abi": None if ephemeral else result["landlock_abi"], "seccomp": "on",
                                    "namespaces": NAMESPACES})
        result["verdict"] = "compatible" if compatible else "incompatible"
    except (OSError, subprocess.TimeoutExpired):
        # Missing runtime/tools or a timed-out observation cannot establish kernel compatibility.
        pass
    return result
