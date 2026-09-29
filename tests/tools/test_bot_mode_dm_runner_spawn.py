"""The DM delivery runner must reach its own dependencies from a dependency-less interpreter.

Regression for the gateway-spawned Bot Chat failure (Brain -> @vex, 2026-09-25): a session
launched through the install launcher spawns the runner with ``sys.executable`` = the PM store
python, whose site-packages carries no project dependencies. ``tools.bot_live_delivery``
(-> ``utils`` -> ``hermes_yaml`` -> ``ruamel.yaml``) then died with
``ModuleNotFoundError: No module named 'ruamel'`` before any state was written, and the send
came back ``ambiguous``: inter-bot messaging was silently broken for exactly the sessions the
gateway launches.

The runner is an entry point like ``cli.py`` / ``gateway/run.py`` / ``batch_runner.py``: its
``__main__`` must import ``hermes_bootstrap`` (which activates the committed dependency
generation, or relaunches onto the managed interpreter) before touching the dependency graph.
The ``--wait-reply`` waiter stays stdlib-only on purpose — it must outlive an install whose
dependencies cannot activate, so it must NOT be routed through the bootstrap.

The child interpreter is spawned ``-S`` with no inherited PYTHONPATH: that is what makes
"dependency-less" true in a test process whose own interpreter has full dependencies. The
dependency tree is reached through a fabricated PM install state whose selected generation
carries a ``.pth`` to the running interpreter's site-packages — the same activation
``hermes_bootstrap`` performs on a real install.
"""
import json
import os
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "tools" / "bot_mode_dm.py"


def _install_key() -> str:
    sys.path.insert(0, str(REPO))
    from pm.environments import install_key

    return install_key(REPO)


def _fabricate_install_state(home: Path) -> Path:
    """A PM install state under *home* whose committed generation carries this interpreter's
    dependency tree.

    Mirrors ``pm.environments``: ``<home>/installs/<install_key>/facts.json`` names the selected
    generation, and what ``activate_dependencies`` adds is its ``site-packages`` — here a real
    directory holding a ``.pth`` that reaches the site-packages running this test. That is the
    contract: the child reaches project dependencies ONLY through the bootstrap's activation,
    never through the interpreter it was spawned as or an inherited PYTHONPATH.
    """
    from pm.environments import site_packages

    generation_venv = home / "installs" / _install_key() / "environments" / ("a" * 32) / "venv"
    generation_venv.mkdir(parents=True, exist_ok=True)
    (generation_venv / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n",
        encoding="utf-8")
    site = site_packages(generation_venv)
    site.mkdir(parents=True, exist_ok=True)
    (site / "deps.pth").write_text(sysconfig.get_paths()["purelib"] + "\n", encoding="utf-8")
    (home / "installs" / _install_key() / "facts.json").write_text(
        json.dumps({"schema": 1, "packages": {"venv": {"environment": str(generation_venv)}}}),
        encoding="utf-8")
    return generation_venv


def _dependent_child(tmp_path: Path) -> tuple[Path, dict]:
    """(home, env) for a runner child in the shape a terminal-tool spawn produces.

    ``PYTHONPATH``/``PYTHONHOME``/``VIRTUAL_ENV`` are absent (``local_pythonpath`` strips
    Hermes-owned entries from every child), and the child is started ``-S`` so its own
    site-packages cannot paper over a missing bootstrap.
    """
    home = tmp_path / ".hermes"
    (home / "profiles" / "target").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "__HERMES_ACTIVATED")}
    env.update(HERMES_HOME=str(home), HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               HERMES_DISABLE_LAZY_INSTALLS="1", HERMES_TEST_ISOLATION=str(home),
               PYTHONDONTWRITEBYTECODE="1")
    return home, env


def _run_runner(args: list[str], env: dict, *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-S", str(RUNNER), *args], env=env, cwd=str(cwd),
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def dependent_child(tmp_path):
    """(dependency-less home, child env) — the store-python shape, machine-independent."""
    return _dependent_child(tmp_path)


def test_runner_entry_point_delivers_from_a_dependency_less_interpreter(dependent_child, tmp_path):
    """From an interpreter that cannot import the runner's dependency chain itself, the runner
    must still execute ``--run-delivery`` to success — dependencies arriving only through
    ``hermes_bootstrap``'s activation of the committed generation.

    End-to-end on the real entry point (no mocked imports): the child runs the real
    ``_delivery_main`` -> ``_run_delivery`` -> transport chain against a stub transport, and
    success is observed as exit 0 + the transport's stdout + the DM file's cleanup. The
    premise (the child really cannot import the chain unaided) is asserted first, so a leaked
    PYTHONPATH cannot make this pass vacuously.
    """
    home, env = dependent_child
    _fabricate_install_state(home)

    premise = subprocess.run(
        [sys.executable, "-S", "-c",
         f"import sys; sys.path.insert(0, {str(REPO)!r}); import tools.bot_live_delivery"],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
    assert premise.returncode != 0 and "No module named" in premise.stderr, (
        f"premise broken: the child could reach the runner's dependency chain without bootstrapping "
        f"(rc={premise.returncode}, stderr={premise.stderr[:400]})")

    dm_file = home / "payload.txt"
    dm_file.write_bytes("héllo 世界".encode("utf-8"))
    observed = tmp_path / "observed.txt"
    transport = [sys.executable, "-S", "-c", textwrap.dedent(
        f"""
        import pathlib, sys
        pathlib.Path({str(observed)!r}).write_bytes(sys.stdin.buffer.read())
        print("TRANSPORT-OK")
        """)]

    proc = _run_runner(["--run-delivery", "stdin", str(dm_file), *transport], env, cwd=tmp_path)

    assert proc.returncode == 0, (
        f"runner failed from a dependency-less interpreter.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
    assert observed.read_bytes() == "héllo 世界".encode("utf-8")
    assert "TRANSPORT-OK" in proc.stdout
    assert not dm_file.exists(), "a settled delivery gives up cleanup ownership to the runner"


def test_wait_reply_stays_stdlib_only_for_a_broken_dependency_install(dependent_child, tmp_path):
    """The relay reply waiter must keep running when dependencies CANNOT activate.

    ``--wait-reply`` is deliberately stdlib-only so it survives an install whose dependency
    generation is broken — bootstrap refuses in exactly that state, so routing the waiter
    through it would trade a working reply notification for a hard exit. The fabricated state
    here names a generation that does not exist; the waiter must still reach its own normal
    outcome (no reply within the budget) with no dependency failure on stderr.
    """
    home, env = dependent_child
    broken = home / "installs" / _install_key()
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "facts.json").write_text(
        json.dumps({"schema": 1, "packages": {"venv": {
            "environment": str(broken / "environments" / ("f" * 32) / "venv")}}}),
        encoding="utf-8")

    proc = _run_runner(["--wait-reply", str(tmp_path / "never.json"), "@target on peer", "1"],
                       env, cwd=tmp_path)

    assert proc.returncode == 1, f"unexpected waiter outcome: rc={proc.returncode} stderr={proc.stderr}"
    assert "No reply from" in proc.stdout
    assert "dependency" not in proc.stderr.lower(), (
        "the waiter went through the dependency bootstrap and died on a broken install: "
        f"{proc.stderr[:400]}")
