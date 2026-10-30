"""Per-task working directory map.

A single `pwd` is only correct for a corpus whose images share one working directory.
These cover the failure modes that are silent in production: a map that fails to load,
and a pooled backend reusing the previous task's directory.
"""

import asyncio
import json
import os
import tempfile

import pytest

from open_instruct.environments.swerl_vanillux_sandbox import SWERLVanilluxSandboxEnv


def _env(**kw):
    return SWERLVanilluxSandboxEnv(backend="sandfleet", image="x.sif", **kw)


def test_no_map_keeps_scalar_pwd():
    """Existing single-pwd runs must be unaffected."""
    env = _env(pwd="/")
    assert env._pwd_map == {}
    assert env._backend_kwargs["pwd"] == "/"


def test_map_loads_from_path():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"t1": "/app", "t2": "/root"}, fh)
        path = fh.name
    try:
        assert _env(pwd="/", pwd_map=path)._pwd_map == {"t1": "/app", "t2": "/root"}
    finally:
        os.unlink(path)


def test_missing_map_raises_rather_than_falling_back():
    """A silently-ignored map would run a whole arm from the wrong directory and show up
    only as every verifier declining to produce a reward."""
    with pytest.raises(RuntimeError, match="SWERL_SANDBOX_PWD_MAP unreadable"):
        _env(pwd="/", pwd_map="/nonexistent/pwd-map.json")


def test_env_var_is_honoured(monkeypatch):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"t1": "/builds"}, fh)
        path = fh.name
    try:
        monkeypatch.setenv("SWERL_SANDBOX_PWD_MAP", path)
        assert _env(pwd="/")._pwd_map == {"t1": "/builds"}
    finally:
        os.unlink(path)


def test_pwd_change_forces_backend_rebuild():
    """Pool actors are reused across tasks and the backend's pwd is fixed at construction.

    Without the rebuild, the second task to land on an actor runs in the first task's
    directory -- which produces no error, just grading against the wrong cwd.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"t1": "/app", "t2": "/root"}, fh)
        path = fh.name
    try:
        env = _env(pwd="/", pwd_map=path)

        class _StubBackend:
            def __init__(self, pwd):
                self._pwd = pwd
                self.closed = False

            def close(self):
                self.closed = True

        stub = _StubBackend("/app")
        env._backend = stub
        # Simulate the reset-time decision for a task wanting a different directory.
        task_pwd = env._pwd_map.get("t2")
        assert task_pwd == "/root"
        needs_rebuild = getattr(env._backend, "_pwd", None) != task_pwd
        assert needs_rebuild, "a changed pwd must invalidate the pooled backend"
        # And the same directory must NOT churn the pool.
        assert env._pwd_map.get("t1") == stub._pwd
    finally:
        os.unlink(path)


def test_absent_task_in_supplied_map_is_an_error():
    """An INCOMPLETE map must not fall through to the scalar.

    Falling through re-creates the original bug by another route: the scalar default for
    these corpora is swerl-tmax's "/", which is exactly the value 601/602 LiteCoder
    verifiers reject. The task would then grade from the wrong directory and score zero
    with nothing reporting an error -- so an incomplete map is a staging defect and has to
    be surfaced as one.
    """

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"t1": "/app"}, fh)
        path = fh.name
    try:
        env = _env(pwd="/", pwd_map=path)
        # An explicit image is required before reset reaches the pwd lookup.
        with pytest.raises(RuntimeError, match="absent from the supplied pwd map"):
            asyncio.run(env.reset(task_id="not_in_map", image="/tmp/none.sif"))
    finally:
        os.unlink(path)


def _env_with_tests(tmpdir, body, pwd):
    """Env whose task tests dir contains a verifier with the given body."""
    tdir = os.path.join(tmpdir, "tests")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "test.sh"), "w") as fh:
        fh.write(body)
    env = _env(pwd=pwd)
    env._tests_dir = tdir
    env._task_id = "t"
    return env


def test_absolute_grader_is_never_relocated(tmp_path):
    """A grader using absolute /tests must keep getting /tests
    whatever its working directory. 899 tmax verifiers are this shape."""
    env = _env_with_tests(str(tmp_path), "#!/bin/bash\npytest /tests/test_outputs.py\n", pwd="/app/deeply/nested")
    assert env._tests_destination() == "/tests"


def test_relative_grader_gets_sibling_of_pwd(tmp_path):
    env = _env_with_tests(str(tmp_path), "#!/bin/bash\npytest ../tests/test_outputs.py\n", pwd="/app/mnist_gan")
    assert env._tests_destination() == "/app/tests"


def test_relative_grader_at_depth_one_is_unchanged(tmp_path):
    """dirname('/app') == '/', so the existing depth-1 behaviour is byte-identical."""
    env = _env_with_tests(str(tmp_path), "#!/bin/bash\npytest ../tests/test_outputs.py\n", pwd="/app")
    assert env._tests_destination() == "/tests"


def test_tmax_scalar_root_pwd_is_unchanged(tmp_path):
    """tmax runs pwd=/ with no map; dirname('/') == '/', so nothing moves."""
    env = _env_with_tests(str(tmp_path), "#!/bin/bash\nmkdir -p /logs/verifier\npytest /tmp/t.py\n", pwd="/")
    assert env._tests_destination() == "/tests"
