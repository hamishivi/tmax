"""Create an Apptainer .def *template* and iterate until tests pass – with masking."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
import re
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from rl_data import chat_completion_batch, DEFAULT_MODEL

SYSTEM_MSG = """ You are an expert in Apptainer/Singularity.
You are given a task description and will be tested so that the initial state of the container is set up in a way that an agent can be tested on the task.
Make sure that the container is set up in a way that an agent can be tested on the task.
Basically ensure that the task is valid when the container is built: Clone a repository, create a file, create a directory, create a process, etc.
Install pytest in the container.
Don't include the tests in the response (no %test)
The agent will not have root access. So make sure that the right permissions are set for the files and directories.
Always use this base image by putting these two lines at the top of the def file:
Bootstrap: docker
From: ubuntu:22.04"""

BASE_USER_TEMPLATE = """
Using the task description template and pytest failures below, output a complete
Apptainer `.def` file.

Question description given to the agent:
{task_description}

Here is some ground truth data that might be useful to you:
{truth}

Here are the tests that will be run on the container:
{test_py}

Previous failures (may be empty):
{failures}

Respond with the Apptainer `.def` file only. You should think step by step and then write the file. The file should be valid and buildable.
Make sure that you create the right files and directories for the task.
Eg: for a csv task you will have to create a csv file. For a process cleanup task you will have to create processes.
Don't include the tests in the response or copy a test file.
Don't add any of the output files or directories that the student will create.
Don't create / touch empty files for the agent.
Remember to install pytest in the container.
The home path is /home/user.
Don't override HOME in the %environment section; let Apptainer bind the host $HOME.
"""


def build_and_test(def_template: str, test_py: str) -> tuple[bool, str]:
    """Build an Apptainer image from a definition *template* and run the
    supplied pytest code inside the container.

    Parameters
    ----------
    def_template:
        The text contents of the Apptainer ``.def`` file to build.
    test_py:
        The pytest test module (as a string) that should be executed inside
        the freshly-built container to validate its initial state.
    """
    # Create an isolated workspace for the build and test run.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # ------------------------------------------------------------------
        # 1. Persist the definition template and the test module to disk
        # ------------------------------------------------------------------
        def_path = td_path / "container.def"
        def_path.write_text(def_template)

        test_file = td_path / "test_initial_state.py"
        test_file.write_text(test_py)

        # ------------------------------------------------------------------
        # 2. Build the container image from the .def file
        # ------------------------------------------------------------------
        sif_path = td_path / "img.sif"
        build_proc = subprocess.run(
            ["apptainer", "build", str(sif_path), str(def_path)],
            capture_output=True, text=True, timeout=180,
        )
        if build_proc.returncode:
            err_snippet = (build_proc.stderr or build_proc.stdout or "")[-500:]
            print(f"Apptainer build failed (rc={build_proc.returncode}): {err_snippet}")
            return False, f"Apptainer build failed: {err_snippet}"

        # copy the test file to the container at /home/agent/test_initial_state.py
        # shutil.copy(test_file, td_path / "home" / "agent" / "test_initial_state.py")

        # ------------------------------------------------------------------
        # 3. Execute the provided pytest module inside the container
        # ------------------------------------------------------------------
        proc = subprocess.run(
            [
                "apptainer",
                "exec",
                "--fakeroot",
                "--userns",
                "--writable-tmpfs",
                "--cleanenv",
                str(sif_path),
                "pytest",
                "-q",
                str(test_file.name),
            ],
            cwd=td,  # Ensure the test module is visible inside the container
            capture_output=True,
            text=True,
        )

        # Remove the SIF image first, then clean up the temporary directory.
        if sif_path.exists():
            sif_path.unlink()

        # Now remove the temporary directory; ignore errors in case it's
        # already gone or cleaned up by the TemporaryDirectory context
        shutil.rmtree(td_path, ignore_errors=True)
        # ------------------------------------------------------------------
        # 4. Return success flag and combined stdout/stderr for inspection
        # ------------------------------------------------------------------
        return proc.returncode == 0, proc.stdout + proc.stderr


def parse_def_template(def_template: str) -> str:
    """
    Clean up the raw response from the language model and return a valid
    Apptainer definition template string.

    The model is expected to reply with only the content of a .def file, yet
    it may still wrap the output in markdown code fences (e.g. ```def or
    ```singularity) or include explanatory text. This helper extracts the first
    fenced code block if present; otherwise it assumes the entire response is
    the definition. Finally, common leading indentation is removed so the
    template can be written directly to disk.
    """
    # Normalise line endings and trim outer whitespace
    cleaned = def_template.replace("\r\n", "\n").strip()

    # Attempt to extract the first fenced code block
    fence_re = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(?P<code>[\s\S]*?)```", re.MULTILINE)
    match = fence_re.search(cleaned)
    if match:
        cleaned = match.group("code").strip()

    # Remove any common leading indentation
    cleaned = textwrap.dedent(cleaned).strip()
    return cleaned

def iterate_def_template_batch(
    items: List[Tuple[str, str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[Optional[str]]:
    """Batched single-shot def generation followed by parallel build/test.

    items: list of (task_description, truth, test_py)
    Returns list aligned with input: the first passing def text per item, or None if fails.
    """

    messages: list[list[dict[str, str]]] = []
    for task_description, truth, test_py in items:
        prompt = BASE_USER_TEMPLATE.format(
            task_description=task_description,
            truth=truth,
            test_py=test_py,
            failures="None yet",
        )
        messages.append([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ])

    responses = chat_completion_batch(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    # Prepare results aligned with input order
    results: List[Optional[str]] = [None] * len(items)

    def worker(index: int, item: Tuple[str, str, str], resp_obj) -> Tuple[int, Optional[str]]:
        try:
            if resp_obj is None:
                return index, None
            content = resp_obj.choices[0].message.content
            def_text = parse_def_template(content)
            _task_description, _truth, test_py = item
            ok, _ = build_and_test(def_text, test_py)
            return index, (def_text if ok else None)
        except Exception:
            return index, None

    # Submit parallel build/test tasks
    futures = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for idx, (item, resp) in enumerate(zip(items, responses)):
            futures.append(executor.submit(worker, idx, item, resp))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, value = fut.result()
            results[idx] = value

    return results


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------
    # Load task template and sample concrete parameters
    # ------------------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = ap.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"
    final_test_path = task_path / "test_final_state.py"

    test_py = initial_test_path.read_text()
    def_text = def_path.read_text()

    success, output = build_and_test(def_text, test_py)
    print(success)
    print(output)
    
