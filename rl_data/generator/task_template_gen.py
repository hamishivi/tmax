"""Ask GPT for a task‑description *template* + parameter schema."""
from __future__ import annotations

import json
import uuid
import random
import re
from pathlib import Path

from rl_data import chat_completion_batch, DEFAULT_MODEL

SYSTEM_MSG = """You are creating realistic Linux-terminal tasks for training an AI agent.

Respond in xml format.

<task>
        Be detailed here. Give the names of the precise contents of files, ports, directories, etc.
        This should be a very detailed description of the final state of the system.
        For example, if you are asking the agent to create a log file, you should precisely specify the format it should be in so that an automated test can verify it.
        Ask the agent to create a log file whenever some verification is required.
        You only have about 1000-1500 words to work with. So balance between conciseness and detail.
        DO NOT directly give the commands to the agent.
</task>

<truth>
        Insert *privileged* ground-truth data that automated
        test suites will rely on to verify correct task execution.
        These values **must NOT** appear in the public task
        description.

        Be very detailed here. Give the names / placeholders of the precise contents of files, ports, directories, repositories, websites etc.
        This should be a very detailed description of the final state of the system.
        For example, if you are asking the agent to create a log file, you should give the name of the log file and the contents of the log file.
        Any processes, files, directories that should be created before the task starts should be mentioned here.
        Any files that should be created by the agent and their contents should be mentioned here.
</truth>

Guidelines:
* Place any secret, ground-truth verification data exclusively under the <truth> element.
* The agent should be able to write to the file and directory that are mentioned in the task description.
* The agent will not have root access. So make sure that the right permissions are set for the files and directories.
* When you mention a file or directory, write the full path to the file or directory, not just relative path.
* The task must be a realistic end-to-end scenario that an AI agent could perform in a Linux terminal. 
* Write the task description in a way that a user might ask an AI assistant.
* Be very specific about the names, paths and contents of the files and directories.
* We will be using apptainer to run the agent. So make sure that the task is valid when the container is built.
* Don't create tasks that require having the latest information.
* The home path is /home/user.
* Don't create tasks the setup of which will require su access.
* The task is multi-turn, so the agent will interact in a terminal to finish the task.
* Don't discourage the agent from using console output to finish the task.
* Do not put a constraint on the number of commands that the agent can use (the complixity that the user provides is for the complexity of the task description)."""


# --- User-message template & combinatorial variation helpers ---------------------------------

# Template with placeholders that can be filled combinatorially to diversify the prompt sent to
# the language model.  Each run of this script will randomly choose values from the option sets
# below, providing a broad variety of task prompts.

# --- Task inspiration for diversity ---
TASK_CATEGORIES = [
    "file and directory management",
    "text processing and manipulation",
    "system monitoring and diagnostics",
    "package management",
    "user and permission management",
    "network diagnostics",
    "log analysis",
    "backup and archiving",
    "process management",
    "disk usage analysis",
    "environment configuration",
    "data transformation",
    "scheduled tasks and cron jobs",
    "service configuration",
    "database operations",
    "container management",
    "git repository operations",
    "security scanning",
    "performance benchmarking",
    "remote file synchronization",
    "API testing and curl operations",
    "certificate management",
    "DNS and hostname resolution",
    "firewall configuration",
    "shell scripting automation",
    "symbolic link management",
    "file compression and extraction",
    "checksum verification",
    "text encoding conversions",
    "CSV/JSON data manipulation",
    "YAML and TOML configuration editing",
    "INI configuration parsing",
    "regex-based log filtering",
    "SQLite database operations via CLI",
    "SSH keypair generation and management",
    "GPG file encryption and signature verification",
    "time zone and locale configuration",
    "cron and systemd timer authoring (user)",
    "Python virtual environment setup with venv",
    "pip package environment management",
    "git submodule management",
    "semantic version bumping and changelogs",
    "Makefile authoring and task automation",
    "text diffing and patch application",
    "markdown documentation generation and linting",
    "environment variable and dotenv management",
    "JSON schema validation and jq processing",
    "find and xargs batch file operations",
    "awk and sed text processing",
    "sort and uniq frequency counting",
    "cut and paste column manipulation",
    "complex permissions management",
    "dev environment setup",
    "headless browser data scraping",
    "distributed system debugging",
    "data pipeline with error recovery",
    "exploiting/fixing security vulnerabilities",
    "performance optimization",
    "running old code",
    "database migration with data validation",
    "launch a webserver",
    "optimization solvers"
]

COMPLEXITY_LEVELS = [
    "simple single terminal command",
    "simple set of 2-3 commands",
    "simple set of 3-4 commands",
    "multi-step sequential commands",
    "multi-step parallel commands",
    "set of 5-10 commands"
]


SCENARIO_CONTEXTS = [
    "developer organizing project files",
    "system administrator maintaining servers",
    "data analyst processing CSV files",
    "DevOps engineer debugging logs",
    "security auditor checking permissions",
    "backup administrator archiving data",
    "researcher organizing datasets",
    "web developer",
    "database administrator optimizing queries",
    "network engineer troubleshooting connectivity",
    "release manager preparing deployments",
    "QA engineer setting up test environments",
    "cloud architect migrating services",
    "site reliability engineer monitoring uptime",
    "data engineer building ETL pipelines",
    "compliance officer auditing systems",
    "technical writer organizing documentation",
    "machine learning engineer preparing training data",
    "penetration tester scanning vulnerabilities",
    "infrastructure engineer automating provisioning",
    "support engineer collecting diagnostics",
    "build engineer managing artifacts",
    "configuration manager tracking changes",
    "capacity planner analyzing resource usage",
    "incident responder investigating issues",
    "automation specialist creating workflows",
    "integration developer testing APIs",
    "performance engineer profiling applications",
    "container specialist managing microservices",
    "backup engineer verifying data integrity",
    "monitoring specialist setting up alerts",
    "deployment engineer rolling out updates",
    "storage administrator managing disk space",
    "log analyst investigating patterns",
    "script developer creating utilities",
    "observability engineer tuning dashboards",
    "data scientist cleaning datasets",
    "site administrator managing user accounts",
    "operations engineer triaging incidents",
    "IT support technician resolving tickets",
    "platform engineer maintaining CI/CD pipelines",
    "FinOps analyst optimizing cloud costs",
    "DevSecOps engineer enforcing policy as code",
    "localization engineer updating translations",
    "MLOps engineer tracking experiment artifacts",
    "edge computing engineer deploying to IoT devices",
    "mobile build engineer maintaining pipelines",
    "security engineer rotating credentials",
    "compliance analyst generating audit trails",
    "backup operator testing restores",
    "database reliability engineer managing backups",
    "linux systems engineer hardening configurations",
    "kubernetes operator managing manifests",
    "artifact manager curating binary repositories",
]

def random_user_msg() -> tuple[str, dict]:
    """Generate a user instruction by randomly selecting inspiration elements.

    Returns (prompt_string, metadata_dict) where metadata_dict has keys
    'category', 'complexity', and 'scenario'.
    """
    category = random.choice(TASK_CATEGORIES)
    complexity = random.choice(COMPLEXITY_LEVELS)
    scenario = random.choice(SCENARIO_CONTEXTS)

    msg = (
        f"Write a new task focusing on {category}. "
        f"Complexity: {complexity}. "
        f"Scenario: {scenario}. "
        "Be very specific about the output format in the task description that the automated test will check. "
        "Write the task description in a way that a user might ask an AI assistant. "
        "The task should be a realistic end-to-end scenario that an AI agent could perform in a Linux terminal."
    )
    metadata = {"category": category, "complexity": complexity, "scenario": scenario}
    return msg, metadata


def generate_templates_batch(
    batch_size: int,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_concurrency: int = 128,
) -> list[dict]:
    """Generate multiple task templates in one batched LLM call set.

    Returns a list of dicts with keys ``description``, ``truth``,
    ``category``, ``complexity``, and ``scenario``.
    """

    messages: list[list[dict[str, str]]] = []
    metadata_list: list[dict] = []
    for _ in range(batch_size):
        user_msg, metadata = random_user_msg()
        messages.append([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ])
        metadata_list.append(metadata)

    responses = chat_completion_batch(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    results: list[dict] = []
    for i, resp in enumerate(responses):
        if resp is None:
            continue
        try:
            content = resp.choices[0].message.content.strip()
            parsed = parse_template(content)
            parsed.update(metadata_list[i])
            results.append(parsed)
        except Exception:
            continue
    return results

def parse_template(raw: str) -> dict:
    """Convert the raw XML *raw* into a structured ``dict``."""

    # Extract the task description template
    template = re.search(r"<task>(.*?)</task>", raw, re.DOTALL).group(1).strip()
    if not template:
        raise ValueError("No task description found in the response.")

    # Extract ground-truth section (optional)
    truth_data = re.search(r"<truth>(.*?)</truth>", raw, re.DOTALL).group(1).strip()
    if not truth_data:
        raise ValueError("No truth data found in the response.")

    return {"description": template, "truth": truth_data}


if __name__ == "__main__":


    tasks = generate_templates_batch(
        batch_size=100,
        model=DEFAULT_MODEL,
        temperature=1.0,
        max_tokens=2048,
        max_concurrency=64,
    )
    # save the tasks to a file
    for task in tasks:
        task_name = str(uuid.uuid4())
        task_path = Path("tasks") / task_name
        task_path.mkdir(parents=True, exist_ok=True)
        with open(task_path / "task.json", "w") as f:
            json.dump(task, f, indent=4)
