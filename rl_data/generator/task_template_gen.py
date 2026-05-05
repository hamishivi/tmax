"""Task template generation with domain-specific skill taxonomy.

Implements the Terminal-Task-Gen approach (Pi et al. 2026): domain modules,
primitive skill composition, two-axis complexity, and domain-tied personas.
"""
from __future__ import annotations

import json
import uuid
import random
import re
from pathlib import Path

from rl_data import chat_completion_batch, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Skill Taxonomy: 9 domains -> skill types -> primitive skills
# Based on Pi et al. 2026 Table 10 and Terminal-Bench 2.0 task distribution.
# ---------------------------------------------------------------------------

SKILL_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "security": {
        "Systems": [
            "Firewall and network policy configuration",
            "SSH hardening and key management",
            "File permission and access control",
            "Service auditing and port scanning",
            "Process isolation and sandboxing",
        ],
        "Data Processing": [
            "Security log parsing and correlation",
            "Payload encoding and decoding",
            "Certificate chain validation",
            "Sensitive data redaction",
            "Binary format and ELF analysis",
        ],
        "Web Security": [
            "Exploit crafting and payload delivery",
            "Injection and XSS vulnerability analysis",
            "HTTP header and cookie inspection",
            "TLS/SSL certificate management",
            "Content security policy enforcement",
        ],
        "Algorithmic": [
            "Cryptographic hashing and checksum verification",
            "Password cracking and brute-force search",
            "Encryption and decryption",
            "Token generation and validation",
            "Pattern matching for intrusion detection",
            "Reverse engineering and disassembly",
            "Cryptanalysis (differential, linear)",
        ],
        "Testing": [
            "Automated vulnerability scanning",
            "Authentication flow testing",
            "File integrity verification",
            "Privilege escalation auditing",
            "CWE identification and code auditing",
        ],
    },
    "software_engineering": {
        "Algorithmic": [
            "Graph traversal and dependency resolution",
            "Custom data structure design",
            "Sorting, merging, and diffing",
            "State machine and parser construction",
            "Constraint satisfaction",
            "Interpreter and emulator implementation",
        ],
        "Systems": [
            "Build system configuration and linking",
            "Shared library and ABI management",
            "Cross-compilation and conditional builds",
            "CI/CD pipeline setup",
            "Memory debugging and profiling",
            "Package and dependency management",
        ],
        "Data Processing": [
            "Structured data parsing and transformation",
            "Serialization and deserialization",
            "Schema migration",
            "Diff and patch processing",
            "Code translation between languages",
        ],
        "Web Security": [
            "REST/GraphQL API construction",
            "Request validation and rate limiting",
            "Reverse proxy configuration",
            "WebSocket communication",
            "URL routing and parameter parsing",
            "gRPC and protobuf service design",
        ],
        "Testing": [
            "Unit and integration testing",
            "Test fixture and mock setup",
            "Property-based testing",
            "Performance benchmarking",
            "End-to-end test orchestration",
        ],
        "Mathematical": [
            "Numerical algorithm implementation",
            "Expression parsing and evaluation",
            "Semantic version comparison",
            "Checksum and error-correcting codes",
            "Character and data encoding",
        ],
        "Multi-Language": [
            "C program compilation, debugging, and Makefile repair",
            "Rust ownership and borrow checker debugging",
            "Go concurrency patterns (goroutines, channels)",
            "C/C++ memory safety and undefined behaviour repair",
            "Polyglot build orchestration (multiple compiled languages)",
            "FFI and cross-language interop (C bindings, ctypes, cgo)",
            "Assembly-level analysis and minimal program construction",
        ],
    },
    "file_operations": {
        "File I/O": [
            "Text and binary file reading/writing",
            "File locking and concurrent access",
            "Streaming and memory-mapped I/O",
            "Atomic writes and temp file management",
            "Standard stream redirection and piping",
        ],
        "Navigation": [
            "Recursive directory traversal",
            "Symbolic and hard link management",
            "File watching and change detection",
            "Filesystem mount and path manipulation",
            "Metadata-based file search",
        ],
        "Data Parsing": [
            "Structured format parsing (JSON, XML, CSV)",
            "Binary format and header extraction",
            "Multi-line log record parsing",
            "Character encoding conversion",
            "Configuration file interpretation",
            "Domain-specific format parsing (GCode, ELF, WAL)",
        ],
        "Transformation": [
            "Bulk file renaming",
            "Format conversion between file types",
            "Text transformation with sed/awk/vim",
            "File splitting, merging, and chunking",
            "Manifest and checksum generation",
            "Large-scale text editing and macro application",
        ],
        "Archives": [
            "Archive creation and extraction",
            "Nested and multi-part archive handling",
            "Incremental and differential backups",
            "Archive integrity verification",
            "Compressed stream processing",
            "Custom compression and decompression",
        ],
    },
    "data_querying": {
        "Query Construction": [
            "Complex joins and subqueries",
            "Window functions and analytical aggregation",
            "Parameterized query construction",
            "Graph query languages (SPARQL, Cypher)",
            "NoSQL aggregation pipelines",
        ],
        "Data Comprehension": [
            "Schema analysis and relationship mapping",
            "Query plan interpretation and optimization",
            "Index strategy design",
            "Data model reverse engineering",
            "Cross-representation mapping (relational, document, graph)",
        ],
        "Graph Processing": [
            "Graph traversal and shortest path computation",
            "Recursive and hierarchical queries",
            "Knowledge graph pattern matching",
            "Graph analytics (centrality, clustering)",
            "Graph projection and materialization",
        ],
        "Result Processing": [
            "Result sorting, pagination, and filtering",
            "Query result export and format conversion",
            "Cross-query aggregation and summarization",
            "Output schema validation",
            "Query-to-pipeline chaining",
        ],
    },
    "data_science": {
        "Systems": [
            "Analysis environment setup",
            "Numerical library configuration",
            "Large-scale data storage management",
            "Experiment tracking",
            "Reproducible pipeline construction",
            "Package and dependency installation",
        ],
        "Data Processing": [
            "Tabular data transformation and aggregation",
            "Missing value and outlier handling",
            "Multi-source data joining",
            "Feature engineering and selection",
            "ETL pipeline construction",
            "Tokenization and dataset preparation",
        ],
        "Algorithmic": [
            "Classification and regression",
            "Cross-validation and hyperparameter tuning",
            "Similarity search and recommendation",
            "Dimensionality reduction",
            "Model training and evaluation",
            "Model architecture reconstruction and inference",
            "Embedding computation and retrieval",
        ],
        "Mathematical": [
            "Hypothesis testing and confidence intervals",
            "Bayesian inference and probabilistic modeling",
            "Correlation and covariance analysis",
            "Sampling and bootstrap methods",
            "Linear algebra for data analysis",
        ],
        "Testing": [
            "Model output validation",
            "Pipeline reproducibility testing",
            "Data schema enforcement",
            "Numerical accuracy testing",
            "Inference performance benchmarking",
        ],
    },
    "debugging": {
        "Systems": [
            "Build failure diagnosis",
            "System call tracing",
            "Container debugging and log inspection",
            "Dependency conflict resolution",
            "Environment misconfiguration repair",
            "Compiler and linker error interpretation",
        ],
        "Debugging": [
            "Interactive debugger usage (breakpoints, memory inspection)",
            "Core dump and stack trace analysis",
            "Logging and traceback analysis",
            "Git bisection for regression finding",
            "Delta debugging and test minimization",
            "Existing codebase comprehension and reading",
        ],
        "Testing": [
            "Regression test construction",
            "Intermittent failure reproduction",
            "Fuzz testing",
            "Minimal reproducible example creation",
            "Assertion-based intermediate validation",
        ],
        "Algorithmic": [
            "Boundary condition and off-by-one repair",
            "Race condition and concurrency debugging",
            "Intermediate state tracing",
            "Floating-point precision repair",
            "Loop termination and recursion fixing",
        ],
        "Data Processing": [
            "Corrupted input handling and recovery",
            "Encoding and serialization troubleshooting",
            "Data transformation diff analysis",
            "Query result debugging",
            "Format parsing edge-case repair",
        ],
        "Mathematical": [
            "Numerical instability diagnosis",
            "Convergence failure repair",
            "Formula implementation correction",
            "Statistical anomaly investigation",
            "Precision loss tracking",
        ],
        "Forensics": [
            "Database recovery from corrupted files (WAL, journal)",
            "Deleted file recovery and filesystem inspection",
            "Binary reverse engineering and decompilation",
            "Network packet capture analysis (pcap)",
            "Git history forensics and secret recovery",
            "Memory dump analysis and string extraction",
            "Log timeline reconstruction across services",
        ],
    },
    "scientific_computing": {
        "Data Processing": [
            "Scientific data format I/O (NetCDF, HDF5, FITS)",
            "Experimental data visualization",
            "Multi-dimensional array manipulation",
            "Bioinformatics format parsing (FASTA, PDB)",
            "Observational data reshaping",
            "Spectroscopy and signal data processing",
        ],
        "Algorithmic": [
            "ODE/PDE numerical solving",
            "Monte Carlo simulation",
            "Optimization (gradient descent, simplex, genetic)",
            "Mesh refinement and domain decomposition",
            "Graph algorithms for molecular/network models",
            "Primer design and sequence alignment",
        ],
        "Mathematical": [
            "Probability distribution distance metrics",
            "Matrix decomposition (SVD, LU, QR, Cholesky)",
            "Numerical integration and differentiation",
            "Linear and nonlinear equation solving",
            "Fourier transforms and spectral analysis",
        ],
        "Systems": [
            "Scientific software compilation from source",
            "Parallel computing setup (MPI, OpenMP)",
            "Scientific environment management",
            "Notebook-based workflow orchestration",
            "Reproducible computation pipelines",
        ],
        "Testing": [
            "Analytical solution validation",
            "Convergence testing",
            "Reference dataset comparison",
            "Scientific code regression testing",
            "Numerical stability testing",
        ],
        "Statistical": [
            "MCMC sampling and posterior estimation",
            "Curve fitting and regression",
            "Statistical hypothesis comparison",
            "Bootstrap confidence intervals",
            "Density estimation and distribution fitting",
        ],
    },
    "data_processing": {
        "Data I/O": [
            "Multi-format file reading and writing",
            "Large-file streaming",
            "Character encoding handling",
            "Database bulk import/export",
            "Local-remote data transfer",
        ],
        "Manipulation": [
            "Cleaning, normalization, and deduplication",
            "Joins, merges, and unions",
            "Wide-long format reshaping",
            "Windowed and rolling aggregation",
            "Data masking and anonymization",
        ],
        "String/Text": [
            "Structured information extraction from text",
            "Regex pattern construction",
            "Tokenization and normalization",
            "Unicode and multi-language text processing",
            "Template-based text generation",
        ],
        "Algorithmic": [
            "Large-scale sorting and grouping",
            "Constraint-based data validation",
            "Hash-based deduplication",
            "Data sampling and stratification",
            "Pipeline DAG orchestration",
        ],
        "Mathematical": [
            "Summary statistics and aggregation",
            "Interpolation and imputation",
            "Normalization and standardization",
            "Feature extraction transforms",
            "Distance and similarity computation",
        ],
        "Time Series": [
            "Timestamp alignment and parsing",
            "Resampling and gap-filling",
            "Rolling statistics computation",
            "Anomaly and changepoint detection",
            "Time-based bucketing and aggregation",
        ],
        "Systems": [
            "Multi-stage pipeline orchestration",
            "Parallel data processing",
            "Pipeline scheduling and cron management",
            "Validation checkpoints and quality gates",
            "Pipeline logging and monitoring",
        ],
    },
    "system_administration": {
        "Filesystem": [
            "Permission and ACL management",
            "Disk quota and storage monitoring",
            "Mount and fstab configuration",
            "Backup and restore strategies",
            "Link and directory structure management",
        ],
        "Process/Service": [
            "Service lifecycle management (systemd, init)",
            "Process monitoring and control",
            "Scheduled task configuration (cron, timers)",
            "Process supervision and restart policies",
            "Container lifecycle management",
        ],
        "Network": [
            "Network interface and routing configuration",
            "Firewall rules and port forwarding",
            "Connectivity diagnostics",
            "SSH tunneling and port forwarding",
            "Web server setup and TLS configuration",
            "Email and mailing list server configuration",
        ],
        "Configuration": [
            "System config file management",
            "Environment variable and shell profile setup",
            "Locale and timezone configuration",
            "User account and group administration",
            "Log configuration and rotation",
        ],
        "Deployment": [
            "CI/CD pipeline construction",
            "Reverse proxy and load balancer setup",
            "Rolling and staged deployments",
            "Health check and monitoring setup",
            "Virtualization and VM management (QEMU, VNC)",
            "Git server and hook configuration",
        ],
        "Shell Scripting": [
            "Robust script writing with error handling",
            "Text processing pipelines (awk, sed, grep)",
            "Task automation scripting",
            "Interactive script construction",
            "Idempotent configuration scripting",
            "Expect scripting for interactive automation",
        ],
    },
}


# ---------------------------------------------------------------------------
# Domain Module Prompts (injected into system prompt per domain)
# Following Pi et al. 2026, Figures 12-20.
# ---------------------------------------------------------------------------

DOMAIN_MODULES: dict[str, str] = {
    "security": """\
# Security Task Builder

Domain Focus
Create tasks involving:
- **Cryptography**: Encryption, decryption, key management, hash functions
- **Vulnerability Analysis**: Code review, exploit identification, security auditing
- **Authentication**: Password handling, token validation, session management
- **Network Security**: Protocol analysis, traffic inspection, firewall rules
- **Secure Coding**: Input validation, output encoding, secure defaults

The task should test security skills and require security knowledge and analytical thinking.""",

    "software_engineering": """\
# Software Engineering Task Builder

Domain Focus
Create tasks involving:
- **Code quality**: Refactoring, testing, documentation
- **Build systems**: Compilation, linking, packaging
- **Version control**: Git operations, merge conflicts, history analysis
- **API design**: REST, GraphQL, protocol design
- **Architecture**: Patterns, modularity, scalability

The task should test software engineering skills and require domain knowledge \
and analytical thinking.""",

    "file_operations": """\
# File Operations Task Builder

Domain Focus
Create tasks involving:
- **File I/O**: Reading, writing, appending, seeking
- **Directory operations**: Traversal, creation, permissions
- **File formats**: Binary, text, structured data
- **Compression**: Zip, tar, gzip, custom formats
- **File system operations**: Links, permissions, metadata

The task should test file operations skills and require domain knowledge \
and analytical thinking.""",

    "data_querying": """\
# Data Querying Task Builder

Domain Focus
Create tasks involving:
- **SQL operations**: Complex joins, window functions, CTEs
- **Query optimization**: Indexes, execution plans, performance
- **Database operations**: Schema design, migrations, constraints
- **NoSQL patterns**: Document, key-value, graph queries
- **Data retrieval**: Pagination, filtering, full-text search

The task should test data querying skills and require domain knowledge \
and analytical thinking.""",

    "data_science": """\
# Data Science Task Builder

Domain Focus
Create tasks involving:
- **Exploratory Analysis**: Statistical summaries, visualization, pattern discovery
- **Feature Engineering**: Transformation, encoding, selection, creation
- **Statistical Modeling**: Regression, hypothesis testing, Bayesian analysis
- **Data Mining**: Clustering, association rules, anomaly detection
- **Reporting**: Automated insights, metric computation, summary generation

The task should test data science skills and require statistical thinking \
and data intuition.""",

    "debugging": """\
# Debugging Task Builder

Domain Focus
Create tasks involving:
- **Error diagnosis**: Stack traces, logs, error messages
- **Root cause analysis**: Bisection, delta debugging
- **Performance debugging**: Profiling, bottleneck identification
- **Memory issues**: Leaks, corruption, allocation problems
- **Concurrency bugs**: Race conditions, deadlocks, livelocks

The task should test debugging skills and require domain knowledge \
and analytical thinking.""",

    "scientific_computing": """\
# Scientific Computing Task Builder

Domain Focus
Create tasks involving:
- **Numerical simulation**: ODEs, PDEs, Monte Carlo
- **Signal processing**: FFT, filtering, spectral analysis
- **Statistical analysis**: Hypothesis testing, regression, sampling
- **Visualization**: Plotting, data exploration
- **Domain-specific**: Physics, biology, chemistry applications

The task should test scientific computing skills and require domain knowledge \
and analytical thinking.""",

    "data_processing": """\
# Data Processing Task Builder

Domain Focus
Create tasks involving:
- **File format handling**: CSV, JSON, XML, Parquet, binary formats
- **Data transformation**: Cleaning, normalization, aggregation
- **ETL pipelines**: Extract, transform, load workflows
- **Stream processing**: Real-time data handling
- **Data validation**: Schema enforcement, error handling

The task should test data processing skills and require domain knowledge \
and analytical thinking.""",

    "system_administration": """\
# System Administration Task Builder

Domain Focus
Create tasks involving:
- **Process management**: Services, daemons, scheduling
- **Network configuration**: Routing, firewall, DNS
- **Storage management**: Filesystems, RAID, backups
- **Monitoring**: Logging, alerting, metrics
- **Automation**: Scripts, configuration management

The task should test system administration skills and require domain knowledge \
and analytical thinking.""",
}


# ---------------------------------------------------------------------------
# Two-axis complexity
# ---------------------------------------------------------------------------

TASK_COMPLEXITY: list[str] = [
    "short task (a few shell commands focused on the core task)",
    "moderate task (several commands across setup, implementation, and verification)",
    "complex task (many commands spanning multiple phases: dependency installation, code writing, configuration, building, and testing)",
    # Intricate (v2 only). Calibrated to push the agent toward the ~40-turn
    # Terminal-Bench 2.0 mean (vs ~10 for the legacy 10k corpus). Composes
    # naturally with verifier_kind=metric_threshold / adversarial_corpus etc.
    (
        "intricate task (multi-stage workflow combining (a) environment setup or "
        "package configuration, (b) primary implementation across multiple files "
        "or languages, (c) iterative refinement against a quantitative or "
        "adversarial verifier, and (d) a final integration step. "
        "Expect 30-60 commands.)"
    ),
]

# Slices used by the bucket-upweight sampler when sampling for v2 corpora.
# The first 3 entries are the legacy values (used at uniform weight by the
# legacy corpus); the 4th is "intricate" and is the new-axis bucket.
_LEGACY_COMPLEXITIES: list[str] = TASK_COMPLEXITY[:3]
_INTRICATE_COMPLEXITY: str = TASK_COMPLEXITY[3]

COMMAND_COMPLEXITY: list[str] = [
    "bash-only (shell built-ins, coreutils, and standard CLI tools)",
    "bash and code (shell commands and writing/running scripts in Python, Perl, Ruby, etc.)",
    (
        "bash, code, and system services (shell commands, scripts, package installation, "
        "service configuration, networking, and containers)"
    ),
]


# ---------------------------------------------------------------------------
# v2 axes — Verifier kind and Fixture kind
# ---------------------------------------------------------------------------
# These axes are sampled in addition to the existing 7 (domain × skill_type ×
# primitive_skills × task_complexity × command_complexity × scenario × language)
# whenever `corpus_kind` is "sft_v2" or "rl_v2". For "legacy" (the default),
# the new axes are forced to their legacy default values so byte-identical
# behaviour is preserved.
#
# See scripts/analysis/tb2_gemini_tassieagent_eval.md §6–§7 for the
# motivation behind each kind.

VERIFIER_KINDS: list[str] = [
    "metric_threshold",       # numerical metric vs reference (similarity / speed / accuracy)
    "adversarial_corpus",     # evil/ corpus must reject + clean/ must preserve
    "fuzz_equivalence",       # bit-exact agreement with a reference oracle
    "multi_protocol",         # real protocol-level requests (HTTP/TCP/gRPC)
    "exact_text",             # legacy default — text equality
]
_VERIFIER_LEGACY: str = "exact_text"
_VERIFIER_NEW: list[str] = [v for v in VERIFIER_KINDS if v != _VERIFIER_LEGACY]

FIXTURE_KINDS: list[str] = [
    "image",                  # task ships a PNG; agent does OCR / vision
    "audio",                  # task ships a WAV; agent does ASR
    "video",                  # task ships an MP4; agent does event detection
    "stripped_binary",        # task ships a stripped binary; agent does RE / fuzz-equivalence
    "vendored_package",       # task ships a pre-vendored real package + a perturbation
    "multi_service_compose",  # task ships multiple cooperating services (compose-style)
    "text_only",              # legacy default — only text descriptions
]
_FIXTURE_LEGACY: str = "text_only"
_FIXTURE_NEW: list[str] = [f for f in FIXTURE_KINDS if f != _FIXTURE_LEGACY]


# ---------------------------------------------------------------------------
# Corpus kinds and their bucket-upweight multipliers
# ---------------------------------------------------------------------------

CORPUS_KINDS: tuple[str, ...] = ("legacy", "sft_v2", "rl_v2")

# Bucket-upweight multipliers per v2 corpus. M is the relative weight of the
# new-axis bucket vs the legacy bucket. Concretely:
#   P(any new) = M / (1 + M),  P(legacy) = 1 / (1 + M).
# Tuned to compensate for the legacy 1k/10k corpora having 0 % new-axis
# representation, so the combined corpus has more balanced new-vs-legacy
# axis coverage. See plan §1 for the resulting per-axis distributions.
_CORPUS_MULTIPLIER: dict[str, float] = {
    "sft_v2": 2.0,
    "rl_v2": 1.5,
}


def _bucket_upweight_choice(
    new_values: list[str],
    legacy_value: str,
    multiplier: float,
) -> str:
    """Sample one value such that the *new bucket* (combined) is ``multiplier``
    times more likely than the legacy default. New values are uniform within
    the new bucket.

    With weights ``[M/K, M/K, ..., M/K, 1]`` (K new + 1 legacy) the totals are
    ``M + 1``, giving ``P(any new) = M/(M+1)`` and ``P(legacy) = 1/(M+1)``.
    """
    if not new_values:
        return legacy_value
    weights = [multiplier / len(new_values)] * len(new_values) + [1.0]
    return random.choices(new_values + [legacy_value], weights=weights, k=1)[0]


def _bucket_upweight_complexity(multiplier: float) -> str:
    """Special-case bucket-upweight for ``TASK_COMPLEXITY``.

    Here the legacy bucket has 3 values and the new bucket has 1 (``intricate``).
    With weights ``[1/3, 1/3, 1/3, M]`` (3 legacy + 1 new) the totals are
    ``1 + M``, giving ``P(intricate) = M/(M+1)`` and ``P(any legacy) = 1/(M+1)``,
    each legacy value uniformly within the legacy bucket.
    """
    if not _LEGACY_COMPLEXITIES:
        return _INTRICATE_COMPLEXITY
    weights = [1.0 / len(_LEGACY_COMPLEXITIES)] * len(_LEGACY_COMPLEXITIES) + [multiplier]
    return random.choices(
        _LEGACY_COMPLEXITIES + [_INTRICATE_COMPLEXITY], weights=weights, k=1,
    )[0]


# ---------------------------------------------------------------------------
# Domain-specific personas (non-exclusive: a persona may appear in >1 domain)
# ---------------------------------------------------------------------------

DOMAIN_SCENARIOS: dict[str, list[str]] = {
    "security": [
        "security auditor checking permissions",
        "penetration tester scanning vulnerabilities",
        "security engineer rotating credentials",
        "DevSecOps engineer enforcing policy as code",
        "compliance analyst generating audit trails",
        "incident responder investigating issues",
        "network engineer inspecting traffic",
        "forensics analyst recovering evidence from a compromised host",
        "red-team operator crafting an evasion payload",
    ],
    "software_engineering": [
        "developer organizing project files",
        "build engineer managing artifacts",
        "release manager preparing deployments",
        "QA engineer setting up test environments",
        "platform engineer maintaining CI/CD pipelines",
        "integration developer testing APIs",
        "web developer building a feature",
        "script developer creating utilities",
        "mobile build engineer maintaining pipelines",
        "open-source maintainer reviewing a broken PR",
        "developer migrating from Python 2 to Python 3",
        "engineer porting a Linux tool to work in a minimal container",
        "developer fixing a multi-file Rust project that fails to compile",
        "systems programmer debugging a C library linking issue",
        "engineer setting up a polyglot build system from scratch",
    ],
    "file_operations": [
        "developer organizing project files",
        "backup administrator archiving data",
        "researcher organizing datasets",
        "storage administrator managing disk space",
        "technical writer organizing documentation",
        "configuration manager tracking changes",
        "artifact manager curating binary repositories",
    ],
    "data_querying": [
        "database administrator optimizing queries",
        "data analyst processing CSV files",
        "data engineer building ETL pipelines",
        "researcher organizing datasets",
        "database reliability engineer managing backups",
        "compliance officer auditing systems",
    ],
    "data_science": [
        "data scientist cleaning datasets",
        "machine learning engineer preparing training data",
        "data analyst processing CSV files",
        "researcher organizing datasets",
        "MLOps engineer tracking experiment artifacts",
        "data engineer building ETL pipelines",
    ],
    "debugging": [
        "DevOps engineer debugging logs",
        "support engineer collecting diagnostics",
        "site reliability engineer monitoring uptime",
        "developer debugging a failing build",
        "operations engineer triaging incidents",
        "performance engineer profiling applications",
        "IT support technician resolving tickets",
        "on-call engineer responding to a 3am page",
        "developer inheriting an unfamiliar codebase",
        "security researcher analysing a suspicious binary",
        "engineer investigating a memory leak in a long-running service",
        "developer bisecting a regression across 200 commits",
    ],
    "scientific_computing": [
        "researcher running simulations",
        "data scientist fitting models",
        "machine learning engineer preparing training data",
        "performance engineer profiling applications",
        "bioinformatics analyst processing sequences",
    ],
    "data_processing": [
        "data engineer building ETL pipelines",
        "data analyst processing CSV files",
        "log analyst investigating patterns",
        "automation specialist creating workflows",
        "localization engineer updating translations",
        "data scientist cleaning datasets",
        "configuration manager tracking changes",
    ],
    "system_administration": [
        "system administrator maintaining servers",
        "linux systems engineer hardening configurations",
        "kubernetes operator managing manifests",
        "cloud architect migrating services",
        "infrastructure engineer automating provisioning",
        "site reliability engineer monitoring uptime",
        "monitoring specialist setting up alerts",
        "deployment engineer rolling out updates",
        "capacity planner analyzing resource usage",
        "backup operator testing restores",
        "FinOps analyst optimizing cloud costs",
        "container specialist managing microservices",
        "network engineer troubleshooting connectivity",
        "edge computing engineer deploying to IoT devices",
        "observability engineer tuning dashboards",
        "site administrator managing user accounts",
        "engineer diagnosing why a systemd service fails to start",
        "admin fixing an nginx config that returns 502 bad gateway",
    ],
}


# ---------------------------------------------------------------------------
# Real-software anchors: concrete buggy-scenario templates sampled as optional
# inspiration for the LLM.  ~35% of tasks include one.
# ---------------------------------------------------------------------------

REAL_SOFTWARE_ANCHORS: dict[str, list[str]] = {
    "software_engineering": [
        "a small C project with a Makefile that has a linking error",
        "a Python package with a broken setup.py/pyproject.toml",
        "a multi-file Rust project that fails to compile due to lifetime issues",
        "a Go module with a circular import that prevents building",
        "a Node.js project whose npm install fails due to conflicting peer deps",
        "a CMake project that can't find a shared library at link time",
        "a Python project whose test suite passes locally but fails in CI due to import ordering",
    ],
    "debugging": [
        "a C program with a buffer overflow that only manifests with specific input",
        "a Python script with a subtle timezone bug",
        "a shell script that breaks on filenames with spaces",
        "a multithreaded program that deadlocks under high contention",
        "a Rust program that panics on unwrap() with certain edge-case data",
        "a Go service that leaks goroutines under cancellation",
        "a program that works on x86 but produces wrong results due to signed integer overflow",
    ],
    "system_administration": [
        "an nginx config that returns 502 due to wrong upstream socket path",
        "a systemd service that fails to start because of a missing After= dependency",
        "a cron job that runs but writes to the wrong location due to PATH differences",
        "an SSH config that silently rejects key-based login",
        "a Docker Compose setup where services can't reach each other due to network misconfiguration",
    ],
    "security": [
        "a web server with an open redirect vulnerability in its login flow",
        "a script that leaks credentials via command-line arguments visible in /proc",
        "a JWT implementation that accepts tokens with algorithm=none",
        "a file upload handler susceptible to path traversal",
    ],
    "data_querying": [
        "a SQL query that returns wrong results due to an implicit cross join",
        "an SQLite database with a corrupted index that returns stale rows",
        "a query that deadlocks two concurrent transactions",
    ],
    "data_processing": [
        "a CSV pipeline that silently drops rows containing embedded newlines",
        "a JSON-lines parser that breaks on unicode escape sequences",
        "an ETL job that produces duplicate records on retry",
    ],
    "file_operations": [
        "a backup script that follows symlinks into infinite loops",
        "an archive extraction that overwrites files outside the target directory (zip slip)",
        "a log rotation script that races with the writing process",
    ],
    "data_science": [
        "a pandas pipeline that silently converts ints to floats via NaN introduction",
        "a scikit-learn pipeline where data leaks between train and test via fit_transform",
        "a matplotlib script that produces blank plots due to backend misconfiguration",
    ],
    "scientific_computing": [
        "a numerical integrator that diverges due to wrong step-size adaptation",
        "a matrix factorisation that fails on near-singular input",
        "a simulation that produces non-reproducible results due to floating-point reduction order",
    ],
}

_ANCHOR_PROBABILITY = 0.35


# ---------------------------------------------------------------------------
# Language axis: sampled alongside domain and complexity to encourage
# non-Python tasks.  Weights are tuned to keep Python dominant but ensure
# meaningful coverage of compiled languages.
# ---------------------------------------------------------------------------

TASK_LANGUAGES: list[tuple[str, float]] = [
    ("Python",          0.35),
    ("C",               0.15),
    ("Bash",            0.15),
    ("C++",             0.10),
    ("Rust",            0.07),
    ("Go",              0.07),
    ("multi-language",  0.06),
    ("any (model's choice)", 0.05),
]

_LANG_NAMES = [l for l, _ in TASK_LANGUAGES]
_LANG_WEIGHTS = [w for _, w in TASK_LANGUAGES]


def _sample_language() -> str:
    return random.choices(_LANG_NAMES, weights=_LANG_WEIGHTS, k=1)[0]


# ---------------------------------------------------------------------------
# v2 system-prompt fragments — injected when verifier_kind / fixture_kind
# are not the legacy default. Each fragment is short and declarative, telling
# the LLM *what kind of task* to build and *what its <truth> must declare*
# so the downstream stages (apptainer_def_gen, fixture_gen, completion_test_gen)
# can materialise the artefact and the verifier reliably.
# ---------------------------------------------------------------------------

_VERIFIER_KIND_FRAGMENTS: dict[str, str] = {
    "metric_threshold": """\
## Verifier kind: metric_threshold
The agent's output will be graded by a NUMERICAL metric against a reference,
not by exact text equality. Examples: image SSIM, audio MSE, model accuracy
on a held-out test set, runtime speedup vs a reference implementation, output
file size, etc.
The <truth> block MUST declare:
  * The *exact* metric (formula, tool, or short Python snippet that computes it).
  * A reference / target value (a number) and a tolerance / threshold the agent
    must meet (e.g. "SSIM >= 0.95", "speedup >= 1.3x", "accuracy >= 0.62").
  * The exact path of the file or program the verifier should evaluate.""",
    "adversarial_corpus": """\
## Verifier kind: adversarial_corpus
The agent must produce a filter / sanitiser / detector / classifier that is
graded against TWO corpora shipped with the task: an "evil" corpus that the
solution MUST reject (or sanitise / flag), and a "clean" corpus that the
solution MUST preserve unchanged (or accept). Pass requires both directions.
The <truth> block MUST declare:
  * The exact paths of the evil/ and clean/ corpora the verifier will load.
  * The pass criterion ("100% of evil rejected AND 100% of clean preserved",
    or a specific pass-rate threshold per corpus).
  * The agent's expected entry point (function signature / CLI invocation).""",
    "fuzz_equivalence": """\
## Verifier kind: fuzz_equivalence
The agent must produce a program whose behaviour is BIT-EXACT equivalent to a
reference oracle that the verifier already has access to (e.g. a stripped
binary or a reference implementation). The verifier random-fuzzes both with
N inputs and asserts identical outputs.
The <truth> block MUST declare:
  * The path of the oracle program the verifier compares against.
  * The fuzz-input distribution (input length range, character set, etc.) and N.
  * The agent's expected entry point (executable path + invocation).""",
    "multi_protocol": """\
## Verifier kind: multi_protocol
The agent must bring up one or more network services; the verifier issues
real protocol-level requests (HTTP / TCP / gRPC / SMTP / SSH / etc.) and
checks the responses.
The <truth> block MUST declare:
  * The exact host:port the agent's service should listen on.
  * The protocol and the request/response patterns the verifier will exercise.
  * Any auth tokens / credentials the agent must accept.""",
}

_FIXTURE_KIND_FRAGMENTS: dict[str, str] = {
    "image": """\
## Fixture kind: image
This task ships a real image artefact (PNG/JPEG) at a specific path under
/app/. The agent typically needs OCR (tesseract is preinstalled) or basic
vision processing to recover information from it.
The <truth> block MUST declare:
  * The exact path of the image (under /app/).
  * The hidden ground truth (text / numbers / structure) the agent must recover.
  * (Implicit) the verifier checks the agent's recovered content against this
    ground truth, not the image itself.""",
    "audio": """\
## Fixture kind: audio
This task ships a real audio artefact (WAV/MP3) at a specific path under
/app/. The agent typically needs a transcription tool to recover the spoken
content (whisper.cpp / ffmpeg-based pipelines).
The <truth> block MUST declare:
  * The exact path of the audio file.
  * The hidden ground truth (transcript / event sequence / measurement).""",
    "video": """\
## Fixture kind: video
This task ships a real video artefact (MP4) at a specific path under /app/.
The agent typically needs frame extraction (ffmpeg is preinstalled) plus
per-frame analysis to recover events, counts, or detections.
The <truth> block MUST declare:
  * The exact path of the video file.
  * The hidden ground truth (frame ranges, counts, detections, etc.).""",
    "stripped_binary": """\
## Fixture kind: stripped_binary
This task ships a stripped, possibly UPX-packed binary at a specific path
under /app/. The agent typically reverse-engineers it (objdump, gdb, strings
are preinstalled) or treats it as a black-box oracle (often combined with
verifier_kind=fuzz_equivalence).
The <truth> block MUST declare:
  * The exact path of the binary.
  * The high-level algorithm the binary implements (so the verifier can
    construct test inputs / fuzz the oracle vs the agent's solution).""",
    "vendored_package": """\
## Fixture kind: vendored_package
This task ships a real third-party package source pre-vendored at a specific
path under /app/. NO INTERNET access is required at solve time. The package
has a deliberate perturbation (broken Makefile, wrong env var, missing patch,
etc.) that the agent must fix to get a known-good code path working.
The <truth> block MUST declare:
  * The package name + version + the exact path of its vendored source.
  * The perturbation that was applied (so the verifier can construct the
    expected post-fix state).
  * The known-good code path the verifier exercises after the agent's fix.""",
    "multi_service_compose": """\
## Fixture kind: multi_service_compose
This task involves multiple cooperating services (e.g. nginx + flask + redis,
postfix + mailman, qemu + telnet, etc.). At solve time, a startup script
brings up the services; the agent must reconfigure / glue them so a specific
end-to-end protocol flow works.
The <truth> block MUST declare:
  * The list of services + their ports / sockets / process roles.
  * The exact end-to-end flow the verifier will exercise.
  * The configuration files / env vars the agent must adjust.""",
}


def _build_v2_axis_block(verifier_kind: str, fixture_kind: str) -> str:
    """Return the conditional v2 instruction block, or empty string if both
    axes are at their legacy defaults.
    """
    parts: list[str] = []
    v_frag = _VERIFIER_KIND_FRAGMENTS.get(verifier_kind)
    f_frag = _FIXTURE_KIND_FRAGMENTS.get(fixture_kind)
    if v_frag is not None:
        parts.append(v_frag)
    if f_frag is not None:
        parts.append(f_frag)
    if not parts:
        return ""
    return "\n\n# v2 Axes (additional task structure)\n\n" + "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    domain: str,
    *,
    verifier_kind: str = _VERIFIER_LEGACY,
    fixture_kind: str = _FIXTURE_LEGACY,
) -> str:
    """Assemble a domain-specific system prompt.

    When ``verifier_kind`` or ``fixture_kind`` is non-legacy, an extra v2-axis
    block is appended that tells the LLM what kind of task to build and what
    its ``<truth>`` block must declare. For legacy values both fragments are
    omitted, so output is byte-identical to the pre-v2 behaviour.
    """
    domain_label = domain.replace("_", " ").title()
    module = DOMAIN_MODULES[domain]
    v2_block = _build_v2_axis_block(verifier_kind, fixture_kind)

    return f"""\
You are an expert at creating {domain_label} tasks for AI agent training.

{module}{v2_block}

Universal Task Requirements:
- Challenging to solve: Requires domain knowledge, analytical thinking, and \
efficient implementation.
- Easy to verify: Success must be determinable by programmatically checking \
outputs, exit codes, or system state.
- Self-contained: All necessary information must be in the prompt.
- Realistic: The problem should resemble tasks professionals face in this domain.

Respond in XML format using these tags:

<task>
        A detailed task description written as a user would ask an AI assistant.
        Give the names of the precise contents of files, ports, directories, etc.
        This should be a very detailed description of the final state of the system.
        For example, if you are asking the agent to create a log file, you should
        precisely specify the format it should be in so that an automated test can
        verify it.
        Ask the agent to create a log file whenever some verification is required.
        You only have about 1000-1500 words to work with. So balance between
        conciseness and detail.
        DO NOT directly give the commands to the agent.
</task>

<truth>
        Insert *privileged* ground-truth data that automated test suites will
        rely on to verify correct task execution.
        These values **must NOT** appear in the public task description.

        Be very detailed here. Give the names / placeholders of the precise
        contents of files, ports, directories, repositories, websites etc.
        Any processes, files, directories that should be created before the task
        starts should be mentioned here.
        Any files that should be created by the agent and their contents should
        be mentioned here.

        Ground-truth principles (for accurate automated verification):
        * **Consistency:** Anything in *truth* that could be computed from the setup
          code, random seeds, or the task rules must actually follow from them. Do not
          assert derived numbers, digests, or file bodies unless they are implied by
          what you specified—avoid plausible-looking literals produced without that chain.
        * **Reproducibility:** Prefer stating *how* to obtain a golden value (procedure,
          formula, or a short **runnable** snippet that prints the canonical result) over
          pasting opaque constants that nothing in the pipeline verifies.
        * **Causal ordering:** When setup involves multiple steps (random draws, mutations,
          I/O), make the sequence explicit. Headline summaries (e.g. simple counts) must
          reflect the real order of operations, not an informal intuition.
        * **Single source of truth:** Setup scripts, narrative expectations, and any
          “expected output” blocks must agree; resolve contradictions before finishing.
</truth>

Critical Rules:
* No Leakage: Never include code that solves the task in the <task> description.
* Verification: Prioritize tasks with clear, programmatic verification.
* Originality: Tasks should require thought, not just copying standard tutorials.
* Complete Specification: Include all information needed to complete the task \
(file paths, formats, constraints).
* Place any secret, ground-truth verification data exclusively under <truth>.
* The agent will not have root access. Make sure the right permissions are set \
for files and directories.
* When you mention a file or directory, write the full path (not relative).
* We will be using apptainer to run the agent. Make sure the task is valid when \
the container is built.
* Don't create tasks that require having the latest information.
* The home path is /home/user.
* Don't create tasks the setup of which will require su access.
* The task is multi-turn, so the agent will interact in a terminal to finish \
the task.
* Don't discourage the agent from using console output to finish the task.
* Do not constrain the number of commands the agent may use."""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _v2_axes_user_block(verifier_kind: str, fixture_kind: str) -> str:
    """Short user-message reminder of the v2 axes when they are non-legacy.

    Re-states the verifier_kind / fixture_kind as concrete bullets so the LLM
    cannot ignore them when sampling produced an exotic combination.
    """
    bullets: list[str] = []
    if verifier_kind != _VERIFIER_LEGACY:
        bullets.append(f"- Verifier kind: **{verifier_kind}** (see system prompt for what <truth> must declare).")
    if fixture_kind != _FIXTURE_LEGACY:
        bullets.append(f"- Fixture kind: **{fixture_kind}** (the task ships this artefact under /app/).")
    if not bullets:
        return ""
    return "\n## v2 Axes\n" + "\n".join(bullets) + "\n"


def random_user_msg(corpus_kind: str = "legacy") -> tuple[str, str, dict]:
    """Generate a domain-specific (system_msg, user_msg, metadata) tuple.

    Selects a domain, samples primitive skills, picks complexity levels,
    draws a domain-appropriate persona, optionally injects a real-software
    anchor, and samples a primary language.

    Parameters
    ----------
    corpus_kind:
        ``"legacy"`` (default) reproduces the pre-v2 behaviour byte-for-byte —
        no verifier_kind / fixture_kind sampling, only the original 3
        ``TASK_COMPLEXITY`` values.

        ``"sft_v2"`` and ``"rl_v2"`` enable the v2 axes:
          * ``verifier_kind`` and ``fixture_kind`` are sampled with the
            bucket-upweight formula (``M=2`` for sft_v2, ``M=1.5`` for rl_v2),
            making the new-axis bucket combined ``M`` × more likely than the
            legacy default. New values are uniform within the new bucket.
          * ``task_complexity`` is sampled with the same upweight, where the
            "new bucket" is the single ``intricate`` value and the legacy
            bucket is the original 3 (``short`` / ``moderate`` / ``complex``).
    """
    if corpus_kind not in CORPUS_KINDS:
        raise ValueError(
            f"corpus_kind must be one of {CORPUS_KINDS}, got {corpus_kind!r}"
        )

    domain = random.choice(list(SKILL_TAXONOMY.keys()))
    skill_types = SKILL_TAXONOMY[domain]

    skill_type = random.choice(list(skill_types.keys()))

    all_skills: list[str] = []
    for skills in skill_types.values():
        all_skills.extend(skills)
    num_skills = random.randint(3, 5)
    primitive_skills = random.sample(all_skills, min(num_skills, len(all_skills)))

    if corpus_kind == "legacy":
        task_complexity = random.choice(_LEGACY_COMPLEXITIES)
        verifier_kind = _VERIFIER_LEGACY
        fixture_kind = _FIXTURE_LEGACY
    else:
        m = _CORPUS_MULTIPLIER[corpus_kind]
        task_complexity = _bucket_upweight_complexity(m)
        verifier_kind = _bucket_upweight_choice(_VERIFIER_NEW, _VERIFIER_LEGACY, m)
        fixture_kind = _bucket_upweight_choice(_FIXTURE_NEW, _FIXTURE_LEGACY, m)

    command_complexity = random.choice(COMMAND_COMPLEXITY)
    scenario = random.choice(DOMAIN_SCENARIOS[domain])
    language = _sample_language()

    anchor: str | None = None
    domain_anchors = REAL_SOFTWARE_ANCHORS.get(domain)
    if domain_anchors and random.random() < _ANCHOR_PROBABILITY:
        anchor = random.choice(domain_anchors)

    system_msg = build_system_prompt(
        domain, verifier_kind=verifier_kind, fixture_kind=fixture_kind,
    )

    skills_formatted = "\n".join(f"- {s}" for s in primitive_skills)

    anchor_block = ""
    if anchor:
        anchor_block = (
            f"\n## Scenario Anchor (use as inspiration, not literally)\n"
            f"{anchor}\n"
        )

    v2_block = _v2_axes_user_block(verifier_kind, fixture_kind)

    user_msg = (
        f"# Task Generation Request\n"
        f"Category: {skill_type}\n"
        f"\n"
        f"## Primary Language\n"
        f"{language}\n"
        f"\n"
        f"## Primitive Skills (Building Blocks)\n"
        f"{skills_formatted}\n"
        f"\n"
        f"## Task Complexity\n"
        f"{task_complexity}\n"
        f"\n"
        f"## Command Complexity\n"
        f"{command_complexity}\n"
        f"\n"
        f"## Scenario\n"
        f"{scenario}\n"
        f"{anchor_block}"
        f"{v2_block}"
        f"\n"
        f"## Instructions\n"
        f"CREATE A NOVEL TASK that:\n"
        f"1. Combines 3-5 of the primitive skills above in a creative, unexpected way\n"
        f"2. Is NOT a recreation of common coding challenges\n"
        f"3. Is challenging to solve but easy to verify\n"
        f"4. Has clear, unambiguous specifications\n"
        f"5. Is a realistic end-to-end scenario that an AI agent could perform "
        f"in a Linux terminal\n"
        f"6. Uses **{language}** as the primary language for any code that must be "
        f"written or debugged (shell commands are always allowed alongside it)\n"
        f"\n"
        f"Think of an original scenario or application -- don't just combine "
        f"primitives mechanically.\n"
        f"Be very specific about the output format in the task description that "
        f"the automated test will check.\n"
        f"Write the task description in a way that a user might ask an AI assistant."
    )

    # Tasks with any non-legacy v2 axis route to the shared "intricate" base
    # SIF (numpy/scipy/Pillow/torch-cpu/tesseract/ffmpeg/UPX/binutils
    # pre-installed). Legacy tasks keep using their per-domain base.
    is_v2_task = (
        verifier_kind != _VERIFIER_LEGACY
        or fixture_kind != _FIXTURE_LEGACY
        or task_complexity == _INTRICATE_COMPLEXITY
    )
    base_image = "intricate" if is_v2_task else None

    metadata = {
        "domain": domain,
        "skill_type": skill_type,
        "primitive_skills": primitive_skills,
        "task_complexity": task_complexity,
        "command_complexity": command_complexity,
        "scenario": scenario,
        "language": language,
        "anchor": anchor,
        # v2 axes — present on every task; legacy tasks have the legacy default.
        "verifier_kind": verifier_kind,
        "fixture_kind": fixture_kind,
        "corpus_kind": corpus_kind,
        # Routing hint consumed by env._resolve_runtime_sif at solve time.
        # `None` (default) means "use base_<domain>.sif".
        "base_image": base_image,
    }
    return system_msg, user_msg, metadata


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_templates_batch(
    batch_size: int,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    max_concurrency: int = 128,
    corpus_kind: str = "legacy",
) -> list[dict]:
    """Generate multiple task templates in one batched LLM call set.

    Returns a list of dicts with keys ``description``, ``truth``,
    ``domain``, ``skill_type``, ``primitive_skills``,
    ``task_complexity``, ``command_complexity``, ``scenario``, ``language``,
    ``anchor``, and (always present) ``verifier_kind``, ``fixture_kind``,
    ``corpus_kind``, ``base_image``.

    Pass ``corpus_kind="sft_v2"`` or ``"rl_v2"`` to enable the v2 axes;
    ``"legacy"`` (default) preserves byte-identical pre-v2 behaviour.
    """
    messages: list[list[dict[str, str]]] = []
    metadata_list: list[dict] = []
    for _ in range(batch_size):
        system_msg, user_msg, metadata = random_user_msg(corpus_kind=corpus_kind)
        messages.append([
            {"role": "system", "content": system_msg},
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
    template = re.search(r"<task>(.*?)</task>", raw, re.DOTALL).group(1).strip()
    if not template:
        raise ValueError("No task description found in the response.")

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
    for task in tasks:
        task_name = str(uuid.uuid4())
        task_path = Path("tasks") / task_name
        task_path.mkdir(parents=True, exist_ok=True)
        with open(task_path / "task.json", "w") as f:
            json.dump(task, f, indent=4)
