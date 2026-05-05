"""Host-side, deterministic fixture materialisation for v2 tasks.

For every v2 task whose ``fixture_kind`` is non-legacy, this module produces
the actual artefact bytes (image / audio / video / stripped binary / vendored
package / multi-service compose stack) on the **host** before the per-task
Apptainer image is built. The artefacts then get baked into the SIF via a
``%files`` section emitted by ``apptainer_def_gen.py``.

Why host-side and deterministic
-------------------------------
The TB 2.0 eval analysis showed that "find this URL on the internet" tasks
(``build-pov-ray``, ``mteb-leaderboard``, etc.) are an anti-pattern: agents
exhaust their budget guessing URLs and the RL signal is sparse / wrong. By
materialising every fixture **before** the SIF is built and shipping the
bytes inside the SIF, no internet access is required at solve time.

Determinism is provided by the ``seed`` argument; passing the same seed +
the same task_description / truth produces byte-identical fixtures, which
makes the corpora reproducible.

Scope of this iteration
-----------------------
Each fixture kind has a *minimal-viable* generator: simple, dependency-light,
deterministic. The intent is to unblock the v2 corpus generation so we can
study how Gemini-3-pro-preview composes a task description around a fixture.
Richer / more diverse generators (e.g. domain-specific image renderers,
real package vendoring with perturbations) are TODO and left as follow-ups.

Public API
----------
``materialize(fixture_kind, task_description, truth, dest_dir, seed)`` →
returns a list of ``(host_path, container_path)`` tuples to be copied into
the SIF via ``%files``. Returns ``[]`` for ``fixture_kind="text_only"`` (the
legacy default) so legacy tasks keep producing identical output.

Each generator writes its outputs under ``dest_dir / "fixtures" /``. The
canonical container destination is ``/app/fixtures/``.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import string
import struct
import subprocess
import textwrap
import wave
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Container-side root where every fixture lands. The system-prompt v2
#: fragments tell the LLM "the task ships its artefact under /app/", so this
#: is the canonical directory.
CONTAINER_FIXTURES_ROOT = "/app/fixtures"

#: List of supported fixture kinds. Mirrors task_template_gen.FIXTURE_KINDS
#: but kept independent so this module can be smoke-tested in isolation.
SUPPORTED_FIXTURE_KINDS = (
    "image", "audio", "video", "stripped_binary",
    "vendored_package", "multi_service_compose",
)

#: ``fixture_kind`` values that materialise nothing (legacy / no-op).
NOOP_FIXTURE_KINDS = ("text_only",)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _stable_rng(seed: int, salt: str) -> random.Random:
    """Return a deterministic RNG seeded by ``(seed, salt)``."""
    h = abs(hash((seed, salt))) % (2**32)
    return random.Random(h)


def _ensure_dest(dest_dir: Path) -> Path:
    """Create + return ``<dest_dir>/fixtures/``."""
    out = Path(dest_dir) / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _container_path(host_path: Path, host_root: Path) -> str:
    """Map a host-side fixture path to its location inside the container."""
    rel = host_path.resolve().relative_to(host_root.resolve())
    return f"{CONTAINER_FIXTURES_ROOT}/{rel}"


def _which(binary: str) -> bool:
    """Is ``binary`` available on PATH? (pure stdlib, no shutil.which call
    overhead per fixture)."""
    return shutil.which(binary) is not None


# ---------------------------------------------------------------------------
# Image fixture
# ---------------------------------------------------------------------------


def _materialize_image(
    *, dest: Path, rng: random.Random, task_description: str, truth: str
) -> List[Path]:
    """Render a small PNG containing a deterministic colour grid + a hidden
    text label. The label is the ground-truth string the agent will need to
    recover via OCR or vision tooling. We try Pillow first; fall back to a
    minimal raw PPM if Pillow isn't available on the build host (the SIF
    *will* have Pillow, but the fixture-gen runs on the host).
    """
    label = _hidden_label_from_truth(truth, rng)
    out_paths: List[Path] = []

    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except Exception:
        # Fallback: emit a minimal PPM with a deterministic colour pattern.
        # No text rendering — the label is dropped to a sidecar file.
        ppm_path = dest / "image.ppm"
        _write_ppm_grid(ppm_path, rng=rng, w=160, h=90)
        out_paths.append(ppm_path)
        (dest / "image.label.txt").write_text(label)
        out_paths.append(dest / "image.label.txt")
        return out_paths

    img = Image.new("RGB", (320, 180), color=(28, 28, 36))
    draw = ImageDraw.Draw(img)

    # Background pattern — random rectangles, deterministic per seed.
    for _ in range(40):
        x0 = rng.randint(0, 300)
        y0 = rng.randint(0, 160)
        x1 = x0 + rng.randint(8, 32)
        y1 = y0 + rng.randint(8, 24)
        col = (rng.randint(40, 220), rng.randint(40, 220), rng.randint(40, 220))
        draw.rectangle((x0, y0, x1, y1), fill=col)

    # Foreground label — the hidden ground-truth string.
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 80), label, fill=(255, 255, 255), font=font)

    png_path = dest / "image.png"
    img.save(png_path, format="PNG", optimize=False)
    out_paths.append(png_path)
    return out_paths


def _write_ppm_grid(path: Path, *, rng: random.Random, w: int, h: int) -> None:
    """Tiny PPM (binary P6) writer — pure stdlib fallback when Pillow is
    unavailable on the build host.
    """
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    pixels = bytearray()
    for y in range(h):
        for x in range(w):
            r = (x * 3 + rng.randint(0, 31)) & 0xFF
            g = (y * 5 + rng.randint(0, 31)) & 0xFF
            b = ((x ^ y) + rng.randint(0, 31)) & 0xFF
            pixels.extend((r, g, b))
    path.write_bytes(header + bytes(pixels))


def _hidden_label_from_truth(truth: str, rng: random.Random) -> str:
    """Pick a short, OCR-friendly string drawn from the ``truth`` text. Falls
    back to a random alnum if truth has nothing usable.
    """
    candidates = [w for w in truth.split() if w.isalnum() and 4 <= len(w) <= 12]
    if candidates:
        return rng.choice(candidates).upper()
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))


# ---------------------------------------------------------------------------
# Audio fixture
# ---------------------------------------------------------------------------


def _materialize_audio(
    *, dest: Path, rng: random.Random, task_description: str, truth: str
) -> List[Path]:
    """Synthesise a 1-second 16-bit PCM WAV at 8 kHz. Frequency and amplitude
    envelope are seeded; the *intended* hidden ground truth (a numeric pattern
    or short utterance) is dropped to a sidecar so the LLM can declare it
    inside ``<truth>``. We don't run espeak/edge-tts here to keep the build-
    host dependency surface minimal; the SIF carries ffmpeg for the agent to
    extract features.
    """
    import math  # noqa: PLC0415  (stdlib-only)

    out_paths: List[Path] = []
    n_samples = 8000  # 1 second at 8 kHz
    framerate = 8000

    # Seeded sequence of dual-tone components to make the signal interesting
    # and deterministic.
    base_freqs = [220, 330, 440, 660][rng.randint(0, 3) :]
    rng.shuffle(base_freqs)

    pcm = bytearray()
    for n in range(n_samples):
        t = n / framerate
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 2.0 * t)  # slow tremolo
        wave_value = sum(
            math.sin(2 * math.pi * f * t) for f in base_freqs[:2]
        ) / 2.0
        sample = int(env * wave_value * 28000)
        pcm.extend(struct.pack("<h", max(-32768, min(32767, sample))))

    wav_path = dest / "audio.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(bytes(pcm))
    out_paths.append(wav_path)

    # Sidecar label: the dominant frequencies (the LLM's <truth> can declare
    # "the agent must report the dominant frequencies sorted ascending").
    label = ",".join(str(f) for f in sorted(base_freqs[:2]))
    (dest / "audio.label.txt").write_text(label)
    out_paths.append(dest / "audio.label.txt")
    return out_paths


# ---------------------------------------------------------------------------
# Video fixture
# ---------------------------------------------------------------------------


def _materialize_video(
    *, dest: Path, rng: random.Random, task_description: str, truth: str
) -> List[Path]:
    """Generate a tiny MP4 with a moving event at a random frame. Requires
    ffmpeg on the build host; falls back to dropping a sentinel + sidecar
    when ffmpeg is unavailable so generation doesn't crash entire batches.
    """
    out_paths: List[Path] = []

    if not _which("ffmpeg"):
        sentinel = dest / "video.unavailable.txt"
        sentinel.write_text(
            "ffmpeg not available on build host; video fixture skipped.\n"
            "Re-generate this task on a host with ffmpeg installed.\n"
        )
        out_paths.append(sentinel)
        return out_paths

    fps = 8
    duration_s = 4
    event_frame = rng.randint(int(fps * 0.3), int(fps * duration_s) - 4)

    # Render a sequence of PPM frames in a temp subdir, then ffmpeg-encode.
    frames_dir = dest / "_video_frames"
    frames_dir.mkdir(exist_ok=True)
    try:
        for fi in range(fps * duration_s):
            frame_path = frames_dir / f"frame_{fi:04d}.ppm"
            _render_event_frame(frame_path, fi, event_frame, rng)
        mp4_path = dest / "video.mp4"
        proc = subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%04d.ppm"),
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-crf", "30",
                str(mp4_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            sentinel = dest / "video.unavailable.txt"
            sentinel.write_text(
                f"ffmpeg failed (rc={proc.returncode}): {proc.stderr[-500:]}\n"
            )
            out_paths.append(sentinel)
            return out_paths
        out_paths.append(mp4_path)
        # Sidecar label: the event frame index.
        (dest / "video.label.txt").write_text(str(event_frame))
        out_paths.append(dest / "video.label.txt")
    finally:
        # Always clean up the per-frame PPMs to keep the task dir small.
        shutil.rmtree(frames_dir, ignore_errors=True)
    return out_paths


def _render_event_frame(
    path: Path, frame_idx: int, event_frame: int, rng: random.Random
) -> None:
    """320x180 PPM with a square that flashes red on the event frame."""
    w, h = 320, 180
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    is_event = frame_idx == event_frame
    sx = 50 + (frame_idx * 8) % (w - 80)
    sy = 60
    sq = 30
    pixels = bytearray(w * h * 3)
    for y in range(h):
        for x in range(w):
            i = (y * w + x) * 3
            in_sq = sx <= x < sx + sq and sy <= y < sy + sq
            if in_sq:
                if is_event:
                    pixels[i] = 240
                    pixels[i + 1] = 30
                    pixels[i + 2] = 30
                else:
                    pixels[i] = 30
                    pixels[i + 1] = 200
                    pixels[i + 2] = 50
            else:
                pixels[i] = 20
                pixels[i + 1] = 20
                pixels[i + 2] = 30
    path.write_bytes(header + bytes(pixels))


# ---------------------------------------------------------------------------
# Stripped-binary fixture
# ---------------------------------------------------------------------------

# A tiny family of "secret algorithms" implemented in C. The generator
# samples one and bakes it into a stripped binary. Each algorithm reads a
# single line from stdin and writes one line to stdout.
_SECRET_ALGORITHMS: dict[str, str] = {
    "xor_const": r"""
#include <stdio.h>
#include <string.h>
int main(void) {
    char buf[256];
    if (!fgets(buf, sizeof(buf), stdin)) return 0;
    size_t n = strlen(buf);
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = 0;
    for (size_t i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)buf[i];
        printf("%02x", c ^ %CONST%);
    }
    putchar('\n');
    return 0;
}
""",
    "rolling_sum": r"""
#include <stdio.h>
#include <string.h>
int main(void) {
    char buf[256];
    if (!fgets(buf, sizeof(buf), stdin)) return 0;
    size_t n = strlen(buf);
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = 0;
    long acc = 0;
    for (size_t i = 0; i < n; ++i) {
        acc = (acc * %MULT% + (long)(unsigned char)buf[i]) %% %MOD%;
    }
    printf("%ld\n", acc);
    return 0;
}
""",
}


def _materialize_stripped_binary(
    *, dest: Path, rng: random.Random, task_description: str, truth: str
) -> List[Path]:
    """Compile a tiny C program implementing a seeded "secret algorithm",
    strip symbols, optionally pack with UPX, then drop the binary. Source is
    NOT shipped — the agent must reverse-engineer or fuzz against this oracle.

    Falls back to dropping a sentinel if no host compiler is available.
    """
    out_paths: List[Path] = []
    if not _which("gcc"):
        sentinel = dest / "binary.unavailable.txt"
        sentinel.write_text(
            "gcc not available on build host; stripped_binary fixture skipped.\n"
        )
        out_paths.append(sentinel)
        return out_paths

    algo_name, src_template = rng.choice(list(_SECRET_ALGORITHMS.items()))
    src = _instantiate_algo(algo_name, src_template, rng)
    src_path = dest / "_oracle.c"
    src_path.write_text(src)

    out_path = dest / "oracle"
    # Try a -static build first (more "stripped binary" feel), then fall back
    # to a dynamically-linked build on hosts that lack glibc-static. Keep the
    # source on disk across both attempts; only unlink after the final result
    # is known.
    proc = subprocess.run(
        ["gcc", "-O2", "-s", "-static", "-o", str(out_path), str(src_path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        proc = subprocess.run(
            ["gcc", "-O2", "-s", "-o", str(out_path), str(src_path)],
            capture_output=True, text=True, timeout=60,
        )
    src_path.unlink(missing_ok=True)
    if proc.returncode != 0 or not out_path.exists():
        sentinel = dest / "binary.unavailable.txt"
        sentinel.write_text(
            f"gcc compile failed (rc={proc.returncode}): {proc.stderr[-500:]}\n"
        )
        out_paths.append(sentinel)
        return out_paths

    # Pack with UPX if available (smaller, more obviously "stripped").
    if _which("upx"):
        subprocess.run(
            ["upx", "-q", "--best", str(out_path)],
            capture_output=True, text=True, timeout=30,
        )

    out_paths.append(out_path)

    # Sidecar: the algorithm name (the LLM's <truth> can declare the high-
    # level algorithm so the verifier knows what fuzz inputs to construct).
    (dest / "binary.algo.txt").write_text(algo_name)
    out_paths.append(dest / "binary.algo.txt")
    return out_paths


def _instantiate_algo(name: str, src: str, rng: random.Random) -> str:
    """Substitute named placeholders (``%CONST%``, ``%MULT%``, ``%MOD%``)
    with seeded constants so each task ships a slightly different oracle.
    """
    if name == "xor_const":
        const = rng.randint(0x10, 0xFE)
        return src.replace("%CONST%", f"0x{const:02x}")
    if name == "rolling_sum":
        mult = rng.choice([31, 37, 53, 131])
        mod = rng.choice([1_000_003, 1_000_033, 998_244_353])
        return src.replace("%MULT%", str(mult)).replace("%MOD%", str(mod))
    return src


# ---------------------------------------------------------------------------
# Vendored-package fixture
# ---------------------------------------------------------------------------

# Curated minimal "fake packages" that ship their own source tarball and a
# deliberate perturbation. Real package vendoring (pyknotid / pmars / etc.)
# is a TODO follow-up — the goal here is to exercise the pipeline end-to-end
# with a known-shaped artefact.
_FAKE_PACKAGES: list[dict] = [
    {
        "name": "minicalc",
        "files": {
            "Makefile": (
                "all: minicalc\n"
                "minicalc: minicalc.c\n"
                "\tgcc -O2 -o minicalc minicalc.c\n"
                "test: minicalc\n"
                "\t./minicalc 2 3 | grep -q 5\n"
                "clean:\n"
                "\trm -f minicalc\n"
            ),
            "minicalc.c": (
                "#include <stdio.h>\n#include <stdlib.h>\n"
                "int main(int argc, char **argv) {\n"
                "    if (argc != 3) return 1;\n"
                "    long a = atol(argv[1]), b = atol(argv[2]);\n"
                "    printf(\"%ld\\n\", a + b);\n"
                "    return 0;\n"
                "}\n"
            ),
            "README.md": "minicalc — a tiny addition CLI for v2-corpus tests.\n",
        },
        "perturbation": "broken_makefile_target",
    },
]


def _materialize_vendored_package(
    *, dest: Path, rng: random.Random, task_description: str, truth: str
) -> List[Path]:
    """Drop a small fake-package source tree under ``fixtures/<pkg_name>/``
    with a deliberate perturbation applied. Real third-party package
    vendoring is left as a TODO follow-up.
    """
    pkg = rng.choice(_FAKE_PACKAGES)
    pkg_dir = dest / pkg["name"]
    pkg_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    files = dict(pkg["files"])
    perturbation = pkg["perturbation"]
    if perturbation == "broken_makefile_target":
        # Rename the `all:` target so `make` (the obvious incantation) fails
        # with "no rule to make target". Agent must read the Makefile and
        # invoke the right target, OR fix the Makefile.
        files["Makefile"] = files["Makefile"].replace("all:", "alllll:", 1)

    for rel, content in files.items():
        p = pkg_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        out_paths.append(p)

    # Sidecar metadata for the LLM's <truth> to reference.
    meta = {
        "package_name": pkg["name"],
        "perturbation": perturbation,
        "expected_invocation": "./minicalc 2 3",
        "expected_output": "5",
    }
    (pkg_dir / "_perturbation.json").write_text(json.dumps(meta, indent=2))
    out_paths.append(pkg_dir / "_perturbation.json")
    return out_paths


# ---------------------------------------------------------------------------
# Multi-service compose fixture
# ---------------------------------------------------------------------------


def _materialize_multi_service_compose(
    *, dest: Path, rng: random.Random, task_description: str, truth: str
) -> List[Path]:
    """Drop a tiny multi-process startup script + a perturbation file. Real
    docker-compose-based fixtures are a TODO — for now we ship a foreman-
    style shell script that pre-runs nginx and a flask-style python server,
    with one service deliberately misconfigured.

    The agent's job is to fix the misconfiguration so the end-to-end flow
    works. The verifier (multi_protocol kind) issues the protocol requests.
    """
    out_paths: List[Path] = []
    out_dir = dest / "compose"
    out_dir.mkdir(exist_ok=True)

    # Misconfiguration: the upstream nginx port the python server listens on
    # is 5050, but the nginx config points at 5051. The agent must fix one
    # side or the other.
    correct_port = 5050
    misconfigured_port = 5051
    nginx_conf = textwrap.dedent(
        f"""\
        # NOTE: deliberate misconfiguration — upstream points at the wrong port.
        events {{}}
        http {{
            server {{
                listen 8080;
                location / {{ proxy_pass http://127.0.0.1:{misconfigured_port}; }}
            }}
        }}
        """
    )
    server_py = textwrap.dedent(
        f"""\
        import http.server, socketserver
        PORT = {correct_port}
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers()
                self.wfile.write(b'hello-from-app\\n')
        socketserver.TCPServer(("127.0.0.1", PORT), H).serve_forever()
        """
    )
    startup_sh = textwrap.dedent(
        """\
        #!/bin/sh
        set -e
        nginx -c "$(pwd)/nginx.conf" -g 'daemon off;' &
        python3 server.py &
        wait
        """
    )

    (out_dir / "nginx.conf").write_text(nginx_conf)
    (out_dir / "server.py").write_text(server_py)
    (out_dir / "start.sh").write_text(startup_sh)
    os.chmod(out_dir / "start.sh", 0o755)
    out_paths.extend([out_dir / "nginx.conf", out_dir / "server.py", out_dir / "start.sh"])

    meta = {
        "services": ["nginx", "python_http_server"],
        "external_port": 8080,
        "internal_app_port": correct_port,
        "misconfigured_upstream_port": misconfigured_port,
        "expected_response": "hello-from-app",
    }
    (out_dir / "_perturbation.json").write_text(json.dumps(meta, indent=2))
    out_paths.append(out_dir / "_perturbation.json")
    return out_paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_DISPATCH = {
    "image": _materialize_image,
    "audio": _materialize_audio,
    "video": _materialize_video,
    "stripped_binary": _materialize_stripped_binary,
    "vendored_package": _materialize_vendored_package,
    "multi_service_compose": _materialize_multi_service_compose,
}


def materialize(
    fixture_kind: str,
    *,
    task_description: str,
    truth: str,
    dest_dir: Path,
    seed: int,
) -> List[Tuple[Path, str]]:
    """Materialise the fixture artefact(s) for one task.

    Parameters
    ----------
    fixture_kind:
        One of ``FIXTURE_KINDS`` (see ``task_template_gen``). The legacy
        default ``"text_only"`` short-circuits to ``[]`` (no artefact).
    task_description / truth:
        Forwarded to the per-kind generators so they can pick parameters
        consistent with what the LLM described and what the ``<truth>`` block
        promises.
    dest_dir:
        Per-task host directory (typically the ``<task_dir>`` from
        ``generate_tasks._format_task_dir``). Fixtures land under
        ``dest_dir / "fixtures"``.
    seed:
        Deterministic seed; same ``(seed, task_description, truth)`` ⇒
        identical bytes.

    Returns
    -------
    list of ``(host_path, container_path)`` tuples. Empty for legacy /
    unknown kinds. The caller (``apptainer_def_gen``) is responsible for
    emitting the matching ``%files`` lines into the per-task ``.def``.
    """
    if fixture_kind in NOOP_FIXTURE_KINDS or fixture_kind not in _DISPATCH:
        return []

    rng = _stable_rng(seed, fixture_kind)
    fixtures_root = _ensure_dest(dest_dir)
    paths = _DISPATCH[fixture_kind](
        dest=fixtures_root, rng=rng,
        task_description=task_description, truth=truth,
    )
    return [(p, _container_path(p, fixtures_root)) for p in paths if p.exists()]


def emit_files_section(
    fixture_pairs: List[Tuple[Path, str]],
) -> str:
    """Render the ``%files`` section block for an Apptainer ``.def`` file.

    Returns an empty string when ``fixture_pairs`` is empty so legacy tasks
    produce byte-identical defs.
    """
    if not fixture_pairs:
        return ""
    lines = ["%files"]
    for host, container in fixture_pairs:
        lines.append(f"    {host.resolve()} {container}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI / smoke-test entry point
# ---------------------------------------------------------------------------


def _smoke(argv: List[str] | None = None) -> int:
    """Generate one fixture per kind into a temp dir for visual inspection.

    Run with ``uv run python -m rl_data.generator.fixture_gen [--out DIR]``.
    """
    import argparse  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out_dir = args.out or Path(tempfile.mkdtemp(prefix="fixture_smoke_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in SUPPORTED_FIXTURE_KINDS:
        sub = out_dir / kind
        sub.mkdir(parents=True, exist_ok=True)
        pairs = materialize(
            kind,
            task_description=f"smoke test for {kind}",
            truth=f"the agent must recover the hidden {kind} ground truth",
            dest_dir=sub,
            seed=args.seed,
        )
        print(f"{kind}: {len(pairs)} files")
        for host, container in pairs:
            size = host.stat().st_size if host.exists() else 0
            print(f"  {host} ({size} bytes) -> {container}")

    print(f"\nFixtures written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
