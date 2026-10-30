"""Build-stage dependency manifest only; never copy the repaired POM to runtime."""

from pathlib import Path

root = Path(__file__).resolve().parent
solve = (root / "tasks/maven-slf4j-conflict/solution/solve.sh").read_text()
marker = "cat > pom.xml << 'EOF'\n"
if solve.count(marker) != 1:
    raise ValueError("Expected exactly one dependency POM heredoc")
body = solve.split(marker, 1)[1]
if "\nEOF" not in body:
    raise ValueError("Dependency POM heredoc terminator missing")
pom = body.split("\nEOF", 1)[0]
(root / "dependency-pom.xml").write_text(pom)
