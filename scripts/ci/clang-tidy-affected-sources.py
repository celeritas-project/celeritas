"""Find source files affected by changed headers in clang-scan-deps output."""

import json
import os
import re
import sys


mode, header_file, source_file, dependency_file, regex_file = sys.argv[1:]
if mode not in {"all", "one"}:
    sys.exit(f"invalid header source selection mode: {mode}")

root = os.path.realpath(os.getcwd())
headers = {
    os.path.realpath(os.path.join(root, line.strip()))
    for line in open(header_file)
    if line.strip()
}
changed_sources = {
    os.path.realpath(os.path.join(root, line.strip()))
    for line in open(source_file)
    if line.strip()
}
data = json.load(open(dependency_file))
affected_sources = set()
source_by_header = {}

for unit in data.get("translation-units", []):
    for command in unit.get("commands", []):
        source = command.get("input-file") or command.get("input_file")
        if source is None:
            continue
        directory = command.get("directory", root)
        source_path = os.path.realpath(os.path.join(directory, source))
        dependencies = command.get("file-deps", command.get("file_deps", []))
        resolved_dependencies = {
            os.path.realpath(os.path.join(directory, dependency))
            for dependency in dependencies
        }
        if source_path.endswith((".cc", ".cpp", ".cu")):
            matching_headers = headers & resolved_dependencies
            if mode == "all":
                if matching_headers:
                    affected_sources.add(source_path)
            else:
                for header in matching_headers:
                    source_by_header[header] = min(
                        source_path, source_by_header.get(header, source_path)
                    )

if mode == "one":
    affected_sources = set(source_by_header.values())

relative_sources = sorted(
    os.path.relpath(source, root) for source in changed_sources | affected_sources
)
with open(regex_file, "w") as output:
    if relative_sources:
        output.write(
            "(?:^|/)(?:" + "|".join(re.escape(path) for path in relative_sources) + ")$"
        )
print(len(relative_sources))
