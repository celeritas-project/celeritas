"""Find source files affected by changed headers in clang-scan-deps output."""

import json
import os
import re
import sys


header_file, source_file, dependency_file, regex_file = sys.argv[1:]
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
        if (
            source_path.endswith((".cc", ".cpp", ".cu"))
            and headers & resolved_dependencies
        ):
            affected_sources.add(source_path)

relative_sources = sorted(
    os.path.relpath(source, root) for source in changed_sources | affected_sources
)
with open(regex_file, "w") as output:
    if relative_sources:
        output.write(
            "(?:^|/)(?:" + "|".join(re.escape(path) for path in relative_sources) + ")$"
        )
print(len(relative_sources))
