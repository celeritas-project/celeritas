"""Find source files affected by changed headers in clang-scan-deps output."""

import json
import os
import re
import sys


header_file, dependency_file, regex_file = sys.argv[1:]
root = os.path.realpath(os.getcwd())
headers = {
    os.path.realpath(os.path.join(root, line.strip()))
    for line in open(header_file)
    if line.strip()
}
data = json.load(open(dependency_file))
selected = set()

for unit in data.get("translation-units", []):
    source = unit.get("input-file") or unit.get("input_file")
    dependencies = []
    for command in unit.get("commands", []):
        dependencies.extend(command.get("file-deps", command.get("file_deps", [])))
    if source is None:
        continue
    source_path = os.path.realpath(source)
    if any(os.path.realpath(dep) in headers for dep in dependencies):
        if source_path.endswith((".cc", ".cpp", ".cu")):
            selected.add(source_path)

relative_sources = sorted(os.path.relpath(source, root) for source in selected)
with open(regex_file, "w") as output:
    if relative_sources:
        output.write(
            "(?:^|/)(?:" + "|".join(re.escape(path) for path in relative_sources) + ")$"
        )
print(len(relative_sources))
