#!/usr/bin/env python3
import sys
import re
from collections import Counter

def main():
    # Regex to extract filename and line number.
    pattern = re.compile(r'^(.*?)\((\d+)\):')
    counts = Counter()

    # Read lines from standard input.
    for line in sys.stdin:
        m = pattern.match(line)
        if m:
            filename = m.group(1)
            lineno = int(m.group(2))
            counts[(filename, lineno)] += 1

    # Create list of tuples (filename, line, count) and sort:
    # sort primarily by decreasing count, then filename and line number.
    entries = [(fn, ln, cnt) for (fn, ln), cnt in counts.items()]
    entries.sort(key=lambda x: (-x[2], x[0], x[1]))

    # Group sequential lines from the same file with identical hit counts.
    groups = []
    if entries:
        current_fn, current_ln, current_cnt = entries[0]
        start_ln = current_ln
        end_ln = current_ln

        for fn, ln, cnt in entries[1:]:
            if fn == current_fn and cnt == current_cnt and ln == end_ln + 1:
                end_ln = ln
            else:
                groups.append((current_fn, start_ln, end_ln, current_cnt))
                current_fn, current_ln, current_cnt = fn, ln, cnt
                start_ln = ln
                end_ln = ln
        groups.append((current_fn, start_ln, end_ln, current_cnt))

    # Print the results.
    for fn, start, end, cnt in groups:
        if cnt == 1:
            continue
        line_range = f"{start}" if start == end else f"{start}-{end}"
        print(f"{fn}({line_range}): {cnt}")

if __name__ == '__main__':
    main()
