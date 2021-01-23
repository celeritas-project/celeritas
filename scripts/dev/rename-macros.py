#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""\
Recursively modify files.
"""

import logging as log
import os
import os.path
import re

REPLACE = {
    'Copyright 2020': 'Copyright 2021',
    'REQUIRE': 'CELER_EXPECT',
    'CHECK': 'CELER_ASSERT',
    'ENSURE': 'CELER_ENSURE',
    'CHECK_UNREACHABLE': 'CELER_ASSERT_UNREACHABLE()',
    'INSIST': 'CELER_VALIDATE',
}
RE_REPLACE = re.compile(r'\b(' + '|'.join(REPLACE.keys()) + r')\b')


def replace_macro_names(matchobj):
    return REPLACE[matchobj.group(1)]


def update_macros(filename):
    with ReWriter(filename) as rewriter:
        (old, new) = rewriter.files
        for line in old:
            orig_line = line
            line = RE_REPLACE.sub(replace_macro_names, line)
            if line != orig_line:
                rewriter.dirty = True
            new.write(line)


class HasExtension(object):
    """Helper class that says "yes the extension is valid" for any non-empty
    extension.
    """

    def __init__(self):
        pass

    def __contains__(self, val):
        # return val != ""
        return bool(val)


def walk_filtered_paths(paths, invisibles=False, extensions=None):
    """Walk the entire directory hierarchy for files to process."""
    if extensions is None:
        # By default, parse anything with an extension.
        extensions = HasExtension()

    for path in paths:
        for (root, dirs, files) in os.walk(path):
            # Don't look in invisible dirs like .hg or .git
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filepath in files:
                # Don't apply to invisibles if applicable
                if not invisibles and filepath.startswith("."):
                    continue

                # Don't apply if not in the given extensions
                (_, ext) = os.path.splitext(filepath)
                if extensions and ext not in extensions:
                    continue

                filepath = os.path.join(root, filepath)

                # Don't apply if it's a symlink
                if os.path.islink(filepath):
                    log.info("Skipping symlink %s", filepath)
                    continue

                yield filepath


class ReWriter(object):
    """Handle safe writing of new files.

    This takes care of error conditions as well as being graceful about not
    touching files that don't get changed.
    """

    def __init__(self, filename, preserve=False):
        (base, ext) = os.path.splitext(filename)

        self.filename = filename
        self.tempfilename = base + ".temp" + ext
        self.origfilename = base + ".orig" + ext

        # New and original files
        self.infile = open(filename, "r")
        self.outfile = open(self.tempfilename, "w")

        # Whether we've changed the file
        self.dirty = False
        # Whether to preserve the original
        self.preserve = preserve

    def close(self, did_error=False):
        self.infile.close()
        self.outfile.close()

        if did_error:
            log.error("ERROR while processing %s" % self.filename)
            if not self.preserve:
                # Failure and we want to delete the temp file
                os.unlink(self.tempfilename)
        elif self.dirty:
            log.info("CHANGED file %s" % self.filename)
            # Swap original and temp
            os.rename(self.filename, self.origfilename)
            os.rename(self.tempfilename, self.filename)

            # Delete the old file
            if not self.preserve:
                os.unlink(self.origfilename)
        else:
            log.info("No changes to %s" % self.filename)
            # No changes, no errors; delete the temp file
            os.unlink(self.tempfilename)

    def __enter__(self):
        "Called when entering context using 'with'"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        "Called when leaving context"
        self.close(did_error=(exc_type is not None))
        return False

    @property
    def files(self):
        """Get the list of in/out files.

        This is for setting:
        >>> (infile, outfile) = rewriter.files
        """
        return (self.infile, self.outfile)


def main(argv=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Update assertion macros"
    )
    parser.add_argument(
        '-r', dest="recursive",
        help="Recursive",
        action='store_true')
    parser.add_argument(
        '-x', '--extensions',
        help="Comma-separated extensions to use when searching recursively",
        default=".hh,.cc,.cu")
    parser.add_argument(
        'path', nargs='+',
        help="Files/dirs to process")

    args = parser.parse_args(argv)

    log.basicConfig(level=log.INFO)

    if args.recursive:
        extensions = None
        if args.extensions:
            extensions = args.extensions.split(',')
            log.info("Recursively searching for %s files",
                     ", ".join(extensions))
        paths = walk_filtered_paths(args.path, extensions=extensions)
    else:
        paths = iter(args.path)

    for filename in paths:
        try:
            update_macros(filename)
        except Exception as e:
            log.error("While processing %s:", filename)
            log.exception(e)

    log.info("Done.")


if __name__ == '__main__':
    main()
