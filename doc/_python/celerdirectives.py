# Copyright Celeritas contributors: see top-level COPYRIGHT file for details
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Custom directive for Celeritas breathe usage.
"""

from docutils.parsers.rst import directives
from breathe.directives import BaseDirective
from breathe.directives.class_like import DoxygenStructDirective


# Custom directive for Celeritas structs
class CelerStructDirective(DoxygenStructDirective):
    """Custom directive that wraps doxygenstruct with Celeritas-specific defaults."""

    def run(self):
        # Get the struct name from arguments
        struct_name = self.arguments[0]

        # Prepend celeritas:: namespace if not already present
        if not struct_name.startswith("celeritas::"):
            self.arguments[0] = f"celeritas::{struct_name}"

        # Set default options
        self.options["members"] = ""
        self.options["no-link"] = True

        # Call the parent implementation
        return super().run()
