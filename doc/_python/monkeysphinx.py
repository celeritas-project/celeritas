# Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
"""
Monkey-patch sphinx to shoehorn latex output into an ORNL TM compatible style.

Note: this is known to work as of ~January 2023::

    py-sphinxcontrib-bibtex@2.5.0
    py-pybtex@0.24.0
    py-sphinx@5.3.0
"""

import sphinx

try:
    from sphinx.writers.html5 import HTML5Translator
    from sphinx.writers.latex import LaTeXTranslator, Table
    from sphinx.builders.latex.transforms import BibliographyTransform
except ImportError as e:
    print("ERROR: failed to import writers/builders:", e)
    LaTeXTranslator = Table = BibliographyTransform = None

def monkey(cls, replace=True):
    if cls is None:
        def _monkey(func):
            return func
    else:
        def _monkey(func, cls=cls, replace=replace):
            exists = hasattr(cls, func.__name__)
            if exists != replace:
                print("ERROR: class {} {} method {}".format(
                    cls.__name__, "has no" if replace else "already has a",
                    func.__name__))
            else:
                # print("Applying patch to {}.{}".format(
                #         cls.__name__, func.__name__))
                setattr(cls, func.__name__, func)
            return func
    return _monkey

@monkey(LaTeXTranslator)
def visit_desc_annotation(self, node):
    self.body.append(r'\sphinxannotation{')

@monkey(LaTeXTranslator)
def depart_desc_annotation(self, node):
    self.body.append(r'}')

@monkey(LaTeXTranslator, replace=False)
def visit_enquote(self, node):
    self.body.append(r'``')

@monkey(LaTeXTranslator, replace=False)
def depart_enquote(self, node):
    self.body.append(r"''")

@monkey(HTML5Translator, replace=False)
def visit_enquote(self, node):
    self.body.append(r'&ldquot;')

@monkey(HTML5Translator, replace=False)
def depart_enquote(self, node):
    self.body.append(r"&rdquot;")

# Replace bibliography's enclosing section rather than moving after appendices
@monkey(BibliographyTransform)
def run(self, **kwargs):
    from docutils import nodes
    from sphinx.builders.latex.nodes import thebibliography
    citations = thebibliography()
    section_parent = None
    for node in self.document.traverse(nodes.citation):
        parent = node.parent
        parent.remove(node)
        citations += node
        if section_parent is None:
            # Find first section parent
            while parent:
                if isinstance(parent, nodes.section):
                    section_parent = parent
                    break
                parent = parent.parent

    if section_parent and len(citations) > 0:
        section_parent.replace_self(citations)


@monkey(LaTeXTranslator)
def visit_colspec(self, node):
    # type: (nodes.Node) -> None
    self.table.colcount += 1
    if 'colwidth' in node:
        self.table.colwidths.append(node['colwidth'])
    if 'stub' in node:
        self.table.stubs.append(self.table.colcount - 1)

@monkey(LaTeXTranslator)
def depart_row(self, node):
    # Don't add horizontal rules between rows
    self.body.append('\\\\\n')
    self.table.row += 1

@monkey(Table)
def get_colspec(self):
    if self.colspec:
        return self.colspec
    if self.get_table_type() == 'tabulary':
        # sphinx.sty sets T to be J by default.
        return '{' + ('T' * self.colcount) + '}\n'
    return '{' + ('l' * self.colcount) + '}\n'
