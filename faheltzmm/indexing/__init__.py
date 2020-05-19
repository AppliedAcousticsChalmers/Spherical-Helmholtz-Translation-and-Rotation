"""Small helper module to manage indexing of arrays of coefficients.

The various coefficients used in this package have multiple indices,
which often does not fit with normal programming conventions.
This module contains a collection of methods to index the arrays
where coefficients are stored, and to convert between various schemes.
"""

from ._expansions import expansions
