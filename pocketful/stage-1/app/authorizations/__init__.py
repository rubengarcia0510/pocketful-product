"""Stage 2: authorisations, holds, captures and voids.

This package owns the persistence layer for stage-2's authorisation
lifecycle (open → captured / voided / expired) and the derived
``available`` / ``held`` funds computation. The HTTP API and the
service-level orchestration live elsewhere; this module deals only with
schema, dataclasses, and the read/write operations on the connection.
"""
