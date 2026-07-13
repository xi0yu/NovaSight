"""Isolated, versioned mouse-control algorithm implementations.

Each child package owns its configuration names and mutable control state. The
runtime selects exactly one algorithm ID; child internals are not combined.
"""
