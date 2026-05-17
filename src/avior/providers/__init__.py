"""Concrete `Provider` implementations bundled with avior.

Each submodule is an opt-in adapter to a specific LLM service.  Submodules
that depend on third-party SDKs (e.g. `anthropic`) require the matching
optional extra to be installed (`pip install avior[anthropic]`).
"""
