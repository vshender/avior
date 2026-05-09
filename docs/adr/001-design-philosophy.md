# ADR-001: Design philosophy

Date: 2026-05-09

## Status

Accepted

## Context

This ADR captures the design DNA that will guide all subsequent decisions in
avior.  Specific architectural and API decisions live in their own ADRs; this
one defines the principles those decisions answer to.

## Decision

avior is built around the following commitments:

1. **Minimal core.**  Core covers what happens inside a single agent run:
   how the agent talks to a model, invokes tools, and surfaces what's
   happening.  Anything that reaches across runs — between agents,
   between sessions, between systems — is built on top of core.
   Smallness is a feature, not a milestone we hit and abandon.

2. **Mechanisms, not policies, in core.**  Core provides building blocks.
   Opinionated decisions about how to combine them — coordination,
   retry and recovery, control flow patterns — are policies, and
   policies live in implementations built on top of core, not in core
   itself.

3. **Orthogonal primitives.**  Each primitive does one thing.  Special
   cases are avoided wherever possible; primitives compose in any
   combination that makes semantic sense.  Composition happens at the
   user's layer, not by combining responsibilities inside primitives.

4. **Minimum magic, maximum transparency.**  No hidden control flow, no
   implicit retries, no opaque queues.  Docstrings explain the mechanism
   clearly enough that the common case doesn't require reading the
   implementation; when you do dive in, any single piece of functionality
   is small enough to understand in one sitting.

5. **Pluggability with sensible defaults.**  Anything that varies in real
   deployments — where tools execute, where state lives, how messages
   are persisted, how telemetry flows — has a default implementation but
   is structured as a seam, not a hard-coded path.  Users can extend or
   replace any default without forking core.

6. **Ergonomic surface.**  The API should be simple and clear to read
   and to use.  Smallness of core is not an excuse for ceremony; sensible
   defaults make the happy path short, and complexity surfaces only when
   the user opts into it.

7. **Resumability and inspectability shape core.**  Any agent run can be
   paused, its state inspected and persisted, and resumed later —
   possibly in a different process, on a different machine, after human
   input, or against a recorded trace.   This is not a layer on top: it
   is a constraint that shapes core types — state is serializable, and
   the loop hands control back to the host at well-defined boundaries
   rather than running as an opaque blocking call.

8. **Provider-honest core.**  Where providers differ in meaningful ways —
   caching semantics, reasoning blocks, structured output guarantees,
   citations — core surfaces those differences in its types rather than
   hiding them behind a lowest-common-denominator abstraction.

## Consequences

- Decisions that add policy to core require explicit justification and
  an ADR documenting why a mechanism alone wouldn't suffice.
- Pattern implementations live outside core, even when convenient to
  inline them.
- Core types must be serializable; runtime state held only in closures
  or non-serializable handles is not allowed in core.
- Provider-specific concepts that don't fit the common shape get
  first-class typed representations, not metadata dictionaries.
- Common APIs cover what is genuinely common across providers;
  provider-unique features remain accessible as first-class, named
  extensions.  Agent code for the common case is provider-agnostic;
  provider-specific code surfaces only when the user opts into a
  provider-unique feature.
