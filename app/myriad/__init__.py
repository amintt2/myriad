"""myriad: a decentralised LLM made of one node per PC, answers fused across model families.

The package was called `essaim` before; the console scripts `essaim` and `essaim-desktop` remain as
aliases, and an existing `essaim` data directory is copied to `myriad` on first start (config.py)."""

__version__ = "0.1.0"
# Wire protocol identifiers. They keep the former name `essaim` ON PURPOSE and must not be renamed:
# PROTOCOL prefixes every signed message (job, result, receipt, challenge answer: crypto.py) and is
# the `version` literal that every node announces (protocol.py), so renaming it would make the nodes,
# gateways and trackers already deployed reject the new ones' signatures and frames (and vice versa).
# It is a protocol name, not the package name.
# Signature domain and wire format of every essaim/1 frame: unchanged in essaim/1.1, so that old and
# new nodes keep verifying each other's signatures.
PROTOCOL = "essaim/1"
# essaim/1.1 adds optional frames, used only when both sides support them (backward compatible):
# server-side peer selection (route), application-level pings with an engine check (ping) and the
# selection endpoint GET /v1/select (select).
# essaim/1.2 adds skill tags (tags): nodes advertise tags, jobs can be routed by tag or by model family.
# "update" (additive, same protocol version): a node that asks for it when it connects (WebSocket query
# `?features=update`) gets the latest app version in its Welcome frame and an UpdateAvailable frame when
# a new release appears. Nodes that do not ask never receive either (their parser forbids both).
PROTOCOL_VERSION = "essaim/1.2"
FEATURES = ("route", "ping", "select", "tags", "update")
