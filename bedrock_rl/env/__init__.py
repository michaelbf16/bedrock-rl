"""The world and one episode of it.

`engine` drives the netherite process and is the only thing here that
knows about pipes and frame files. `episode` is the one class every live
harness and data generator runs through, and it owns deterministic reward.
`task` is the YAML layer, `observation` turns
engine state into the text a model reads, `actions` is the base action
language, `gui_layout` is the inventory screen's slot geometry, and
`snapshots` builds the start states a task restores from.

Nothing here imports a trainer, and nothing here imports torch.
"""
