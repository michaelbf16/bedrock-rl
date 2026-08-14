"""ProgramError: what an action language raises on a payload it cannot
compile.

An action language turns the text a model emitted into a schedule of
engine action keys; the shipped one is the computer-use compiler
(bedrock_rl/adapters/netherite/computer.py::compile_actions). It does not own the
exception, because Episode.step is what catches it, and what it does with
it is the definition of a malformed turn. The turn executes no ACTIONS,
it costs 0.2 of reward, and the episode carries on. It still spends the
one tick every turn spends taking its screenshot, even when rendering is
disabled.

Keeping it in its own module is what stops the observation renderer from
owning an exception it never raises, and lets a second compiler be added
without importing the first.
"""


class ProgramError(ValueError):
    """A turn whose action payload did not compile. Worth -0.2 and no
    executed actions; see bedrock_rl/env/episode.py::Episode.step."""
