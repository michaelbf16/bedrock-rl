# Third-party patch notices

bedrock-rl ships local patch files for pinned upstream projects. Those
patches remain subject to the upstream projects' licenses; the repository's
MIT license applies to bedrock-rl itself.

- `patches/verl/*.diff` modifies
  [verl](https://github.com/verl-project/verl) at commit
  `7aed6b230776f963fa09509c10d9c3a767d1102c`. The verbatim Apache 2.0
  license and upstream notice are included under `verl/`.
- `patches/netherite/*.diff` modifies
  [Netherite](https://github.com/Infatoshi/netherite) at commit
  `e78272703f38c823db917ea5e66ce336101499ae`. That pinned upstream tree
  does not declare a project license. These files are recorded here for
  provenance; this repository does not represent that its MIT license grants
  rights in Netherite.
