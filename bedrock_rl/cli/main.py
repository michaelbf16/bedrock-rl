"""`bedrock`, the one command.

The repo has shell launchers in bin/, python utilities in bin/, and
`python -m` module entry points, and nothing in the names says which
applies. This module is one front door over all of them, grouped by what
a user is trying to do.

It wraps rather than reimplements. Training commands shell out to the
launchers in bin/, which own their argument handling. Only pure reads of
package data run in process.

`train` and `generate` take config files rather than growing a second
command-line schema. Training maps onto the launchers; generation resolves
import-path components from ``bedrock_rl.sdg.generation``.
"""
import argparse
import asyncio
import os
import subprocess
import sys

from rich.text import Text

from bedrock_rl import resources
from bedrock_rl.cli import console as C
from bedrock_rl.config import training as config
from bedrock_rl.config.paths import (
    DEFAULT_MODEL,
    DEFAULT_TASK,
    TASK_HELP,
    absolute_path as here,
    config_relative,
    model_arg,
    task_files,
    task_yaml,
)

PROG = "bedrock"

# What each group is for, in the order a first run meets them. The help
# screen is generated from this, so a new group cannot be added without
# appearing there.
GROUPS = [
    ("setup", "build the engine and the training venv", ""),
    ("doctor", "check this box and say what is broken", ""),
    ("smoke", "env, task and reward sanity, no GPU", ""),
    ("trial", "run one agent attempt at an import-path task", ""),
    ("train", "run one trainer or workflow from a config file", ""),
    ("generate", "run data generation from a config file",
     "bedrock generate examples/equip_pickaxe_grpo/data.yaml train"),
    ("eval", "score and compare checkpoints on real episodes",
     "trials | summarize"),
    ("models", "the supported-model registry", "list | resolve | check"),
    ("task", "task files and what they check",
     "list | show | validate | render | snapshots"),
]

START_HERE = [
    ("bedrock task show equip_pickaxe_grpo",
     "inspect a Minecraft task and its reward"),
    ("bedrock setup", "build Netherite and the training venv"),
    ("bedrock doctor", "confirm the box is ready"),
    ("bedrock smoke", "sanity-check the env with no GPU"),
    ("bedrock generate examples/equip_pickaxe_grpo/data.yaml --check",
     "validate every job in the included example"),
]


# ── locating the checkout ────────────────────────────────────────────────

def repo_root():
    root = resources.repo_root()
    if root is None:
        raise SystemExit(
            "this command needs the bedrock-rl checkout, and the package "
            "was imported from an installed wheel, where bin/ and examples/ "
            "do "
            "not exist.\n"
            "  git clone https://github.com/michaelbf16/bedrock-rl "
            "bedrock-rl")
    return root


def script(*parts):
    return str(repo_root().joinpath(*parts))


def display_path(path):
    """A path as a reader should see it: relative to the checkout when
    there is one and the file is inside it, absolute otherwise.

    An install from a wheel has no checkout, and os.path.relpath against
    the SystemExit repo_root() raises is how `bedrock task show <a
    packaged task>` used to die AFTER loading the file and finding
    nothing wrong with it.
    """
    root = resources.repo_root()
    if root is not None:
        try:
            rel = os.path.relpath(path, str(root))
        except ValueError:                  # different drive on Windows
            return str(path)
        if not rel.startswith(".."):
            return rel
    return str(path)


def package_env(a, knobs=None):
    """The environment shared by package and checkout commands.

    This half deliberately does not ask for a source checkout.  Commands
    which only validate a user-owned config must keep working when the CLI
    came from a wheel.  Checkout launchers add their import root in
    ``checkout_env`` below.

    `knobs` is layer 2, the config file. The three location flags and
    `-e` are layer 4 and are applied after it, so the command line wins
    over the file and the file wins over whatever the shell exported.
    bedrock_rl/config/training.py states that order once and this is the
    only place it is carried out.
    """
    env = dict(os.environ)
    # So a tee'd log holds the child's output in the order it happened.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(knobs or {})
    for attr, name in (("netherite_home", "NETHERITE_HOME"),
                       ("verl_env", "VERL_ENV"),
                       ("verl_home", "VERL_HOME")):
        v = getattr(a, attr, None)
        if v:
            # here(), not expanduser alone: these are the same three
            # paths as the netherite_home / verl_env / verl_home config
            # keys, and layer 4 must not resolve them differently from
            # layer 2.
            env[name] = here(v)
    for kv in getattr(a, "env", None) or []:
        if "=" not in kv:
            raise SystemExit(f"--env wants KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        env[k] = v
    return env


def checkout_env(a, knobs=None):
    """``package_env`` plus the source root required by ``bin/``.

    The separate training venv does not install this package; launchers
    import it through ``PYTHONPATH``.  Keeping that checkout-only fact out of
    ``package_env`` is what lets installed-wheel validation commands run.
    """
    env = package_env(a, knobs)
    root = str(repo_root())
    # No trailing separator on an empty PYTHONPATH: an empty entry means the
    # current directory, which is not what this is asking for.
    env["PYTHONPATH"] = os.pathsep.join(
        [root] + [p for p in [env.get("PYTHONPATH", "")] if p])
    return env


def echo_env(env):
    """The knobs a child will get that this shell does not have.

    On a wrapping CLI they are the part a reader cannot see in the
    command line, and they are exactly the thing this layer translates.
    """
    for k, v in sorted((env or {}).items()):
        if os.environ.get(k) != v:
            C.assign(k, v)


def run(argv, env=None, cwd=None, dry=False, show_env=True):
    """Hand over to a launcher or a module. Its output is its own.

    `show_env` is off for the second and later stages of one command, so
    a dry run that hands the same environment to three modules prints it
    once rather than three times.
    """
    if dry:
        if show_env:
            echo_env(env)
        C.command(argv)
        return 0
    C.command(argv)
    # the child writes straight to this process's file descriptors, so
    # anything still buffered above would land after its output
    C.flush()
    sys.stdout.flush()
    try:
        return subprocess.call(argv, env=env, cwd=cwd)
    except FileNotFoundError as e:
        C.error(e)
        return 127
    except KeyboardInterrupt:
        return 130


def verl_python(env, dry=False):
    """The training venv's interpreter, the one CUDA-locked half runs in.

    A missing venv is only fatal when something is about to run in it.
    --dry-run says it prints the command and stops, and the box where
    you most want to read that command is the one that is not set up.
    """
    venv = os.path.expanduser(env.get("VERL_ENV", "~/verl-env"))
    py = os.path.join(venv, "bin", "python")
    if not os.access(py, os.X_OK) and not dry:
        raise SystemExit(
            f"training venv missing at {venv}; run `bedrock setup`, "
            "or set VERL_ENV")
    return py


def in_verl(env, argv, dry=False):
    """Run something in the training venv, the way the launchers do.

    --no-project keeps uv from syncing this repo's env over a stack that
    is pinned to a CUDA ABI.
    """
    return ["uv", "run", "--no-project", "--python", verl_python(env, dry),
            "python"] + argv


# ── tasks ────────────────────────────────────────────────────────────────
# A path is taken as given, a bare name is looked up, and the lookup is
# bedrock_rl/resources.py::find_task. That is the ONE rule: this file
# and bin/_common.sh both call it, because they used to hold a rule each
# and the two disagreed about every generated rung.
#
# The set of directories searched is `resources.TASK_DIRS` and nothing
# else. Anything a listing command can see reads as a thing to train on,
# so a directory is added there deliberately or not at all; a held-out
# split or a shaped variant that appears in `bedrock task list` is an
# invitation to train on it.

# ── setup ────────────────────────────────────────────────────────────────

def cmd_setup(a):
    env = checkout_env(a)
    # This is an internal shell hand-off, not an ambient setup knob. A prior
    # invocation must not make a later plain `bedrock setup` skip training.
    env.pop("BRL_RUNTIME_ONLY", None)
    if a.runtime_only:
        training_flags = []
        if a.python:
            training_flags.append("--python")
        if a.no_flash_attn:
            training_flags.append("--no-flash-attn")
        if a.install_deepspeed:
            training_flags.append("--install-deepspeed")
        if training_flags:
            raise SystemExit(
                "--runtime-only stops after building the Minecraft runtime; "
                f"it cannot be combined with {', '.join(training_flags)}")
        env["BRL_RUNTIME_ONLY"] = "1"
    if a.mc_jar:
        env["MC_JAR"] = os.path.expanduser(a.mc_jar)
    if a.texture_pack:
        env["TEXTURE_PACK"] = os.path.expanduser(a.texture_pack)
    if a.python:
        env["VERL_PYTHON"] = a.python
    if a.no_flash_attn:
        env["FLASH_ATTN"] = "0"
    if a.install_deepspeed:
        env["INSTALL_DEEPSPEED"] = "1"
    if a.force:
        env["FORCE_PINNED_CHECKOUT"] = "1"
    return run(["bash", script("bin", "setup_deps.sh")], env=env,
               dry=a.dry_run)


# ── smoke ────────────────────────────────────────────────────────────────

def cmd_smoke(a):
    env = checkout_env(a)
    return run([sys.executable, "-m", "bedrock_rl.cli.smoke",
                task_yaml(a.task)], env=env, dry=a.dry_run)


# ── doctor ───────────────────────────────────────────────────────────────

def cmd_doctor(a):
    from bedrock_rl.cli import doctor
    con = C.get_console()
    checks = doctor.run(netherite_home=a.netherite_home,
                        verl_env=a.verl_env, verl_home=a.verl_home,
                        model=a.model, runtime_only=a.runtime_only)
    t = C.table({"header": ""}, {"header": "check", "style": "brl.head"},
                {"header": "detail", "overflow": "fold"},
                title="bedrock doctor")
    for c in checks:
        t.add_row(C.mark(c.state), c.name, c.detail)
    con.print(t)

    # Every line below is something to paste, so all of them go through
    # C.paste, which does not fold a command line onto a second row.
    broken = [c for c in checks if c.broken]
    warned = [c for c in checks if c.state == "warn"]
    if broken or warned:
        con.print()
        con.print("what to do", style="brl.head")
        seen = set()
        for c in broken + warned:
            if c.fix and c.fix not in seen:
                seen.add(c.fix)
                C.paste(c.fix, style="brl.cmd" if c.broken else "brl.dim")
    if broken:
        con.print(f"\n{len(broken)} required check"
                  f"{'' if len(broken) == 1 else 's'} failed.",
                  style="brl.fail")
        return 1
    con.print("\nready.", style="brl.ok")
    return 0


# ── train and generate ───────────────────────────────────────────────────
# Both take a config file. Every key it may carry, every knob those keys
# become, and the precedence between the file, the environment and the
# command line are in bedrock_rl/config/training.py. Do NOT add a flag here
# that sets a knob: it would be a second spelling of something that file
# already names, which is how the previous interface came to reach only
# the knobs somebody had written a flag for.


def config_path(arg, group):
    """The config file, or an error that names one that exists.

    The old verb form is recognised HERE rather than left to fail as a
    missing file, because `bedrock train rl` is in a lot of shell
    history and "no config file at ./rl" does not tell anyone what to do
    about it.
    """
    if arg is None:
        raise SystemExit(f"{PROG} {group} takes a config file\n"
                         + _example_lines(group))
    path = here(arg)
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        raise SystemExit(f"{path} is a directory, and this takes one "
                         f"config file\n" + _example_lines(group))
    if group == "train" and arg in config.TARGETS:
        kind = "trainer" if arg in config.TRAINERS else "workflow"
        shipped = config.example_for(kind, arg) or ""
        line = (f"  {PROG} train {config.relative(shipped)}\n"
                if shipped else "")
        raise SystemExit(
            f"`{PROG} train {arg}` is gone: the {kind} is named inside a "
            f"config file now, so a run is one file rather than a line of "
            f"flags that could only reach the knobs somebody had named.\n"
            + line +
            f"  {PROG} train <config.yaml> --set steps=4   to vary one key\n"
            f"The launcher and its environment knobs are unchanged:\n"
            f"  bash bin/{config.TARGETS[arg].script} equip_pickaxe_grpo "
            f"qwen3-vl 30")
    raise SystemExit(f"no config file at {path}\n" + _example_lines(group))


def _example_lines(group):
    """The shipped configs for this command, as lines to paste.

    A file that will not parse is skipped rather than raised through,
    because the docs invite people to keep their own configs in that
    directory and one broken file there must not turn every unrelated
    error into that file's YAML error.
    """
    shipped = []
    for p in config.examples():
        try:
            doc = config.load(p)
            is_train = "trainer" in doc or "workflow" in doc
        except SystemExit:
            continue
        if is_train == (group == "train"):
            shipped.append(p)
    if not shipped:
        return f"  {PROG} {group} <config.yaml>"
    return "\n".join(f"  {PROG} {group} {config.relative(p)}"
                     for p in shipped)


def warn_inherited(plan, env):
    """Say when a knob this run reads came from the shell.

    Layer 1 of the precedence order is the environment, and it is the
    only layer `--dry-run` cannot show, because the echo prints what the
    child gets that this shell does not already have. So an `export
    OUT=/tmp/x` in a tmux profile silently moves the checkpoints of a run
    whose config names no `out:`, and the one command whose whole job is
    to be read before spending GPUs says nothing about it.

    Only knobs THIS training target reads, so the warning stays rare enough to
    mean something.
    """
    for knob in config.knobs_for(plan.target):
        if knob.ambient or knob.env in plan.env:
            continue
        if knob.env not in os.environ:
            continue
        C.warn(f"{knob.env}={os.environ[knob.env]} is inherited from this "
               f"shell and not from the config, and bin/{plan.script} "
               f"reads it")


def cmd_knobs(a):
    """Every key a trainer or workflow config may carry.

    The old interface listed its knobs in `--help`, because each one was
    a flag. A config file has no such listing for free, and a schema
    nobody can enumerate is one people guess at, so this is that listing
    read out of the same table the config is checked against.
    """
    name = a.knobs
    if name not in config.TARGETS:
        raise SystemExit(
            f"no trainer or workflow named {name!r}\n"
            "  trainers: " + " | ".join(config.TRAINERS) + "\n"
            "  workflows: " + " | ".join(config.WORKFLOWS))
    kind = "trainer" if name in config.TRAINERS else "workflow"
    spec = config.TARGETS[name]
    con = C.get_console()
    con.print(Text(f"{PROG} train <config.yaml>  with {kind}: {name}",
                   style="brl.head")
              + Text(f"\n{spec.what}, bin/{spec.script}", style="brl.dim"))
    t = C.table({"header": "key", "style": "brl.head"},
                {"header": "becomes"},
                {"header": "what it does", "overflow": "fold"},
                title=f"{name} config keys")
    for key, env, what in (
            (kind, "-", f"{name}"),
            ("task", "positional 1", "example name, or a task YAML path"),
            ("model", "positional 2",
             "registry alias, HuggingFace id or checkpoint path"),
            ("steps", "positional 3", spec.steps),
            ("env", "-", "any variable at all, verbatim")):
        t.add_row(key, env, what)
    if name == "rl":
        t.add_row("algorithm", "ADV", " | ".join(config.RL_ALGORITHMS))
    for knob in config.knobs_for(name):
        what = knob.what
        if knob.switch:
            what += f" (true is {knob.env}={knob.on})"
        t.add_row(knob.key, knob.env, what)
    con.print(t)
    return 0


def cmd_train(a):
    if a.knobs:
        return cmd_knobs(a)
    path = config_path(a.config, "train")
    config_dir = os.path.dirname(path)
    resolve = lambda value: config_relative(value, config_dir)
    plan = config.train_plan(config.load(path, a.preset), a.set, resolve=resolve,
                             origin=(config.relative(path)
                                     + (f":{a.preset}" if a.preset else "")))
    if a.modal:
        from bedrock_rl.launch.modal import ModalLaunchError, build_launch

        try:
            launch = build_launch(
                root=str(repo_root()),
                config=path,
                preset=a.preset,
                target=plan.target,
                nodes=a.nodes,
                gpu=a.gpu,
                volume=a.volume,
                secrets=a.secret or (),
                sets=a.set or (),
                train_env=a.env or (),
                timeout=a.timeout,
                rdma=a.rdma,
                smoke=a.modal_smoke,
                detach=a.detach,
                run_id=a.modal_run,
            )
        except ModalLaunchError as e:
            raise SystemExit(str(e))
        env = dict(os.environ)
        env.update(launch.environment())
        return run(launch.command(), env=env, cwd=str(repo_root()),
                   dry=a.dry_run)
    modal_only = (
        a.nodes != 1 or a.gpu is not None or a.secret or a.rdma
        or a.modal_smoke or a.detach or a.timeout != 86400
        or a.volume != "bedrock-rl-runs" or a.modal_run is not None
    )
    if modal_only:
        raise SystemExit("--nodes, --gpu, --volume, --secret, --timeout, "
                         "--rdma, --modal-smoke, --modal-run and --detach require "
                         "--modal")
    root = resources.repo_root()
    env = (checkout_env(a, plan.env) if root is not None
           else package_env(a, plan.env))
    warn_inherited(plan, env)
    # The launchers take [TASK] [MODEL] [N] and default each one. A
    # positional can only be defaulted by omitting it, and omitting one
    # in the middle is not a thing a shell can do, so an unset TASK or
    # MODEL is filled in with the same value the launcher would have
    # chosen, MODEL through $MODEL exactly as bin/_common.sh reads it.
    # Trailing unset positionals are dropped instead, which leaves the
    # launcher's own default in the one place it is written down.
    pos = [task_yaml(plan.task or DEFAULT_TASK, base_dir=config_dir),
           model_arg(plan.model or os.environ.get("MODEL", DEFAULT_MODEL),
                     base_dir=config_dir)]
    if plan.steps is not None:
        pos.append(str(plan.steps))
    # A dry run is a config operation: it is useful before cloning the large
    # engine and trainer checkouts.  Show the checkout-relative launcher when
    # this package came from a wheel; executing it still requires the clone.
    launcher = (str(root.joinpath("bin", plan.script)) if root is not None
                else os.path.join("bin", plan.script))
    argv = ["bash", launcher] + pos
    if a.dry_run:
        return run(argv, env=env,
                   cwd=(None if root is None else str(root)), dry=True)
    if root is None:
        repo_root()                 # the single, actionable clone error
    return run(argv, env=env, cwd=str(root))


def cmd_generate(a):
    path = config_path(a.config, "generate")
    if a.modal_scout:
        if a.check:
            raise SystemExit("--check and --modal-scout are mutually exclusive")
        from bedrock_rl.config.overrides import apply_override
        from bedrock_rl.sdg.generation import (
            GenerationSpec,
            load_generation_jobs,
        )
        from bedrock_rl.launch.modal import (
            ModalLaunchError,
            build_scout_launch,
        )

        jobs = load_generation_jobs(path)
        if a.preset is not None:
            if a.preset not in jobs:
                raise SystemExit(
                    f"{path} has no job {a.preset!r}; choose "
                    + ", ".join(str(name) for name in jobs))
            document = jobs[a.preset]
        elif len(jobs) == 1:
            document = next(iter(jobs.values()))
        else:
            raise SystemExit(
                f"{path} has multiple jobs; choose one: "
                + ", ".join(str(name) for name in jobs))
        for expression in a.set or ():
            apply_override(document, expression)
        spec = GenerationSpec.from_dict(document, path=path)
        if spec.worlds is None:
            raise SystemExit(
                "--modal-scout requires a grouped config with `worlds`")
        target = a.scout_worlds or spec.worlds
        attempts = (a.scout_attempts or spec.max_world_attempts or target)
        try:
            launch = build_scout_launch(
                root=str(repo_root()), config=path, job=a.preset,
                sets=a.set or (), target=target,
                max_candidates=attempts,
                containers=a.scout_containers,
                results=a.scout_results,
                run_id=a.scout_run,
            )
        except ModalLaunchError as exc:
            raise SystemExit(str(exc))
        env = dict(os.environ)
        env.update(launch.environment())
        return run(launch.command(), env=env, cwd=str(repo_root()),
                   dry=a.dry_run)
    scout_only = (
        a.scout_worlds is not None or a.scout_attempts is not None
        or a.scout_run is not None or a.scout_containers != 1000
        or a.scout_results != "bedrock-sdg-scout-v1")
    if scout_only:
        raise SystemExit("--scout-* options require --modal-scout")
    argv = [sys.executable, "-m", "bedrock_rl.cli.generate", path]
    if a.preset:
        argv.append(a.preset)
    for value in a.set or ():
        argv += ["--set", value]
    if a.check:
        argv.append("--check")
    # Generation resolves relative paths beside its manifest.  Keeping the
    # caller's cwd also leaves custom importable components reachable from an
    # installed package; no checkout script participates in this command.
    return run(argv, env=package_env(a), dry=a.dry_run)


# ── eval ─────────────────────────────────────────────────────────────────

def cmd_eval_trials(a, rest):
    env = checkout_env(a)
    data = here(a.data)
    if a.keep_latest_images is not None:
        if a.keep_latest_images < 0:
            raise SystemExit("--keep-latest-images cannot be negative")
    task = task_yaml(a.task)
    argv = ["--data", data, "--task", task, "--model", a.model]
    if a.n is not None:
        argv += ["--n", str(a.n)]
    if a.samples is not None:
        argv += ["--samples", str(a.samples)]
    if a.keep_latest_images is not None:
        argv += ["--keep-latest-images", str(a.keep_latest_images)]
    # `none` means no hint and no stop scaffold, so it passes no --hint
    # at all. Forwarding --hint with --hint-level none still applied
    # The evaluation driver's own --finish-level default, which is a
    # scaffold, so "none" was not hint-free.
    if a.hint and a.hint != "none":
        argv += ["--hint", task, "--hint-level", a.hint]
    if a.lora:
        argv += ["--lora", here(a.lora)]
    if a.blank_frames:
        argv.append("--blank-frames")
    cmd = in_verl(env, ["-m", "bedrock_rl.eval.trials"] + argv + rest,
                  dry=a.dry_run)
    return run(cmd, env=env, cwd=str(repo_root()), dry=a.dry_run)


def cmd_eval_summarize(a):
    argv = []
    for label, step, records in a.point:
        argv += ["--point", label, step, records]
    if a.base:
        argv += ["--base", a.base]
    argv += ["--summary", a.summary, "--svg", a.svg]
    if a.title:
        argv += ["--title", a.title]
    if a.training_metrics:
        argv += ["--training-metrics", a.training_metrics]
    if a.success_max is not None:
        argv += ["--success-max", str(a.success_max)]
    for label, summary in a.heldout_summary or ():
        argv += ["--heldout-summary", label, summary]
    if a.dry_run:
        print("summary", a.summary)
        print("curve  ", a.svg)
        return 0
    from bedrock_rl.eval import summary
    return summary.main(argv)


# ── models ───────────────────────────────────────────────────────────────

def cmd_models_list(a):
    from bedrock_rl import models as reg
    if a.markdown:
        print(reg.readme_table())
        return 0
    t = C.table("family", "sizes",
                {"header": "default HuggingFace id", "overflow": "fold"},
                {"header": "template", "overflow": "fold"},
                "status", title="supported models")
    for m in reg.MODELS:
        tmpl = "shipped by the model" if m.chat_template is None \
            else m.chat_template
        t.add_row(m.key, "/".join(m.sizes), m.default_id, tmpl,
                  Text(m.status,
                       style="brl.ok" if m.status == "tested" else "brl.dim"))
    con = C.get_console()
    con.print(t)
    con.print("a family alias, a size alias (qwen3-vl-4b), a HuggingFace "
              "id, or a local checkpoint path all resolve here",
              style="brl.dim")
    return 0


def cmd_models_resolve(a):
    import json

    from bedrock_rl import lora, models as reg
    try:
        spec = reg.resolve(a.model)
    except reg.UnknownModel as e:
        C.error(str(e).splitlines()[0], "\n".join(str(e).splitlines()[1:]))
        return 2
    path = reg.model_path(a.model)
    if a.json:
        d = dict(spec.__dict__)
        d["resolved_path"] = path
        print(json.dumps(d, indent=2, default=list))
        return 0
    t = C.table("field", "value", title=f"{a.model} resolves to {spec.key}")
    # The registry's DECLARED attention, not the resolved one. "auto"
    # means flash_attention_2 when flash-attn imports and sdpa
    # otherwise, and the venv that has to answer that is the training
    # venv, not this one. bin/_common.sh runs the lookup there for the
    # same reason. Printing sdpa here because the CPU env has no
    # flash-attn would be a confident wrong answer.
    attn = spec.attn_impl
    if attn == "auto":
        attn = "auto, resolved in the training venv by `bedrock doctor`"
    rows = [("resolved path", path),
            ("family", f"{spec.display} ({spec.provider})"),
            ("status", spec.status),
            ("chat template", spec.chat_template or "shipped by the model"),
            ("tool parser", spec.tool_parser),
            ("attention", attn),
            ("image history", "selected by the task context policy"),
            ("max prompt length", str(spec.max_prompt_length)),
            ("pixel band", f"{spec.image_min_pixels}"
                           f"-{spec.image_max_pixels}"),
            ("trust remote code", str(spec.trust_remote_code)),
            # Derived from vision_patterns above it, so a LoRA run does not
            # train adapters the rollout engine would throw away.
            ("lora exclusion", lora.exclude_regex(spec) or "nothing")]
    ov = reg.verl_overrides(spec, path)
    if ov:
        rows.append(("verl overrides", "\n".join(ov)))
    if spec.notes:
        rows.append(("notes", spec.notes))
    for k, v in rows:
        t.add_row(k, v)
    C.get_console().print(t)
    return 0


def cmd_models_check(a, rest):
    env = checkout_env(a)
    cmd = in_verl(
        env,
        ["-m", "bedrock_rl.cli.model_check", a.model] + rest,
        dry=a.dry_run,
    )
    return run(cmd, env=env, cwd=str(repo_root()), dry=a.dry_run)


# ── task ─────────────────────────────────────────────────────────────────

def _load(path):
    from bedrock_rl.env.task import Task
    return Task.load(path)


def cmd_task_list(a):
    t = C.table({"header": "task", "overflow": "fold"},
                {"header": "seeds", "justify": "right"},
                {"header": "turns", "justify": "right"},
                {"header": "success check"},
                {"header": "file", "overflow": "fold", "style": "brl.path"},
                title="tasks")
    for p in task_files():
        rel = display_path(p)
        try:
            task = _load(p)
            from bedrock_rl.adapters.netherite.computer import validate_cu_example
            validate_cu_example(task)
        except Exception as e:
            t.add_row(Text(os.path.basename(p), style="brl.fail"), "", "",
                      Text(type(e).__name__, style="brl.fail"), rel)
            continue
        t.add_row(task.name, str(len(task.seeds)),
                  str(task.max_turns), task.success.type, rel)
    con = C.get_console()
    con.print(t)
    con.print("pass a name or a path to any command that takes a task; "
              "`bedrock task show <name>` for the goal and the checks",
              style="brl.dim")
    return 0


def cmd_task_show(a):
    path = task_yaml(a.task)
    task = _load(path)
    con = C.get_console()
    t = C.table("field", {"header": "value", "overflow": "fold"},
                title=task.name)
    t.add_row("file", display_path(path))
    t.add_row("goal", task.goal)
    if task.raw.get("goal_cu"):
        t.add_row("goal shown to the model", task.raw["goal_cu"].strip())
    t.add_row("seeds", ", ".join(str(s) for s in task.seeds))
    t.add_row("snapshots", task.snapshots_dir or "none, worldgen from seed")
    t.add_row("budgets", f"{task.max_turns} turns, "
                         f"{task.max_ticks} ticks, "
                         f"{task.max_lines} actions per call")
    t.add_row("decision cost", str(task.decision_cost))
    t.add_row("success", f"{task.success.type} "
                         f"(worth {task.success.reward})")
    for k, v in task.success.spec.items():
        if k not in ("type", "reward"):
            t.add_row(f"  {k}", str(v))
    for i, s in enumerate(task.shaping):
        t.add_row(f"shaping {i + 1}",
                  f"{s.type} (worth {s.reward})")
    # The give-up defences are part of what a task pays, so they print
    # here for the same reason the shaping rungs do: this is the
    # documented way to see what a task is worth, and a number it does not
    # show is a number somebody will read off a comment instead.
    if task.fail is not None:
        t.add_row("fail", f"{task.fail.type} (worth {task.fail.reward}, "
                          "and ends the episode)")
    if task.no_commit:
        t.add_row("no commit", f"{task.no_commit}, charged to an episode "
                               "that neither succeeded nor failed")
    if task.converge is not None:
        t.add_row("converge", f"±{task.converge.reward} over "
                              f"{task.converge.span_deg} degrees, "
                              "potential-based")
    con.print(t)
    return 0


def cmd_task_validate(a):
    paths = [task_yaml(x) for x in a.tasks] if a.tasks else task_files()
    t = C.table({"header": ""},
                {"header": "task", "overflow": "fold", "style": "brl.path"},
                {"header": "detail", "overflow": "fold"},
                title="task validation")
    bad = 0
    for p in paths:
        rel = display_path(p)
        try:
            task = _load(p)
            from bedrock_rl.adapters.netherite.computer import validate_cu_example
            validate_cu_example(task)
        except Exception as e:
            bad += 1
            t.add_row(C.mark("fail"), rel, f"{type(e).__name__}: {e}")
            continue
        n = 1 + len(task.shaping) + (1 if task.fail is not None else 0)
        t.add_row(C.mark("ok"), rel,
                  f"{task.name}, {n} check{'' if n == 1 else 's'}, "
                  f"{len(task.seeds)} seed"
                  f"{'' if len(task.seeds) == 1 else 's'}")
    con = C.get_console()
    con.print(t)
    if bad:
        con.print(f"\n{bad} task file{'' if bad == 1 else 's'} will not load.",
                  style="brl.fail")
        return 1
    con.print(f"\n{len(paths)} task files load.", style="brl.ok")
    return 0


def cmd_task_render(a):
    env = package_env(a)
    argv = [sys.executable, "-m", "bedrock_rl.cli.task_render",
            task_yaml(a.task), here(a.out)]
    if a.play_oracle:
        argv.append("--play-oracle")
    return run(argv, env=env, dry=a.dry_run)


def cmd_task_snapshots(a):
    env = package_env(a)
    return run([sys.executable, "-m", "bedrock_rl.cli.task_snapshots",
                "--task", task_yaml(a.task)], env=env, dry=a.dry_run)


def cmd_trial(a):
    """Run the lightweight core harness described by one YAML file."""
    from dataclasses import replace
    from bedrock_rl.core.run import RunSpec

    spec = RunSpec.load(here(a.config))
    if a.output:
        spec = replace(spec, output=here(a.output))
    if a.check:
        spec.validate_components()
        print(f"task       {spec.task.name}")
        print(f"model      {spec.model.type}")
        print(f"harness    {spec.harness.type}")
        print(f"tools      {', '.join(item.type for item in spec.tools) or 'none'}")
        print(f"context    {', '.join(item.type for item in spec.context) or 'none'}")
        print(f"guidance   {spec.guidance_provider.type if spec.guidance_provider else 'none'}")
        print(f"probes     {', '.join(item.type for item in spec.probes) or 'none'}")
        verifier = spec.verifier or spec.task.verifier
        print(f"verifier   {verifier.type if verifier else 'none'}")
        print(f"output     {spec.output or 'not saved'}")
        return 0
    trajectory = asyncio.run(spec.execute())
    where = spec.output or "not saved"
    print(f"trial {trajectory.id} reward={trajectory.reward.total:g} -> {where}")
    return 0


# ── the help screen ──────────────────────────────────────────────────────

def _version():
    try:
        from importlib.metadata import version
        release = version("bedrock-rl")
        return release if release.startswith("v") else f"v{release}"
    except Exception:
        # a checkout with nothing installed. "unknown" cannot go stale,
        # a copied literal can.
        return "unknown"


def help_screen():
    from rich.panel import Panel
    from rich.text import Text
    con = C.get_console()
    head = Text()
    head.append("bedrock-rl", style="brl.head")
    head.append(f"  {_version()}\n", style="brl.dim")
    head.append("Minecraft agent environments, data and training",
                style="brl.dim")
    con.print(Panel(head, border_style="brl.dim", padding=(0, 2),
                    box=C.box(), expand=False))
    con.print(Text("usage: ", style="brl.dim")
              + Text(f"{PROG} <command> [options]", style="brl.cmd"))

    con.print("\nstart here", style="brl.head")
    t = C.grid("brl.cmd", "brl.dim")
    for cmd, why in START_HERE:
        t.add_row(cmd, why)
    C.print_grid(t)

    con.print("\ncommands", style="brl.head")
    t = C.grid("brl.cmd", None)
    for name, what, subs in GROUPS:
        cell = Text(what)
        if subs:
            cell.append("\n" + subs, style="brl.dim")
        t.add_row(name, cell)
    C.print_grid(t)
    con.print(Text(f"\n{PROG} <command> --help", style="brl.cmd")
              + Text(" for one command", style="brl.dim"))
    return 0


def group_screen(name, description, members):
    """The same shape as the root screen, for one group."""
    from rich.text import Text
    con = C.get_console()
    con.print(Text(f"{PROG} {name}", style="brl.head")
              + Text(f"  {description}\n", style="brl.dim"))
    t = C.grid("brl.cmd", "brl.dim")
    for key, desc in members:
        t.add_row(f"{PROG} {name} {key}", desc)
    C.print_grid(t)
    con.print(Text(f"\n{PROG} {name} <command> --help", style="brl.cmd")
              + Text(" for its flags", style="brl.dim"))
    return 0


# ── the parser ───────────────────────────────────────────────────────────

def _add_common(p):
    p.add_argument("--netherite-home", metavar="DIR",
                   help="engine checkout (default $NETHERITE_HOME or "
                        "~/netherite)")
    p.add_argument("--verl-env", metavar="DIR",
                   help="training venv (default $VERL_ENV or ~/verl-env)")
    p.add_argument("--verl-home", metavar="DIR",
                   help="verl checkout (default $VERL_HOME or ~/verl)")
    p.add_argument("-e", "--env", metavar="KEY=VALUE", action="append",
                   help="any other environment knob, repeatable")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command and stop")
    return p


def _new(sub, name, **kw):
    """One subparser, with prefix abbreviation OFF.

    argparse expands an unambiguous prefix by default, which once turned
    a trainer's `--data run.parquet` into --data-dir and pointed a data
    DIRECTORY at a file. Those two are config keys now and
    bedrock_rl/config/training.py names a near miss rather than applying it,
    which is the same rule; the flags that remain here still differ by a
    suffix on purpose, so guessing between them is not something this
    layer may do. It also keeps a mistyped flag in `rest`, where the
    module being wrapped rejects it by name.
    """
    kw.setdefault("allow_abbrev", False)
    return sub.add_parser(name, **kw)


def build_parser():
    ap = argparse.ArgumentParser(prog=PROG, add_help=True,
                                 allow_abbrev=False,
                                 description="a modular Minecraft agent-"
                                             "environment framework")
    ap.add_argument("--version", action="version", version=_version())
    sub = ap.add_subparsers(dest="group")

    p = _add_common(_new(sub,
        "setup", description="build the engine and optional training venv",
        help="build the engine and optional training venv"))
    p.add_argument("--runtime-only", action="store_true",
                   help="build the patched Minecraft engine and texture "
                        "assets, then stop before cloning verl or installing "
                        "the CUDA training stack")
    p.add_argument("--mc-jar", metavar="PATH",
                   help="Minecraft 1.11.2 jar you own. Without one the "
                        "texture atlases are generated by "
                        "tools/stub_atlases.py, which builds and runs but "
                        "does not look like Minecraft")
    p.add_argument("--texture-pack", metavar="ZIP_OR_DIR",
                   help="resource pack overlaid onto --mc-jar before the "
                        "renderer atlases are compiled")
    p.add_argument("--python", metavar="X.Y",
                   help="interpreter for the training venv (default 3.12)")
    p.add_argument("--no-flash-attn", action="store_true",
                   help="skip flash-attn, which leaves data generation and "
                        "SFT working and RL unavailable")
    p.add_argument("--install-deepspeed", action="store_true",
                   help="also install DeepSpeed, which only the SFT trainer "
                        "can use. Named apart from `trainer: sft` with a "
                        "`deepspeed:` config key, which picks a ZeRO preset, "
                        "because "
                        "bin/setup_deps.sh keeps those two ideas apart on "
                        "purpose")
    p.add_argument("--force", action="store_true",
                   help="move a checkout to the pin even when that strands "
                        "local work")
    p.set_defaults(fn=cmd_setup)

    # doctor reads and runs nothing, so it takes the two locations it
    # inspects and not the common block, which would offer --dry-run and
    # --env and then ignore both.
    p = _new(sub,
        "doctor", description="check this box and say what is broken",
        help="check this box and say what is broken")
    p.add_argument("--netherite-home", metavar="DIR",
                   help="engine checkout to inspect (default "
                        "$NETHERITE_HOME or ~/netherite)")
    p.add_argument("--verl-env", metavar="DIR",
                   help="training venv to inspect (default $VERL_ENV or "
                        "~/verl-env)")
    p.add_argument("--verl-home", metavar="DIR",
                   help="verl checkout to inspect (default $VERL_HOME or "
                        "~/verl)")
    p.add_argument("--model", metavar="MODEL",
                   help="model whose weights path to report (default "
                        "$MODEL or the registry default). On more than "
                        "one node a local path has to exist on all of "
                        "them")
    p.add_argument("--runtime-only", action="store_true",
                   help="check the task/runtime image without requiring "
                        "the optional GPU training stack")
    p.set_defaults(fn=cmd_doctor)

    p = _add_common(_new(sub,
        "smoke", description="env, task and reward sanity, no GPU",
        help="env, task and reward sanity, no GPU"))
    p.add_argument("task", nargs="?", default=DEFAULT_TASK,
                   help=TASK_HELP + " (default %(default)s)")
    p.set_defaults(fn=cmd_smoke)

    p = _new(sub, "trial",
             description="run one agent attempt at an import-path task",
             help="run one agent attempt at an import-path task")
    p.add_argument("config", help="run YAML naming the task, model, tools, "
                                  "context policies and state probes")
    p.add_argument("--output", help="override the trajectory JSON or JSONL "
                                    "path in the run file")
    p.add_argument("--check", action="store_true",
                   help="validate and show resolved component paths without "
                        "constructing them")
    p.set_defaults(fn=cmd_trial)

    # train and generate. A second optional positional selects one independent
    # preset from a manifest; it does not schedule or chain presets.
    for name, fn, what, keys, sets in (
            ("train", cmd_train,
             "run one trainer or workflow from a config file",
             "trainer or workflow, task, model, steps, and environment knobs",
             "override one key of the config, repeatable. It writes into "
             "the file's own namespace, so it can also set task, model, "
             "steps and algorithm"),
            ("generate", cmd_generate,
             "generate records through import-path components",
             "count, sampler, generator, executor, writer, and output",
             "override one dotted config key, repeatable")):
        p = _add_common(_new(sub, name, description=what, help=what))
        p.add_argument("config", nargs="?", default=None, metavar="CONFIG",
                       help=f"a YAML file carrying {keys}")
        p.add_argument("preset", nargs="?", default=None,
                       metavar=("PRESET" if name == "train" else "JOB"),
                       help=("one named training preset" if name == "train"
                             else "one named generation job"))
        p.add_argument("--set", metavar="KEY=VALUE", action="append",
                       help=sets)
        if name == "train":
            # The old interface listed its knobs in --help because each
            # was a flag. A schema nobody can enumerate is one people
            # guess at, so the table prints itself.
            p.add_argument("--knobs", metavar="TARGET", default=None,
                           help="print every config key that trainer or "
                                "workflow "
                                "reads, and the knob each becomes, "
                                "instead of running anything")
            p.add_argument("--modal", action="store_true",
                           help="launch this same config on Modal")
            p.add_argument("--nodes", type=int, default=1,
                           help="Modal nodes (default 1; multi-node is RL only)")
            p.add_argument("--gpu", default=None, metavar="TYPE[:N]",
                           help="Modal GPU allocation (default H100; a "
                                "multi-node launch expands it to a full host)")
            p.add_argument("--volume", default="bedrock-rl-runs",
                           help="Modal v2 Volume for weights and run outputs")
            p.add_argument("--secret", action="append", metavar="NAME",
                           help="Modal Secret to inject, repeatable")
            p.add_argument("--timeout", type=int, default=86400,
                           help="Modal job timeout in seconds (default 86400)")
            p.add_argument("--rdma", action="store_true",
                           help="request Modal RDMA on a multi-node cluster "
                                "(requires workspace support)")
            p.add_argument("--modal-smoke", action="store_true",
                           help="validate CUDA, NCCL and Ray across the "
                                "requested Modal nodes without training")
            p.add_argument("--detach", action="store_true",
                           help="let the Modal app survive a local disconnect")
            p.add_argument("--modal-run", default=None, metavar="ID",
                           help="stable Modal output id; reusing it resumes a "
                                "supported trainer from durable checkpoints")
        else:
            p.add_argument("--check", action="store_true",
                           help="resolve component paths without running")
            p.add_argument("--modal-scout", action="store_true",
                           help="headlessly prefilter exact scripted worlds "
                                "on a detached Modal CPU fleet")
            p.add_argument("--scout-worlds", type=int, default=None,
                           help="accepted world groups to retain (default: "
                                "the config's worlds)")
            p.add_argument("--scout-attempts", type=int, default=None,
                           help="maximum candidate worlds (default: the "
                                "config's max_world_attempts)")
            p.add_argument("--scout-containers", type=int, default=1000,
                           help="Modal CPU container ceiling (default 1000)")
            p.add_argument("--scout-results",
                           default="bedrock-sdg-scout-v1",
                           help="persistent Modal Dict for progress/results")
            p.add_argument("--scout-run", default=None,
                           help="durable result key prefix (default: unique)")
        p.set_defaults(fn=fn)

    # eval
    p = _new(sub, "eval", description="score a checkpoint on real "
                                           "episodes", help="score a "
                                                            "checkpoint")
    esub = p.add_subparsers(dest="eval_cmd")
    q = _add_common(_new(esub,
        "trials", description="run multi-turn trials with vLLM and score "
                              "them through the task environment",
        help="run trials against a checkpoint"))
    q.add_argument("--model", required=True,
                   help="checkpoint path or HuggingFace id")
    q.add_argument("task", help=TASK_HELP)
    q.add_argument("--data", required=True,
                   help="explicit frozen prompt parquet; evaluation never "
                        "defaults to training rows")
    q.add_argument("-n", type=int, default=None, help="prompt rows to use")
    q.add_argument("--samples", type=int, default=None,
                   help="trials per prompt row")
    q.add_argument("--hint", default=None,
                   choices=["none", "coarse", "procedure"],
                   help="sample with the task's hint in a system message")
    q.add_argument("--lora", default=None,
                   help="a LoRA adapter to serve on top of --model; "
                        "inferred when --model is itself an adapter or a "
                        "merged RL checkpoint")
    q.add_argument("--blank-frames", action="store_true",
                   help="grey out every frame; a policy that keeps its "
                        "score is not reading the screen")
    q.add_argument("--keep-latest-images", type=int, default=None,
                   help="match the checkpoint's training view: retain the "
                        "newest N images; 0 keeps all")
    q.epilog = FORWARDS
    q.set_defaults(fn=cmd_eval_trials, passthrough=True)
    q = _add_common(_new(
        esub, "summarize",
        description="build a paired checkpoint summary and SVG curve from "
                    "trial records",
        help="compare paired trial-record files"))
    q.add_argument(
        "--point", nargs=3, action="append", required=True,
        metavar=("LABEL", "STEP", "RECORDS"),
        help="curve point; repeat for the base and each checkpoint")
    q.add_argument("--base", default=None,
                   help="baseline label; defaults to the first point")
    q.add_argument("--summary", required=True,
                   help="output summary JSON")
    q.add_argument("--svg", required=True, help="output SVG curve")
    q.add_argument("--title", default=None, help="curve title")
    q.add_argument("--training-metrics", default=None,
                   help="optional trainer JSONL to overlay")
    q.add_argument("--success-max", type=float, default=None,
                   metavar="PERCENT",
                   help="top of the success-rate axis (default: 100)")
    q.add_argument(
        "--heldout-summary", nargs=2, action="append", default=None,
        metavar=("LABEL", "SUMMARY"),
        help="additional held-out summary marker; repeat as needed")
    q.set_defaults(fn=cmd_eval_summarize)

    # models
    p = _new(sub, "models", description="the supported-model registry",
                       help="the supported-model registry")
    msub = p.add_subparsers(dest="models_cmd")
    q = _new(msub, "list", description="every supported family",
                        help="every supported family")
    q.add_argument("--markdown", action="store_true",
                   help="bedrock_rl/models.py's own markdown table "
                        "instead of a rendered one")
    q.set_defaults(fn=cmd_models_list)
    q = _new(msub, "resolve", description="what one name resolves to",
                        help="what one name resolves to")
    q.add_argument("model", help="family alias, size alias, HuggingFace id "
                                 "or local checkpoint path")
    q.add_argument("--json", action="store_true",
                   help="the whole registry entry, as json")
    q.set_defaults(fn=cmd_models_resolve)
    q = _add_common(_new(msub,
        "check", description="verify a registry entry against the real "
                             "model, no GPU needed",
        help="verify a registry entry against the real model"))
    q.add_argument("model", help="family alias, size alias, HuggingFace "
                                 "id or local checkpoint path")
    q.epilog = FORWARDS
    q.set_defaults(fn=cmd_models_check, passthrough=True)

    # task
    p = _new(sub, "task", description="task files and what they check",
                       help="task files and what they check")
    ksub = p.add_subparsers(dest="task_cmd")
    q = _new(ksub, "list", description="every included example task",
                        help="every included example task")
    q.set_defaults(fn=cmd_task_list)
    q = _new(ksub, "show", description="one task, its budgets and its "
                                            "checks",
                        help="one task, its budgets and its checks")
    q.add_argument("task", nargs="?", default=DEFAULT_TASK,
                   help=TASK_HELP + " (default %(default)s)")
    q.set_defaults(fn=cmd_task_show)
    q = _new(ksub, "validate", description="load task files and report "
                                                "what will not load",
                        help="load task files and report what will not load")
    q.add_argument("tasks", nargs="*",
                   help="example names or task paths (default every "
                        "included example task)")
    q.set_defaults(fn=cmd_task_validate)
    q = _add_common(_new(ksub,
        "render", description="write the frame and the prompt the model "
                              "would see",
        help="write the frame and the prompt the model would see"))
    q.add_argument("task", nargs="?", default=DEFAULT_TASK,
                   help=TASK_HELP + " (default %(default)s)")
    q.add_argument("out", nargs="?", default="/tmp/bedrock_sample",
                   help="directory to write the frame and prompt into")
    q.add_argument("--play-oracle", action="store_true",
                   help="also write every frame the scripted oracle sees")
    q.set_defaults(fn=cmd_task_render)
    q = _add_common(_new(ksub,
        "snapshots", description="pre-build a task's per-seed start states",
        help="pre-build a task's per-seed start states"))
    q.add_argument("task", nargs="?", default=DEFAULT_TASK,
                   help=TASK_HELP + " (default %(default)s)")
    q.set_defaults(fn=cmd_task_snapshots)

    # The groups that have subcommands of their own, so the help screens
    # can list them without reaching into argparse's internals. `train`
    # and `generate` are not here: their argument is a file, not a verb.
    return ap, {"eval": esub, "models": msub, "task": ksub}


# Printed under the flags of every command that hands the rest on.
FORWARDS = ("anything after -- goes to the command underneath, "
            "unchanged")

GROUP_NAMES = [g[0] for g in GROUPS]
HELPS = ("-h", "--help", "help")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap, groups = build_parser()

    # Flags for the wrapped module go after a literal `--`, and nothing
    # before it is guessed at. Letting argparse hand back what it did
    # not recognise looked like the friendlier design and was not: an
    # unknown OPTION lands in the leftovers but its VALUE stays a free
    # positional, so `eval trials --model M --temperature 0.7` fed 0.7
    # to the TASK argument and forwarded a --temperature with nothing
    # after it. `--` is the convention every tool that forwards
    # arguments already uses, and it cannot be misread.
    extras = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extras = argv[:cut], argv[cut + 1:]

    # The styled screens are for the two places a person is looking for
    # orientation rather than a flag: the bare command and a group with
    # nothing after it. Everything else keeps argparse's help, which is
    # the one every other python tool prints.
    if not argv or argv[0] in HELPS:
        return help_screen()
    if argv[0] in groups and (len(argv) == 1 or argv[1] in HELPS):
        what = {g[0]: g[1] for g in GROUPS}[argv[0]]
        members = [(k, p.description or "")
                   for k, p in groups[argv[0]].choices.items()]
        return group_screen(argv[0], what, members)

    if argv[0] not in GROUP_NAMES and argv[0] != "--version":
        # A flag before any command is not a command, and saying "no
        # command named '-x'" is closer to the truth than letting
        # argparse drop it and then reporting a missing subcommand.
        what = ("flag" if argv[0].startswith("-") else "command")
        C.error(f"no {what} {argv[0]!r} before a command",
                "commands are " + ", ".join(GROUP_NAMES) + f"\n{PROG} --help")
        return 2

    # Parsing is inside the try because argparse reports `--help` and a
    # bad argument by RAISING SystemExit, and main() is a function that
    # returns an exit code rather than one that leaves the process.
    try:
        a, rest = ap.parse_known_args(argv)
        if not hasattr(a, "fn"):
            C.error(f"{PROG} {argv[0]} needs a subcommand",
                    f"{PROG} {argv[0]} --help")
            return 2
        forwards = getattr(a, "passthrough", False)
        if rest:
            # `bedrock train rl equip_pickaxe_grpo qwen3-vl -n 30` is
            # the shape people actually have in shell history, and every
            # one of them lands HERE rather than on the refusal below,
            # because the trailing words are what argparse cannot place.
            # So the old form is recognised before the generic message.
            if (argv[0] in ("train", "generate") and len(argv) > 1
                    and argv[1] in config.TARGETS
                    and not os.path.isfile(here(argv[1]))):
                config_path(argv[1], argv[0])
            # Reconstructing the corrected line would mean guessing which
            # of the remaining words were this flag's value, which is the
            # guess that caused the problem. Name the convention instead.
            C.error("unrecognised arguments, " + " ".join(rest),
                    (f"anything after -- goes to the command "
                     f"{' '.join(argv[:2])} wraps\n"
                     f"  {PROG} {' '.join(argv[:2])} <flags> -- {rest[0]} ...")
                    if forwards else f"{PROG} {argv[0]} --help")
            return 2
        if extras and not forwards:
            C.error(f"{PROG} {argv[0]} forwards nothing, so it takes no "
                    "arguments after --")
            return 2
        if forwards:
            return a.fn(a, extras)
        return a.fn(a)
    except KeyboardInterrupt:
        return 130
    except SystemExit as e:
        # `raise SystemExit("...")` is how this repo's modules report a
        # user error. Its first line is the message and the rest is the
        # fix, which is the shape C.error prints.
        if isinstance(e.code, str):
            lines = e.code.splitlines()
            C.error(lines[0], "\n".join(lines[1:]))
            return 1
        return e.code or 0
    except Exception as e:
        # A traceback out of a CLI is a bug report nobody asked for. The
        # message and the type are what a user can act on.
        if os.environ.get("BEDROCK_TRACEBACK"):
            raise
        C.error(f"{type(e).__name__}: {e}",
                "BEDROCK_TRACEBACK=1 for the traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
