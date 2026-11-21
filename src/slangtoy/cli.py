import argparse
import re
import runpy
from contextlib import ExitStack
from pathlib import Path


TEMPLATES = Path(__file__).with_name("templates")
SKETCH_NAME = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")


def create_sketch(directory: Path, name: str) -> None:
    title = name.replace("-", " ").replace("_", " ").title()
    python_template = (TEMPLATES / "sketch.py.txt").read_text(encoding="utf-8")
    shader_template = (TEMPLATES / "sketch.slang").read_text(encoding="utf-8")
    sources = {
        directory / f"{name}.py": python_template.replace("__TITLE__", repr(title)),
        directory / f"{name}.slang": shader_template,
    }
    with ExitStack() as rollback:
        for path, source in sources.items():
            with path.open("x", encoding="utf-8") as output:
                rollback.callback(path.unlink)
                output.write(source)
        rollback.pop_all()
    print(f"Created sketches/{name}.py and sketches/{name}.slang")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sketch",
        usage="%(prog)s NAME | %(prog)s --list | %(prog)s new NAME",
        description="Run or create a SlangToy sketch.",
    )
    parser.add_argument("sketch", nargs="?", help="sketch name, or 'new' to create one")
    parser.add_argument("name", nargs="?", help="name of a new sketch")
    parser.add_argument("--list", action="store_true", help="list available sketches")
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    directory = next(
        (
            root / "sketches"
            for root in (cwd, *cwd.parents)
            if (root / "pyproject.toml").is_file() and (root / "sketches").is_dir()
        ),
        None,
    )
    if directory is None:
        parser.error("could not find the SlangToy project's sketches directory")

    if args.sketch is None or args.list:
        if args.sketch is not None:
            parser.error("--list takes no sketch name")
        for path in sorted(directory.glob("*.py")):
            if path.is_file() and path.with_suffix(".slang").is_file():
                print(path.stem)
        return

    creating = args.sketch == "new"
    name = args.name if creating else args.sketch
    if not creating and args.name is not None:
        parser.error("expected a single sketch name")
    if name is None or name == "new" or not SKETCH_NAME.fullmatch(name):
        parser.error("use a sketch name like 'water' or 'my_sketch'")

    if creating:
        try:
            create_sketch(directory, name)
        except FileExistsError as error:
            parser.error(f"{Path(error.filename).name} already exists")
        return

    path = directory / f"{name}.py"
    if not path.is_file() or not path.with_suffix(".slang").is_file():
        parser.error(f"sketch {name!r} needs both {name}.py and {name}.slang")
    runpy.run_path(str(path), run_name="__main__")


def square_world() -> None:
    main(["square_world"])
