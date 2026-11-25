"""Install project-linked Blender add-on entry points into the given add-ons directory."""
import argparse
import ast
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("addons", type=Path, help="Blender scripts/addons directory")
    args = parser.parse_args()
    sources = Path(__file__).resolve().parent / "tools"
    loaders = {}
    for name in ("sam_ellen_bridge", "mocap_v2", "motion_tweak"):
        source = sources / (name + ".py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        meta = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "bl_info" for t in n.targets))
        loaders[args.addons / source.name] = (
            "# blender-v2 project loader\n"
            f"bl_info = {meta!r}\n"
            "from pathlib import Path\nimport sys\n"
            f"__file__ = {str(source)!r}\n"
            "sys.path.insert(0, str(Path(__file__).parent))\n"
            "exec(compile(Path(__file__).read_text(encoding='utf-8'), __file__, 'exec'), globals())\n"
        )
    for path, content in loaders.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            parser.error(f"Existing add-on at {path}. Select a separate Blender scripts directory.")
    args.addons.mkdir(parents=True, exist_ok=True)
    for path, content in loaders.items():
        path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
