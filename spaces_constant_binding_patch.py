"""Bind AoTI constants that `torch.export` lifted anonymously.

`spaces.zero.torch.aoti.LazyAOTIModel` binds a package's constants by intersecting the module's `state_dict()` with
`compiled_model.get_constant_fqns()`, and keeps whatever it cannot match. `torch.export` only gives a lifted tensor a
real FQN when it was a registered parameter or buffer; anything else is named `_tensor_constant<N>`, which no
`state_dict()` can contain, so the compiled model runs against constants nobody set — a SIGSEGV rather than an error.

`write_constant_aliases` records the real names on the compile side; `apply_spaces_constant_binding_patch` uses that
sidecar on the load side, falls back to matching by dtype+shape, and raises if the binding is still not total.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import torch

ALIASES_FILENAME = "constant_aliases.json"

_DTYPES = {
    "float32": torch.float32, "float64": torch.float64, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2, "float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "float8_e5m2fnuz": torch.float8_e5m2fnuz, "int8": torch.int8, "uint8": torch.uint8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64, "bool": torch.bool,
}


# --------------------------------------------------------------------------- compile side


def register_loose_tensors(module: torch.nn.Module, prefix: str = "") -> list[str]:
    """Re-register plain tensor attributes as buffers so `torch.export` gives them real FQNs.

    Run on the shallow clone, right before `torch.export.export`. Returns the names it re-registered.
    """
    registered = []
    for name, value in list(vars(module).items()):
        if not isinstance(value, torch.Tensor) or name.startswith("_"):
            continue
        if name in module._parameters or name in module._buffers:
            continue
        object.__delattr__(module, name)
        module.register_buffer(name, value, persistent=True)
        registered.append(f"{prefix}{name}")
    for child_name, child in module.named_children():
        registered += register_loose_tensors(child, f"{prefix}{child_name}.")
    return registered


def constant_aliases_from_exported_program(exported_program) -> dict[str, str]:
    """`{'_tensor_constant<N>': '<real dotted fqn>'}` for every anonymously lifted constant.

    AOT Inductor numbers its slots in the order the `CONSTANT_TENSOR` inputs appear in the graph signature, which
    still carries each one's real FQN.
    """
    targets = [
        spec.target
        for spec in exported_program.graph_signature.input_specs
        if spec.kind.name == "CONSTANT_TENSOR"
    ]
    return {f"_tensor_constant{index}": target for index, target in enumerate(targets)}


def write_constant_aliases(package_dir, exported_program, submodule: str | None = None) -> Path | None:
    """Drop the alias sidecar next to the `package.pt2` `aoti_compile_and_save` just wrote."""
    aliases = constant_aliases_from_exported_program(exported_program)
    if not aliases:
        return None
    subdir = Path(package_dir) / ("submodules/" + submodule if submodule else "root")
    path = subdir / ALIASES_FILENAME
    path.write_text(json.dumps(aliases, indent=2))
    return path


# --------------------------------------------------------------------------- load side


def _package_constants_info(archive_file) -> list[dict]:
    """Read `constants_info_` (dtype, shape, in slot order) out of a `.pt2`'s wrapper source."""
    if isinstance(archive_file, (str, Path)):
        handle: object = str(archive_file)
    else:
        position = archive_file.tell()
        archive_file.seek(0)
        handle = io.BytesIO(archive_file.read())
        archive_file.seek(position)
    with zipfile.ZipFile(handle) as archive:  # pyright: ignore[reportArgumentType]
        names = [n for n in archive.namelist() if n.endswith(".wrapper.cpp")]
        if not names:
            return []
        source = archive.read(names[0]).decode()
    info: dict[int, dict] = {}
    for match in re.finditer(r"constants_info_\[(\d+)\]\.(\w+) = ([^;]+);", source):
        index, field, value = int(match.group(1)), match.group(2), match.group(3).strip()
        entry = info.setdefault(index, {})
        if field == "dtype":
            entry["dtype"] = _DTYPES.get(value.replace("cached_torch_dtype_", ""))
        elif field == "shape":
            entry["shape"] = tuple(int(x) for x in re.findall(r"-?\d+", value))
        elif field in ("name", "original_fqn"):
            entry[field] = value.strip('"')
    return [info[index] for index in sorted(info)]


def resolve_constant_map(
    archive_file,
    constant_fqns,
    weights: dict[str, torch.Tensor],
    aliases=None,
    allow_shape_fallback: bool = False,
):
    """Map every compiled constant FQN onto one of `weights`, or report what is left over."""
    constant_map = {name: weights[name] for name in constant_fqns if name in weights}
    missing = [name for name in constant_fqns if name not in constant_map]
    if not missing:
        return constant_map, []

    aliases = aliases or {}
    for name in list(missing):
        target = aliases.get(name)
        if target is not None and target in weights:
            constant_map[name] = weights[target]
            missing.remove(name)
    if not missing or not allow_shape_fallback:
        return constant_map, missing

    # Match by dtype+shape against the unclaimed `state_dict()` entries, preserving each side's own order inside a
    # (dtype, shape) group. `get_constant_fqns()` returns slots lexicographically (`_tensor_constant10` before
    # `_tensor_constant2`), so the package's own `constants_info_` index is the only correct order to walk them in.
    info = _package_constants_info(archive_file)
    by_name = {entry.get("name"): entry for entry in info}
    slot_index = {entry.get("name"): index for index, entry in enumerate(info)}
    taken = {id(tensor) for tensor in constant_map.values()}
    buckets: dict[tuple, list[torch.Tensor]] = {}
    for tensor in weights.values():
        if id(tensor) not in taken:
            buckets.setdefault((tensor.dtype, tuple(tensor.shape)), []).append(tensor)
    for name in sorted(list(missing), key=lambda n: slot_index.get(n, 1 << 30)):
        entry = by_name.get(name)
        if entry is None or entry.get("dtype") is None:
            continue
        bucket = buckets.get((entry["dtype"], entry["shape"]))
        if bucket:
            constant_map[name] = bucket.pop(0)
            missing.remove(name)
    return constant_map, missing


def apply_spaces_constant_binding_patch(strict: bool = True, allow_shape_fallback: bool = False):
    """Make `spaces`' AoTI loader bind anonymous constants, and fail loudly if it still cannot.

    Call once, before any `spaces.aoti_*` loading. Idempotent.
    """
    from spaces.zero.torch import aoti as spaces_aoti

    if getattr(spaces_aoti.LazyAOTIModel, "_constant_binding_patched", False):
        return

    original_call = spaces_aoti.LazyAOTIModel.__call__

    def patched_call(self, weights, check_full_update, *args, **kwargs):
        compiled_model = self.compiled_model.get()
        if compiled_model is None:
            with spaces_aoti._register_aoti_cleanup():
                compiled_model = torch._inductor.aoti_load_package(self.archive_file)
            self.compiled_model.set(compiled_model)
        loaded = self.loaded_weights.get()
        if loaded is None or loaded is not weights:
            fqns = compiled_model.get_constant_fqns()
            aliases = getattr(self, "_constant_aliases", None)
            if aliases is None:
                aliases = {}
                if isinstance(self.archive_file, (str, Path)):
                    sidecar = Path(self.archive_file).with_name(ALIASES_FILENAME)
                    if sidecar.is_file():
                        aliases = json.loads(sidecar.read_text())
                self._constant_aliases = aliases
            constant_map, missing = resolve_constant_map(
                self.archive_file, fqns, weights, aliases, allow_shape_fallback
            )
            if missing and strict:
                raise RuntimeError(
                    f"{len(missing)} of {len(fqns)} AoTI constants could not be bound to the module's "
                    f"state_dict: {missing[:8]}. Anonymous `_tensor_constant*` names mean the export saw "
                    f"plain tensor attributes rather than registered parameters or buffers. Register them "
                    f"(or write a {ALIASES_FILENAME} sidecar at compile time) — binding them partially "
                    f"would leave the compiled model dereferencing unset constants."
                )
            compiled_model.load_constants(
                constant_map, check_full_update=check_full_update and not missing, user_managed=True
            )
            self.loaded_weights.set(weights)
        return compiled_model(*args, **kwargs)

    spaces_aoti.LazyAOTIModel.__call__ = patched_call
    spaces_aoti.LazyAOTIModel._constant_binding_patched = True
    spaces_aoti.LazyAOTIModel._original_call = original_call
