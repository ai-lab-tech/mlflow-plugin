"""Observe prediction without reloading weights or freezing training."""

import dis
import inspect
import sys
from contextlib import contextmanager
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import resolve_name
from pathlib import Path
from types import (
    BuiltinFunctionType,
    FunctionType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
)

import numpy as np
from packaging.utils import canonicalize_name

_MISSING = object()
_EXCLUDED = {"mlflow", "mlflow_triton"}
_LOCAL_LOADS = {"LOAD_DEREF", "LOAD_CLASSDEREF"}


def _module_dict(module):
    # Bypass ModuleType subclasses' __getattribute__ / lazy __getattr__ hooks.
    return ModuleType.__dict__["__dict__"].__get__(module)


def _static_attribute(value, name):
    if value is _MISSING:
        return _MISSING
    if issubclass(type(value), ModuleType):
        return _module_dict(value).get(name, _MISSING)
    return inspect.getattr_static(value, name, _MISSING)


class PredictionDependencies:
    """Observe executed module references on the prediction thread.

    Capture is sample-based: unexecuted paths and work delegated to other
    threads/processes require explicit requirements. No weights are copied.
    """

    def __init__(self):
        # The copied backend imports both, regardless of the model flavor.
        self.modules = {"numpy", "pandas"}
        self._instructions = {}
        self._ufunc_modules = {}

    def _record(self, name):
        if type(name) is str and name:
            root = name.split(".", 1)[0]
            if root not in _EXCLUDED and root not in sys.stdlib_module_names:
                self.modules.add(name)

    def _record_value(self, value):
        if value is _MISSING:
            return
        if issubclass(type(value), ModuleType):
            self._record(_module_dict(value).get("__name__"))
        elif issubclass(type(value), type):
            self._record(type.__dict__["__module__"].__get__(value))
        elif type(value) in (FunctionType, BuiltinFunctionType):
            self._record(value.__module__)
            if type(value) is BuiltinFunctionType and value.__self__ is not None:
                self._record_value(type(value.__self__))
        elif type(value) is MethodType:
            self._record_value(value.__func__)
            self._record_value(type(value.__self__))
        elif type(value) is np.ufunc:
            # scipy.special ufuncs lack __module__ and emit no Python call
            # events. Find their exports rather than just their numpy type.
            if value not in self._ufunc_modules:
                exporters = {
                    name: _module_dict(module)
                    for name, module in list(sys.modules.items())
                    if issubclass(type(module), ModuleType)
                    and _module_dict(module).get(value.__name__) is value
                }
                native_exporters = {
                    name
                    for name, attributes in exporters.items()
                    if type(attributes.get("__file__")) is str
                    and attributes["__file__"].endswith(tuple(EXTENSION_SUFFIXES))
                }
                # Python modules may merely re-export the same function (e.g.
                # an already-imported training helper). Prefer its native owner.
                self._ufunc_modules[value] = native_exporters or set(exporters)
            for name in self._ufunc_modules[value]:
                self._record(name)
        else:
            self._record(type.__dict__["__module__"].__get__(type(value)))
            if callable(value):
                # Static lookup never evaluates user properties or __getattr__.
                module = _static_attribute(value, "__module__")
                if type(module) is MemberDescriptorType:
                    # Cython functions store __module__ in a native member slot.
                    try:
                        module = module.__get__(value)
                    except (AttributeError, TypeError):
                        module = None
                self._record(module)

    def _observe_opcode(self, frame, instruction, previous_value):
        name = instruction.argval
        operation = instruction.opname
        if operation.startswith("LOAD_FAST"):
            # Python 3.13+ may combine two local loads in one instruction;
            # Python 3.14 also has borrowed-reference variants.
            for local_name in name if type(name) is tuple else (name,):
                value = frame.f_locals.get(local_name, _MISSING)
                self._record_value(value)
            return value
        elif operation in _LOCAL_LOADS:
            value = frame.f_locals.get(name, _MISSING)
        elif operation == "LOAD_GLOBAL":
            value = frame.f_globals.get(name, frame.f_builtins.get(name, _MISSING))
        elif operation == "LOAD_NAME":
            value = frame.f_locals.get(
                name, frame.f_globals.get(name, frame.f_builtins.get(name, _MISSING))
            )
        elif operation in {"LOAD_ATTR", "LOAD_METHOD"}:
            value = _static_attribute(previous_value, name)
        else:
            return _MISSING
        self._record_value(value)
        return value

    @staticmethod
    def _import_name(frame, instruction, instructions):
        offsets = list(instructions)
        index = offsets.index(instruction.offset)
        level = instructions[offsets[index - 2]].argval if index >= 2 else 0
        if type(level) is int and level:
            package = frame.f_globals.get("__package__")
            if type(package) is str and package:
                try:
                    return resolve_name("." * level + instruction.argval, package)
                except ImportError:
                    # Let the actual import raise; failed optional imports must
                    # not disable tracing by raising inside the trace callback.
                    return None
        return instruction.argval

    @contextmanager
    def capture(self):
        previous_trace = sys.gettrace()
        active = True

        def trace(frame, event, arg):
            prior = previous_trace(frame, event, arg) if previous_trace else None
            module = frame.f_globals.get("__name__", "")
            root = module.split(".", 1)[0]
            dynamic_import = (
                module == "importlib" and frame.f_code.co_name == "import_module"
            )
            if (
                not active
                or root in _EXCLUDED
                or (root in sys.stdlib_module_names and not dynamic_import)
            ):
                return prior
            self._record(module)
            code = frame.f_code
            if code not in self._instructions:
                self._instructions[code] = {
                    instruction.offset: instruction
                    for instruction in dis.get_instructions(code)
                }
            instructions = self._instructions[code]
            prior_opcodes = frame.f_trace_opcodes
            prior_lines = frame.f_trace_lines
            frame.f_trace_opcodes = True
            # Some interpreter versions suppress opcode events with line
            # tracing disabled. Keep it enabled, but don't forward unwanted
            # line/opcode events to an existing debugger or coverage hook.
            frame.f_trace_lines = True
            previous_value = _MISSING
            pending_import = None
            pending_local = None
            imported_module = _MISSING

            def observe(frame, event, arg):
                nonlocal prior, previous_value, pending_import, imported_module
                nonlocal pending_local
                if not active:
                    frame.f_trace_opcodes = prior_opcodes
                    frame.f_trace_lines = prior_lines
                    frame.f_trace = prior
                    return prior(frame, event, arg) if prior else None
                if (
                    prior
                    and (event != "opcode" or prior_opcodes)
                    and (event != "line" or prior_lines)
                ):
                    replacement = prior(frame, event, arg)
                    if replacement is not None:
                        prior = replacement
                # Re-apply after the local trace is installed. In Python 3.14
                # enabling this only in the global call hook is insufficient.
                frame.f_trace_opcodes = True
                if event == "opcode":
                    instruction = instructions.get(frame.f_lasti)
                    if pending_local is not None:
                        previous_value = frame.f_locals.get(pending_local, _MISSING)
                        self._record_value(previous_value)
                        pending_local = None
                    if pending_import is not None:
                        imported_module = sys.modules.get(pending_import, _MISSING)
                        self._record_value(imported_module)
                        pending_import = None
                    if instruction is not None and instruction.opname == "IMPORT_NAME":
                        pending_import = self._import_name(
                            frame, instruction, instructions
                        )
                    elif (
                        instruction is not None and instruction.opname == "IMPORT_FROM"
                    ):
                        self._record_value(
                            _static_attribute(imported_module, instruction.argval)
                        )
                    elif instruction is not None and instruction.opname == "POP_TOP":
                        imported_module = _MISSING
                    elif (
                        instruction is not None
                        and instruction.opname == "STORE_FAST_LOAD_FAST"
                    ):
                        pending_local = instruction.argval[-1]
                    previous_value = (
                        self._observe_opcode(frame, instruction, previous_value)
                        if instruction is not None
                        else _MISSING
                    )
                elif event in {"return", "exception"}:
                    # Do not retain tensors through completed/suspended frames.
                    previous_value = _MISSING
                    imported_module = _MISSING
                    pending_import = None
                    pending_local = None
                    if event == "return":
                        if dynamic_import:
                            self._record_value(arg)
                        # Restore suspended generators as well as finished calls.
                        frame.f_trace_opcodes = prior_opcodes
                        frame.f_trace_lines = prior_lines
                        frame.f_trace = prior
                        return prior
                return observe

            return observe

        # Python 3.12 requires an opcode-enabled frame before settrace().
        # https://docs.python.org/3.12/library/sys.html#sys.settrace
        anchor = sys._getframe()
        anchor_opcodes = anchor.f_trace_opcodes
        if sys.version_info[:2] == (3, 12):
            anchor.f_trace_opcodes = True
        try:
            sys.settrace(trace)
            yield
        finally:
            active = False
            sys.settrace(previous_trace)
            anchor.f_trace_opcodes = anchor_opcodes
            del anchor
            self._instructions.clear()
            self._ufunc_modules.clear()

    @staticmethod
    def _owners(module_name, candidates):
        if len(candidates) <= 1:
            return candidates
        module = sys.modules.get(module_name)
        filename = (
            _module_dict(module).get("__file__")
            if issubclass(type(module), ModuleType)
            else None
        )
        if not filename:
            # A namespace (e.g. google) does not identify a distribution.
            # Executed submodules and accessed attributes identify its owner.
            return set()
        origin = Path(filename).resolve()
        owners = set()
        for package in candidates:
            distribution = metadata.distribution(package)
            if any(
                path.name == origin.name
                and Path(distribution.locate_file(path)).resolve() == origin
                for path in distribution.files or ()
            ):
                owners.add(package)
        if len(owners) != 1:
            raise ValueError(
                f"Cannot identify the installed distribution for {module_name!r}; "
                "provide explicit pip_requirements or conda_env"
            )
        return owners

    def requirements(self):
        """Pin distributions, resolving shared namespaces by file owner."""
        providers = metadata.packages_distributions()
        packages = set()
        for module in self.modules:
            root = module.split(".", 1)[0]
            if root in _EXCLUDED:
                continue
            candidates = {
                canonicalize_name(package) for package in providers.get(root, ())
            }
            packages.update(self._owners(module, candidates))
        packages -= {"mlflow", "mlflow-skinny", "mlflow-triton"}
        return [
            f"{package}=={metadata.version(package)}" for package in sorted(packages)
        ]
