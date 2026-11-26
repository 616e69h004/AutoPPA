"""
AutoPPA — Closed-Loop Engineering System (Fixed & Enhanced)
============================================================
Advanced LLM-driven optimizer with closed-loop verification,
deterministic fixes, and strong functional validation.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from math import isfinite
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langgraph.graph import END, START, StateGraph

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
ERROR_MEMORY_FILE = "autoppa_error_memory.json"

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN,
)

PARSE_SYSTEM = """You are an ngspice output parser. Extract simulation results from the log.

Output JSON format:
{
  "sim_ok": true/false,
  "delay_ps": <delay in picoseconds or null>,
  "power_uw": <power in microwatts or null>,
  "area_um2": <area in square microns or null>,
  "notes": "<any relevant observations>"
}

Look for:
- .measure results showing delay_val, avg_pwr
- Successful completion messages
- Error messages indicating failure
"""

SAFE_OPTIMIZATION_RULES = """
STRICT OPTIMIZATION CONSTRAINTS (MANDATORY):
1) Preserve circuit functionality and I/O behavior.
2) Never create empty .subckt blocks.
3) Never remove required logic devices or break signal paths.
4) Allowed changes only:
   - transistor sizing (W/L scaling)
   - safe reduction of redundant passive load
   - buffer insertion for high fanout
   - minor topology cleanup that does NOT alter logic function
5) Forbidden changes:
   - logic removal
   - random rewiring
   - functionality-changing topology edits
6) Every output must remain driven and connected to a valid input-to-output path.
"""


class AutoPPA(TypedDict, total=False):
    # inputs
    netlist_raw: str
    working_netlist: str
    circuit_name: str
    circuit_type: str
    prompt: str
    brief_description: str

    # UI parameters
    opt_objective: str
    max_iterations: int
    analysis_type: str

    # per-iteration LLM outputs
    proposed_netlist: str
    testbench: str
    reasoning: str
    targets: dict
    missing_models: list

    # structural analysis
    netlist_valid: bool
    circuit_structure: dict
    connectivity_map: dict
    fanout_analysis: dict
    load_analysis: dict
    timing_paths: list

    # functional validation
    functional_valid: bool
    functional_errors: list
    empty_subckts: list
    floating_outputs: list
    missing_nodes: list
    path_exists: bool

    # electrical validation
    electrical_valid: bool
    signal_integrity: dict
    power_delivery: dict
    timing_analysis: dict

    # measurement validation
    measurement_valid: bool
    constraint_status: dict

    # simulation
    ngspice_log: str
    sim_failed: bool
    sim_error_pattern: str

    # metrics
    delay_ps: float
    power_uw: float
    area_um2: float
    cost: float
    constraints_met: bool
    met_details: dict

    # baseline tracking
    baseline_delay: float
    baseline_power: float
    baseline_area: float
    baseline_netlist: str

    # convergence tracking
    iteration: int
    history: list
    pareto_front: list
    convergence_score: float
    error_memory: list

    # best-so-far
    best_delay: float
    best_power: float
    best_area: float
    best_cost: float
    best_netlist: str
    best_iteration: int

    # acceptance
    accepted: bool
    reject_reason: str

    # fix tracking
    applied_fixes: list
    fix_history: list
    design_quality: dict

    # error analysis
    error_analysis: dict
    continue_loop: bool
    continue_reason: str
    combined_circuit: str
    fix_attempts: int
    area_estimate_um2: float
    area_estimate_method: str
    area_is_physical: bool
    llm_error: str
    llm_status: str
    measurement_error: str
    functional_validation_level: str
    validation_warnings: list


DEVICE_RE = re.compile(r"^[RCLDVIQMBX]\w*", re.IGNORECASE)
MOS_RE = re.compile(r"^M\w*", re.IGNORECASE)
SUBCKT_RE = re.compile(r"(?i)^\.subckt\s+(\S+)(.*)$")
ENDS_RE = re.compile(r"(?i)^\.ends\b")


def _extract_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    if not candidates:
        braces = re.findall(r"\{.*\}", text, flags=re.DOTALL)
        candidates.extend(braces)
    for raw in candidates:
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue
    return None


def _load_error_memory() -> list:
    path = Path(ERROR_MEMORY_FILE)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(x) for x in data if str(x).strip()] if isinstance(data, list) else []
    except Exception:
        return []


def _save_error_memory(items: list) -> None:
    try:
        Path(ERROR_MEMORY_FILE).write_text(json.dumps(items[-20:], indent=2), encoding="utf-8")
    except Exception:
        pass


def _decode_escaped_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _strip_comments(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        semicolon = line.find(";")
        if semicolon >= 0:
            line = line[:semicolon].rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _sanitize_spice_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Drop non-printable ASCII and replace high bytes with safe placeholders.
    return text.encode("ascii", errors="ignore").decode("ascii")


def _objective_weights(obj: str) -> Tuple[float, float, float]:
    s = (obj or "").lower()
    if "delay" in s or "performance" in s:
        return (0.70, 0.20, 0.10)
    if "power" in s:
        return (0.20, 0.70, 0.10)
    if "area" in s:
        return (0.20, 0.20, 0.60)
    return (0.40, 0.40, 0.20)


def _extract_ngspice_error_line(log: str) -> Optional[str]:
    for line in (log or "").splitlines():
        l = line.lower()
        if any(k in l for k in ("error", "fatal", "unknown", "failed")):
            return line.strip()
    return None


def _extract_defined_models(netlist: str) -> Set[str]:
    models: Set[str] = set()
    for line in (netlist or "").splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"(?i)^\.model\s+(\S+)\s+", s)
        if m:
            models.add(m.group(1).lower())
    return models


def _inject_fallback_models(netlist: str, missing_model_names: List[str]) -> Tuple[str, List[str]]:
    if not missing_model_names:
        return netlist, []

    # Do not silently substitute generic MOS models: they invalidate PPA fidelity.
    return netlist, []
    
    existing = _extract_defined_models(netlist)
    to_add: List[str] = []
    added_names: List[str] = []

    for name in missing_model_names:
        key = (name or "").strip().lower()
        if key in fallback_cards and key not in existing:
            to_add.append(fallback_cards[key])
            added_names.append(key)
            existing.add(key)

    if not to_add:
        return netlist, []

    prefix = "* AutoPPA fallback models for missing MOS definitions\n"
    return f"{prefix}" + "\n".join(to_add) + "\n" + netlist, added_names


def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 2200) -> str:
    """Return an empty string on provider failure; callers must treat this as failure."""
    try:
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens)
        content = resp.choices[0].message.content
        return content.strip() if isinstance(content, str) else ""
    except Exception as exc:
        print(f"[LLM] call failed: {exc}")
        return ""


def _parse_subckt_ports(netlist: str) -> Dict[str, List[str]]:
    ports: Dict[str, List[str]] = {}
    for line in netlist.splitlines():
        m = SUBCKT_RE.match(line.strip())
        if m:
            name = m.group(1).lower()
            p = m.group(2).split()
            ports[name] = p
    return ports


def _element_nodes(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    kind = tokens[0][0].upper()
    if kind in ("R", "C", "L", "D", "V", "I"):
        return tokens[1:3] if len(tokens) >= 3 else []
    if kind == "M":
        return tokens[1:5] if len(tokens) >= 5 else []
    if kind == "Q":
        return tokens[1:4] if len(tokens) >= 4 else []
    if kind in ("X", "B"):
        return tokens[1:-1] if len(tokens) >= 3 else []
    return []


def _analyze_circuit_structure(netlist: str) -> Dict[str, Any]:
    structure: Dict[str, Any] = {
        "nodes": set(),
        "subcircuits": {},
        "connectivity": {},
        "inputs": set(),
        "outputs": set(),
        "fanout_map": {},
        "load_map": {},
        "issues": [],
        "element_map": {},
        "subckt_ports": _parse_subckt_ports(netlist),
    }

    in_subckt = False
    current_subckt = ""
    subckt_lines: Dict[str, List[str]] = {}

    for raw in netlist.splitlines():
        s = raw.strip()
        if not s:
            continue
        subckt_match = SUBCKT_RE.match(s)
        if subckt_match:
            current_subckt = subckt_match.group(1).lower()
            subckt_lines[current_subckt] = []
            in_subckt = True
            continue
        if ENDS_RE.match(s):
            in_subckt = False
            current_subckt = ""
            continue
        if in_subckt and current_subckt:
            subckt_lines[current_subckt].append(s)

    for name, lines in subckt_lines.items():
        device_lines = [l for l in lines if DEVICE_RE.match(l)]
        structure["subcircuits"][name] = {
            "line_count": len(lines),
            "device_count": len(device_lines),
            "has_devices": len(device_lines) > 0,
        }
        if not device_lines:
            structure["issues"].append(f"Empty subcircuit: {name}")

    for raw in netlist.splitlines():
        s = raw.strip()
        if not s or s.startswith("*") or s.startswith("."):
            continue
        tokens = s.split()
        if not tokens:
            continue
        elem = tokens[0]
        nodes = _element_nodes(tokens)
        structure["element_map"][elem] = {"line": s, "nodes": nodes}
        for n in nodes:
            structure["nodes"].add(n)
            structure["connectivity"].setdefault(n, []).append(elem)

        # Infer stimulus inputs from pulse/sin/pwl voltage sources.
        if elem[0].upper() == "V" and len(tokens) >= 4:
            upper_line = s.upper()
            if any(k in upper_line for k in ("PULSE", "SIN", "PWL")):
                structure["inputs"].add(tokens[1])

        # Infer outputs from explicit capacitive loads.
        if elem[0].upper() == "C" and len(tokens) >= 3 and tokens[2].lower() in ("0", "gnd"):
            structure["outputs"].add(tokens[1])

    # Additional output inference: nodes with high fanout and not rails.
    rails = {"0", "gnd", "vdd", "vss"}
    for node, elems in structure["connectivity"].items():
        mos_count = sum(1 for e in elems if e and e[0].upper() in ("M", "X", "Q", "B"))
        structure["fanout_map"][node] = mos_count
        structure["load_map"][node] = sum(1 for e in elems if e and e[0].upper() in ("C", "R"))
        if node.lower() not in rails and mos_count >= 2:
            structure["outputs"].add(node)

    return structure


def _build_node_graph(netlist: str) -> Dict[str, Set[str]]:
    """
    Build only a passive/source connectivity graph.

    A MOSFET's gate is a control terminal, not a conductive drain-source edge;
    treating all MOS terminals as one undirected component creates false paths.
    This graph is therefore a coarse topology aid, never proof of functionality.
    """
    graph: Dict[str, Set[str]] = {}
    for raw in netlist.splitlines():
        s = raw.strip()
        if not s or s.startswith(("*", ".")):
            continue
        tokens = s.split()
        if not tokens:
            continue
        kind = tokens[0][0].upper()
        if kind in ("R", "C", "L", "V", "I") and len(tokens) >= 3:
            a, b = tokens[1], tokens[2]
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        # D/BJT/MOS/subcircuit behavior is not represented as a simple wire.
    return graph


def _has_path(graph: Dict[str, Set[str]], src: str, dst: str) -> bool:
    if src == dst:
        return True
    if src not in graph or dst not in graph:
        return False
    q: deque[str] = deque([src])
    seen: Set[str] = {src}
    while q:
        n = q.popleft()
        for nxt in graph.get(n, set()):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def _extract_measure_nodes(testbench: str) -> Set[str]:
    nodes: Set[str] = set()
    for line in testbench.splitlines():
        if ".measure" not in line.lower():
            continue
        for m in re.findall(r"v\(([^)]+)\)", line, flags=re.IGNORECASE):
            nodes.add(m.strip())
    return nodes


def _validate_functional(netlist: str, testbench: str, structure: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "valid": True,
        "empty_subckts": [],
        "floating_outputs": [],
        "missing_nodes": [],
        "path_exists": False,
        "errors": [],
    }

    for name, info in structure.get("subcircuits", {}).items():
        if not info.get("has_devices", False):
            result["empty_subckts"].append(name)

    if result["empty_subckts"]:
        result["valid"] = False
        result["errors"].append(f"Empty .subckt blocks: {result['empty_subckts']}")

    outputs = sorted(structure.get("outputs", set()))
    connectivity = structure.get("connectivity", {})

    # Output node is floating if no connected active driver class M/X/Q/B.
    for out in outputs:
        elements = connectivity.get(out, [])
        has_driver = any(e and e[0].upper() in ("M", "X", "Q", "B") for e in elements)
        if not has_driver:
            result["floating_outputs"].append(out)

    if result["floating_outputs"]:
        result["valid"] = False
        result["errors"].append(f"Floating outputs: {result['floating_outputs']}")

    inputs = sorted(structure.get("inputs", set()))
    graph = _build_node_graph(netlist)
    # A passive-only graph cannot establish active transistor functionality.
    # Preserve the diagnostic, but never promote a coarse graph path to proof.
    path_exists = any(_has_path(graph, i, o) for i in inputs for o in outputs) if inputs and outputs else False
    result["path_exists"] = path_exists
    result["validation_level"] = "structural_only"
    result["warnings"] = [
        "Static connectivity checks do not prove transistor-level logical equivalence.",
        "Run NGSpice with explicit, circuit-specific stimulus and compare against a golden response."
    ]
    if not inputs or not outputs:
        result["valid"] = False
        result["errors"].append("Unable to infer circuit inputs and outputs; explicit testbench required")
    elif not path_exists:
        # Passive connectivity is not expected to connect a driven input to an
        # output through MOS devices, so this is a warning, not proof of failure.
        result["warnings"].append("No passive-only input-to-output path detected; active device behavior requires simulation.")

    # Measurement node existence validation.
    measure_nodes = _extract_measure_nodes(testbench)
    all_nodes = set(structure.get("nodes", set()))
    for node in sorted(measure_nodes):
        if node not in all_nodes and node.lower() not in {"0", "gnd"}:
            result["missing_nodes"].append(node)
    if result["missing_nodes"]:
        result["valid"] = False
        result["errors"].append(f"Measurement nodes missing in circuit: {result['missing_nodes']}")

    return result


def _validate_electrical(netlist: str, structure: Dict[str, Any], testbench: str) -> Dict[str, Any]:
    validation = {
        "valid": True,
        "signal_integrity": {},
        "power_delivery": {},
        "timing_analysis": {},
        "issues": [],
    }

    pulse_inputs = []
    for line in testbench.splitlines():
        if line.strip().upper().startswith("V") and any(k in line.upper() for k in ("PULSE", "PWL", "SIN")):
            parts = line.split()
            if len(parts) >= 3:
                pulse_inputs.append(parts[1])
    validation["signal_integrity"]["input_signals"] = pulse_inputs

    has_supply = any(re.search(r"\bvdd\b", l, flags=re.IGNORECASE) for l in netlist.splitlines())
    validation["power_delivery"]["has_vdd"] = has_supply
    if not has_supply:
        validation["issues"].append("No VDD rail found")
        validation["valid"] = False

    mos_gates: Set[str] = set()
    potentially_driven: Set[str] = set(structure.get("inputs", set()))
    potentially_driven.update(structure.get("outputs", set()))

    for line in netlist.splitlines():
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("."):
            continue
        parts = s.split()
        if MOS_RE.match(parts[0]) and len(parts) >= 4:
            mos_gates.add(parts[2])
        elif parts[0][0].upper() in ("V", "X", "B", "E", "G") and len(parts) >= 3:
            potentially_driven.update(parts[1:3])

    floating = sorted(n for n in mos_gates if n.lower() not in {"0", "gnd", "vdd", "vss"} and n not in potentially_driven)
    if floating:
        validation["issues"].append(f"Floating MOS gates: {floating}")
        validation["valid"] = False

    return validation


def _infer_inputs_outputs(netlist: str, structure: Dict[str, Any]) -> Tuple[List[str], List[str], str]:
    inputs = sorted(structure.get("inputs", set()))
    outputs = sorted(structure.get("outputs", set()))

    if not inputs:
        # Fallback to first subckt external port if available.
        ports = structure.get("subckt_ports", {})
        if ports:
            first_ports = next(iter(ports.values()))
            if first_ports:
                inputs = [first_ports[0]]

    if not outputs:
        # Infer by capacitor-to-ground in the top-level netlist.
        for line in netlist.splitlines():
            s = line.strip()
            if not s or s.startswith("*") or s.startswith("."):
                continue
            p = s.split()
            if p and p[0][0].upper() == "C" and len(p) >= 3 and p[2].lower() in ("0", "gnd"):
                outputs.append(p[1])

    power_node = "vdd"
    for line in netlist.splitlines():
        if re.search(r"\bvdd\b", line, flags=re.IGNORECASE):
            power_node = "vdd"
            break

    # Final defaults only if nothing inferred.
    if not inputs:
        inputs = ["in"]
    if not outputs:
        outputs = ["out"]

    return inputs, outputs, power_node


def _generate_testbench(netlist: str, analysis_type: str, structure: Dict[str, Any]) -> str:
    inputs, outputs, power_node = _infer_inputs_outputs(netlist, structure)
    primary_input = inputs[0]
    primary_output = outputs[0]

    lines = ["* Auto-generated Testbench"]
    raise ValueError(
        "Automatic testbench generation requires explicit circuit-specific supply, "
        "input vectors, load, and measurement thresholds. Provide a reviewed testbench "
        "instead of relying on guessed 1.8 V / 10 fF defaults."
    )

    lines.append(f"VDD {power_node} 0 DC 1.8")

    # Drive all inferred inputs; use one toggling source and bias others.
    for idx, node in enumerate(inputs):
        if (analysis_type or "").lower().startswith("trans") and idx == 0:
            lines.append(f"VIN_{idx} {node} 0 PULSE(0 1.8 0 10p 10p 500p 1n)")
        else:
            lines.append(f"VIN_{idx} {node} 0 DC 0.9")

    # Ensure measurable output load.
    has_output_cap = False
    for line in netlist.splitlines():
        p = line.split()
        if p and p[0][0].upper() == "C" and len(p) >= 3 and p[1] == primary_output and p[2].lower() in ("0", "gnd"):
            has_output_cap = True
            break
    if not has_output_cap:
        lines.append(f"CLOAD_{primary_output} {primary_output} 0 10f")

    analysis_norm = (analysis_type or "Transient").lower()
    if analysis_norm.startswith("trans"):
        lines.append(".tran 1p 2n")
        lines.append(
            f".measure tran delay_val TRIG v({primary_input}) VAL=0.9 RISE=1 "
            f"TARG v({primary_output}) VAL=0.9 FALL=1"
        )
        lines.append(f".measure tran avg_pwr AVG par('-v({power_node})*i(VDD)') FROM=0 TO=2n")
    elif analysis_norm.startswith("ac"):
        lines.append(".ac dec 20 1 1G")
        lines.append(f".measure ac gain_at_1M FIND vdb({primary_output}) AT=1MEG")
    else:
        lines.append(".op")

    lines.append(".end")
    return "\n".join(lines)


def _combine_netlist_and_testbench(netlist: str, testbench: str) -> str:
    core = re.sub(r"(?im)^\s*\.end\s*$", "", netlist).strip()
    tb = re.sub(r"(?im)^\s*\.end\s*$", "", testbench).strip()
    combined = f"{core}\n\n* ---- AUTO TESTBENCH ----\n{tb}\n.end\n"
    return _sanitize_spice_text(combined)


def _run_ngspice_combined(netlist: str, testbench: str) -> Dict[str, Any]:
    combined = _combine_netlist_and_testbench(netlist, testbench)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sp", delete=False, encoding="ascii") as tf:
            tf.write(combined)
            tmp_path = tf.name

        proc = subprocess.run(
            ["ngspice", "-b", tmp_path],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return {"log": log.strip(), "failed": proc.returncode != 0, "combined": combined}
    except Exception as exc:
        return {"log": f"Execution failed: {exc}", "failed": True, "combined": combined}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _analyze_simulation_errors(log: str) -> Dict[str, Any]:
    log_l = (log or "").lower()
    analysis = {
        "error_type": "none",
        "severity": "low",
        "fix_strategy": "none",
        "details": {},
        "log": log,
    }
    if "too few parameters" in log_l or "parameter mismatch" in log_l:
        analysis.update(error_type="subcircuit_port_mismatch", severity="high", fix_strategy="fix_ports")
    elif "could not find a valid modelname" in log_l or "unknown model" in log_l:
        missing = sorted({m.lower() for m in re.findall(r"\b([np]mos)\b", log_l, flags=re.IGNORECASE)})
        analysis.update(
            error_type="missing_model",
            severity="high",
            fix_strategy="add_default_models",
            details={"missing_models": missing},
        )
    elif "unknown subckt" in log_l or ("unknown" in log_l and "subcircuit" in log_l):
        analysis.update(error_type="missing_subcircuit", severity="high", fix_strategy="reject")
    elif "no such vector" in log_l or "vector not found" in log_l:
        analysis.update(error_type="invalid_measure_node", severity="high", fix_strategy="fix_measure")
    elif "singular matrix" in log_l or "timestep too small" in log_l:
        analysis.update(error_type="convergence", severity="medium", fix_strategy="add_small_stability")
    elif "error" in log_l or "fatal" in log_l:
        analysis.update(error_type="generic", severity="medium", fix_strategy="reject")
    return analysis


def _apply_fixes(netlist: str, testbench: str, error_analysis: Dict[str, Any], structure: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    fixes: List[str] = []
    netlist_out = netlist
    tb_out = testbench
    err = error_analysis.get("error_type", "none")

    if err == "invalid_measure_node":
        nodes = sorted(structure.get("nodes", set()))
        preferred = next((n for n in nodes if n.lower() not in {"0", "gnd", "vdd", "vss"}), None)
        if preferred:
            tb_out = re.sub(r"v\([^)]+\)", f"v({preferred})", tb_out, count=1, flags=re.IGNORECASE)
            fixes.append(f"Repointed first measurement node to existing node '{preferred}'")

    if err == "convergence":
        if not re.search(r"(?im)^\s*\.options\s+.*gmin", netlist_out):
            netlist_out = ".options gmin=1e-12 reltol=1e-3\n" + netlist_out
            fixes.append("Added conservative .options for convergence")

    if err == "missing_model":
        missing_models = error_analysis.get("details", {}).get("missing_models", [])
        netlist_out, added = _inject_fallback_models(netlist_out, missing_models)
        if added:
            fixes.append(f"Injected fallback .model cards for: {', '.join(sorted(added))}")

    return netlist_out, tb_out, fixes


def _parse_spice_number(raw: str) -> Optional[float]:
    """Parse common SPICE numeric suffixes into SI values."""
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([a-z]+)?\s*", raw, re.I)
    if not match:
        return None
    value = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    scale = {
        "": 1.0, "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
        "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
    }.get(suffix)
    return value * scale if scale is not None else None


def _estimate_area_from_netlist(netlist: str) -> Optional[float]:
    """Return sum(W*L) in square metres as a sizing proxy, not physical area."""
    area_m2 = 0.0
    found_mos = False
    for line in netlist.splitlines():
        s = line.strip()
        if not s or s.startswith(("*", ".")):
            continue
        parts = s.split()
        if not parts or not MOS_RE.match(parts[0]):
            continue
        found_mos = True
        params = {p.split("=", 1)[0].lower(): p.split("=", 1)[1]
                  for p in parts[5:] if "=" in p}
        w = _parse_spice_number(params.get("w", ""))
        l = _parse_spice_number(params.get("l", ""))
        if w is None or l is None or w <= 0 or l <= 0:
            return None
        area_m2 += w * l
    if not found_mos:
        return None
    return round(area_m2 * 1e12, 6)  # square micrometres (proxy)


def _parse_measures_from_log(log: str) -> Tuple[float, float]:
    delay_ps = float("inf")
    power_uw = float("inf")

    for line in (log or "").splitlines():
        l = line.strip().lower()
        if "delay_val" in l:
            m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if m:
                val = float(m.group(1))
                if val > 0:
                    delay_ps = val * 1e12
        elif "avg_pwr" in l:
            m = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if m:
                val = float(m.group(1))
                power_uw = abs(val) * 1e6

    return delay_ps, power_uw


def _improvement_summary(state: AutoPPA, current: Dict[str, float]) -> Dict[str, Any]:
    baseline_delay = state.get("baseline_delay", float("inf"))
    baseline_power = state.get("baseline_power", float("inf"))
    baseline_area = state.get("baseline_area", float("inf"))

    def pct(old: float, new: float) -> Optional[float]:
        if old in (0.0, float("inf")) or new == float("inf"):
            return None
        return (old - new) / old * 100.0

    return {
        "delay_pct_vs_baseline": pct(baseline_delay, current["delay"]),
        "power_pct_vs_baseline": pct(baseline_power, current["power"]),
        "area_pct_vs_baseline": pct(baseline_area, current["area"]),
    }


def input_loader(state: AutoPPA) -> Dict[str, Any]:
    raw = ""
    for key in ("netlist_raw", "working_netlist", "netlist", "spice_netlist"):
        val = state.get(key, "")
        if isinstance(val, str) and val.strip():
            raw = val.strip()
            break
    if not raw:
        raise ValueError("SPICE netlist is empty.")

    explicit_tb = state.get("testbench", "")
    if not isinstance(explicit_tb, str) or not explicit_tb.strip():
        raise ValueError(
            "A circuit-specific testbench is required. Supply explicit power rails, "
            "input stimulus, output load, and measurement directives."
        )

    cleaned = _strip_comments(_sanitize_spice_text(raw))
    return {
        "netlist_raw": raw,
        "working_netlist": cleaned,
        "baseline_netlist": cleaned,
        "circuit_name": state.get("circuit_name", "Untitled"),
        "circuit_type": state.get("circuit_type", "Logic"),
        "prompt": state.get("prompt", ""),
        "brief_description": state.get("brief_description", ""),
        "opt_objective": state.get("opt_objective", "Balanced PPA"),
        "max_iterations": int(state.get("max_iterations", 5)),
        "analysis_type": state.get("analysis_type", "Transient"),
        "iteration": 0,
        "history": [],
        "error_memory": _load_error_memory(),
        "fix_history": [],
        "applied_fixes": [],
        "best_cost": float("inf"),
        "best_delay": float("inf"),
        "best_power": float("inf"),
        "best_area": float("inf"),
        "best_netlist": cleaned,
        "best_iteration": 0,
        "accepted": False,
        "reject_reason": "",
        "baseline_delay": float("inf"),
        "baseline_power": float("inf"),
        "baseline_area": float("inf"),
        "continue_loop": True,
        "continue_reason": "init",
        "fix_attempts": 0,
        "testbench": explicit_tb,
        "area_estimate_um2": _estimate_area_from_netlist(cleaned),
        "area_estimate_method": "sum(W*L) proxy; not physical layout area",
        "area_is_physical": False,
        "functional_validation_level": "simulation_required",
        "validation_warnings": [
            "Static topology checks cannot prove logic equivalence.",
            "Physical area requires synthesis/layout data; W*L is only a sizing proxy."
        ],
    }


def structural_validator(state: AutoPPA) -> Dict[str, Any]:
    netlist = state.get("proposed_netlist") or state.get("working_netlist", "")
    structure = _analyze_circuit_structure(netlist)
    return {
        "circuit_structure": structure,
        "connectivity_map": structure["connectivity"],
        "fanout_analysis": structure["fanout_map"],
        "load_analysis": structure["load_map"],
        "timing_paths": [],
    }


def functional_validator_node(state: AutoPPA) -> Dict[str, Any]:
    netlist = state.get("proposed_netlist") or state.get("working_netlist", "")
    structure = _analyze_circuit_structure(netlist)
    testbench = state.get("testbench", "")
    if not testbench:
        testbench = _generate_testbench(netlist, state.get("analysis_type", "Transient"), structure)

    result = _validate_functional(netlist, testbench, structure)
    return {
        "circuit_structure": structure,
        "functional_valid": result["valid"],
        "functional_errors": result["errors"],
        "empty_subckts": result["empty_subckts"],
        "floating_outputs": result["floating_outputs"],
        "missing_nodes": result["missing_nodes"],
        "path_exists": result["path_exists"],
        "testbench": testbench,
    }


def electrical_validator(state: AutoPPA) -> Dict[str, Any]:
    netlist = state.get("proposed_netlist") or state.get("working_netlist", "")
    testbench = state.get("testbench", "")
    structure = state.get("circuit_structure", {})
    result = _validate_electrical(netlist, structure, testbench)
    return {
        "electrical_valid": result["valid"],
        "signal_integrity": result["signal_integrity"],
        "power_delivery": result["power_delivery"],
        "timing_analysis": result["timing_analysis"],
    }


def llm_design_generator(state: AutoPPA) -> Dict[str, Any]:
    iteration = state.get("iteration", 0)
    current_netlist = state.get("working_netlist", "")
    structure = state.get("circuit_structure", {})
    history = state.get("history", [])

    if iteration == 0:
        # Establish baseline by simulating original design first.
        return {
            "proposed_netlist": current_netlist,
            "reasoning": "Baseline iteration: using original netlist for reference metrics.",
            "targets": {"delay_target_ps": 100.0, "power_target_uw": 200.0},
            "area_um2": _estimate_area_from_netlist(current_netlist),
            "fix_attempts": 0,
            "testbench": state.get("testbench") or _generate_testbench(
                current_netlist, state.get("analysis_type", "Transient"), structure
            ),
        }

    last = history[-1] if history else {}
    error_context = "\n".join(state.get("error_memory", [])[-5:])

    prompt = f"""
You are an expert VLSI SPICE optimizer.

Iteration: {iteration}
Objective: {state.get('opt_objective', 'Balanced PPA')}
Last metrics: delay_ps={last.get('delay_ps')}, power_uw={last.get('power_uw')}, area_um2={last.get('area_um2')}
Functional errors to fix: {state.get('functional_errors', [])}

MANDATORY RULES:
{SAFE_OPTIMIZATION_RULES}

Current netlist:
{current_netlist}

Structural summary:
- Empty subckts: {state.get('empty_subckts', [])}
- Floating outputs: {state.get('floating_outputs', [])}
- High-fanout nodes (>4): {[n for n, f in (structure.get('fanout_map', {}) or {}).items() if f > 4]}

Past recurring errors:
{error_context}

Return ONLY JSON:
{{
  "proposed_netlist": "<full valid SPICE netlist>",
  "reasoning": "<short summary>",
  "targets": {{"delay_target_ps": 100, "power_target_uw": 200}}
}}
"""

    raw = _call_llm(
        [
            {
                "role": "system",
                "content": "You optimize SPICE safely. Preserve functionality; emit strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=3200,
    )
    parsed = _extract_json(raw)
    if not raw or not isinstance(parsed, dict) or not isinstance(parsed.get("proposed_netlist"), str):
        return {
            "proposed_netlist": current_netlist,
            "reasoning": "LLM generation failed or returned invalid JSON; candidate rejected.",
            "llm_status": "error",
            "llm_error": "Empty response or missing proposed_netlist",
            "candidate_unchanged": True,
            "fix_attempts": 0,
        }

    candidate = _decode_escaped_text(parsed.get("proposed_netlist", ""))
    candidate = _strip_comments(candidate)
    candidate = _sanitize_spice_text(candidate)

    if not candidate.strip() or candidate == current_netlist:
        return {
            "proposed_netlist": current_netlist,
            "reasoning": "LLM did not produce a changed candidate; no-op rejected.",
            "llm_status": "no_change",
            "llm_error": "Candidate is empty or unchanged",
            "candidate_unchanged": True,
            "fix_attempts": 0,
        }

    testbench = state.get("testbench", "")
    return {
        "proposed_netlist": candidate,
        "testbench": testbench,
        "reasoning": parsed.get("reasoning", "Applied safe optimization constraints."),
        "targets": parsed.get("targets", {"delay_target_ps": 100.0, "power_target_uw": 200.0}),
        "area_um2": _estimate_area_from_netlist(candidate),
        "fix_attempts": 0,
        "llm_status": "ok",
        "llm_error": "",
        "candidate_unchanged": False,
        "area_estimate_um2": _estimate_area_from_netlist(candidate),
        "area_estimate_method": "sum(W*L) proxy; not physical layout area",
        "area_is_physical": False,
    }


def simulation_runner(state: AutoPPA) -> Dict[str, Any]:
    netlist = state.get("proposed_netlist") or state.get("working_netlist", "")
    structure = state.get("circuit_structure") or _analyze_circuit_structure(netlist)
    testbench = state.get("testbench") or _generate_testbench(netlist, state.get("analysis_type", "Transient"), structure)
    if not testbench.strip():
        raise ValueError("Missing explicit circuit-specific testbench.")
    result = _run_ngspice_combined(netlist, testbench)
    return {
        "ngspice_log": result["log"],
        "sim_failed": result["failed"],
        "combined_circuit": result["combined"],
        "testbench": testbench,
    }


def error_analyzer(state: AutoPPA) -> Dict[str, Any]:
    analysis = _analyze_simulation_errors(state.get("ngspice_log", ""))
    return {"error_analysis": analysis, "sim_error_pattern": analysis.get("error_type", "none")}


def deterministic_fixer(state: AutoPPA) -> Dict[str, Any]:
    fixed_netlist, fixed_tb, fixes = _apply_fixes(
        state.get("proposed_netlist", state.get("working_netlist", "")),
        state.get("testbench", ""),
        state.get("error_analysis", {}),
        state.get("circuit_structure", {}),
    )
    return {
        "proposed_netlist": fixed_netlist,
        "testbench": fixed_tb,
        "applied_fixes": fixes,
        "fix_history": state.get("fix_history", []) + fixes,
        "fix_attempts": int(state.get("fix_attempts", 0)) + 1,
    }


def llm_parse_results(state: AutoPPA) -> Dict[str, Any]:
    if state.get("sim_failed", True):
        return {
            "delay_ps": float("inf"),
            "power_uw": float("inf"),
            "area_um2": state.get("area_um2", _estimate_area_from_netlist(state.get("proposed_netlist", ""))),
        }

    log = state.get("ngspice_log", "")
    delay, power = _parse_measures_from_log(log)
    if delay == float("inf") and power == float("inf"):
        raw = _call_llm(
            [{"role": "system", "content": PARSE_SYSTEM}, {"role": "user", "content": log[:4500]}],
            max_tokens=600,
        )
        parsed = _extract_json(raw) or {}
        delay = float(parsed.get("delay_ps", float("inf")) or float("inf"))
        power = float(parsed.get("power_uw", float("inf")) or float("inf"))

    area = _estimate_area_from_netlist(state.get("proposed_netlist", ""))
    return {
        "delay_ps": delay,
        "power_uw": power,
        "area_um2": area if area is not None else float("inf"),
        "area_estimate_um2": area,
        "area_estimate_method": "sum(W*L) proxy; not physical layout area",
        "area_is_physical": False,
    }


def acceptance_checker(state: AutoPPA) -> Dict[str, Any]:
    current_delay = state.get("delay_ps", float("inf"))
    current_power = state.get("power_uw", float("inf"))
    current_area = state.get("area_um2", float("inf"))

    sim_ok = not state.get("sim_failed", True)
    # Simulation with a user-supplied golden testbench is required. Static
    # connectivity checks are not a functional-equivalence proof.
    func_ok = bool(state.get("functional_valid", False)) and bool(state.get("golden_validation_passed", False))

    best_delay = state.get("best_delay", float("inf"))
    best_power = state.get("best_power", float("inf"))
    best_area = state.get("best_area", float("inf"))

    reject_reasons: List[str] = []
    if not sim_ok:
        reject_reasons.append("simulation failed")
    if not func_ok:
        reject_reasons.append("functional validation failed")
    if state.get("functional_errors"):
        reject_reasons.append(f"functional errors: {state.get('functional_errors')}")

    improved_any = False
    if sim_ok and func_ok:
        objective = (state.get("opt_objective") or "Balanced PPA").lower()
        delay_ok = isfinite(float(current_delay))
        power_ok = isfinite(float(current_power))
        area_ok = isfinite(float(current_area))
        if not (delay_ok and power_ok):
            reject_reasons.append("delay/power measurements unavailable")
        elif best_delay == float("inf") and best_power == float("inf"):
            improved_any = True  # baseline only after verified simulation + golden checks
        else:
            # Objective-aware acceptance: primary metric must improve; other
            # measured metrics may not regress beyond a small explicit tolerance.
            tolerance = float(state.get("regression_tolerance", 0.02))
            if "delay" in objective or "performance" in objective:
                improved_any = current_delay < best_delay and current_power <= best_power * (1 + tolerance)
            elif "power" in objective:
                improved_any = current_power < best_power and current_delay <= best_delay * (1 + tolerance)
            elif "area" in objective:
                improved_any = area_ok and current_area < best_area and current_delay <= best_delay * (1 + tolerance) and current_power <= best_power * (1 + tolerance)
            else:
                # Balanced: require Pareto dominance (no metric worsens, at least
                # one improves). If area unavailable, compare delay and power.
                compared = [(current_delay, best_delay), (current_power, best_power)]
                if area_ok and isfinite(float(best_area)):
                    compared.append((current_area, best_area))
                improved_any = all(c <= b * (1 + tolerance) for c, b in compared) and any(c < b for c, b in compared)
            if not improved_any:
                reject_reasons.append("candidate does not satisfy objective-aware non-regression criteria")

    accepted = sim_ok and func_ok and improved_any and not reject_reasons

    print(
        f"[Acceptance] iter={state.get('iteration', 0)} accepted={accepted} "
        f"delay={current_delay} power={current_power} area={current_area} "
        f"reason={' | '.join(reject_reasons) if reject_reasons else 'accepted'}"
    )

    out: Dict[str, Any] = {
        "accepted": accepted,
        "reject_reason": "; ".join(reject_reasons),
        "best_delay": best_delay,
        "best_power": best_power,
        "best_area": best_area,
        "best_netlist": state.get("best_netlist", state.get("working_netlist", "")),
        "best_iteration": state.get("best_iteration", 0),
        "working_netlist": state.get("working_netlist", ""),
    }

    if accepted:
        out.update(
            {
                "best_delay": current_delay,
                "best_power": current_power,
                "best_area": current_area,
                "best_netlist": state.get("proposed_netlist", state.get("working_netlist", "")),
                "best_iteration": state.get("iteration", 0),
                "working_netlist": state.get("proposed_netlist", state.get("working_netlist", "")),
            }
        )
    else:
        # Explicit revert to previous best design.
        out["proposed_netlist"] = state.get("best_netlist", state.get("working_netlist", ""))

    return out


def cost_tracker(state: AutoPPA) -> Dict[str, Any]:
    delay = state.get("delay_ps", float("inf"))
    power = state.get("power_uw", float("inf"))
    area = state.get("area_um2", float("inf"))
    targets = state.get("targets", {})
    dt = max(float(targets.get("delay_target_ps", 100.0)), 1e-9)
    pt = max(float(targets.get("power_target_uw", 200.0)), 1e-9)

    d_score = (delay / dt) if isfinite(float(delay)) else 999.0
    p_score = (power / pt) if isfinite(float(power)) else 999.0
    a_score = (area / max(float(state.get("area_target_um2", 1.0)), 1e-12)
               if isfinite(float(area)) else 999.0)

    w_d, w_p, w_a = _objective_weights(state.get("opt_objective", "Balanced PPA"))
    cost = w_d * d_score + w_p * p_score + w_a * a_score

    # Baseline tracking
    baseline_delay = state.get("baseline_delay", float("inf"))
    baseline_power = state.get("baseline_power", float("inf"))
    baseline_area = state.get("baseline_area", float("inf"))

    updates: Dict[str, Any] = {
        "cost": round(cost, 6),
        "constraints_met": delay <= dt and power <= pt,
        "met_details": {
            "delay_target_ps": dt,
            "power_target_uw": pt,
            "delay": delay,
            "power": power,
            "area": area,
            "delay_met": delay <= dt,
            "power_met": power <= pt,
        },
    }

    if state.get("iteration", 0) == 0 and delay < float("inf") and power < float("inf"):
        updates["baseline_delay"] = delay
        updates["baseline_power"] = power
        updates["baseline_area"] = area

    updates["metric_tracking"] = {
        "baseline": {
            "delay_ps": baseline_delay if baseline_delay < float("inf") else updates.get("baseline_delay", baseline_delay),
            "power_uw": baseline_power if baseline_power < float("inf") else updates.get("baseline_power", baseline_power),
            "area_um2": baseline_area if baseline_area < float("inf") else updates.get("baseline_area", baseline_area),
        },
        "current": {"delay_ps": delay, "power_uw": power, "area_um2": area},
        "best": {
            "delay_ps": state.get("best_delay", float("inf")),
            "power_uw": state.get("best_power", float("inf")),
            "area_um2": state.get("best_area", float("inf")),
        },
    }
    return updates


def history_logger(state: AutoPPA) -> Dict[str, Any]:
    history = list(state.get("history", []))
    iter_idx = state.get("iteration", 0)
    lessons = list(state.get("error_memory", []))

    err_line = _extract_ngspice_error_line(state.get("ngspice_log", ""))
    if err_line and err_line not in lessons:
        lessons.append(err_line)
        lessons = lessons[-12:]
        _save_error_memory(lessons)

    improv = _improvement_summary(
        state,
        {
            "delay": state.get("delay_ps", float("inf")),
            "power": state.get("power_uw", float("inf")),
            "area": state.get("area_um2", float("inf")),
        },
    )

    entry = {
        "iteration": iter_idx,
        "accepted": state.get("accepted", False),
        "reject_reason": state.get("reject_reason", ""),
        "sim_ok": not state.get("sim_failed", True),
        "functional_ok": state.get("functional_valid", False),
        "delay_ps": state.get("delay_ps"),
        "power_uw": state.get("power_uw"),
        "area_um2": state.get("area_um2"),
        "cost": state.get("cost"),
        "improvement": improv,
        "reasoning": state.get("reasoning", ""),
        "fixes": state.get("applied_fixes", []),
    }
    history.append(entry)

    return {"history": history, "error_memory": lessons, "iteration": iter_idx + 1}


def convergence_checker(state: AutoPPA) -> Dict[str, Any]:
    it = state.get("iteration", 0)
    max_it = state.get("max_iterations", 5)
    history = state.get("history", [])

    if it >= max_it:
        return {"continue_loop": False, "continue_reason": "max_iterations"}

    # Stop on stagnation: 3 consecutive rejected candidates.
    if len(history) >= 3 and all(not h.get("accepted", False) for h in history[-3:]):
        return {"continue_loop": False, "continue_reason": "three_rejections"}

    # Stop on convergence if very small cost delta across recent accepted iterations.
    accepted = [h for h in history if h.get("accepted") and h.get("cost") is not None]
    if len(accepted) >= 3:
        recent = accepted[-3:]
        delta = max(h["cost"] for h in recent) - min(h["cost"] for h in recent)
        if delta < 1e-3:
            return {"continue_loop": False, "continue_reason": "cost_converged"}

    return {"continue_loop": True, "continue_reason": "continue"}


def final_optimizer(state: AutoPPA) -> Dict[str, Any]:
    best_netlist = state.get("best_netlist", state.get("working_netlist", ""))
    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best Iteration: {state.get('best_iteration', 0)}")
    print(f"Best Delay: {state.get('best_delay', float('inf'))} ps")
    print(f"Best Power: {state.get('best_power', float('inf'))} uW")
    print(f"Best Area: {state.get('best_area', float('inf'))} um2")

    return {
        "optimised_netlist": best_netlist,
        "final_delay": state.get("best_delay"),
        "final_power": state.get("best_power"),
        "final_area": state.get("best_area"),
        "convergence_score": state.get("cost", 0.0),
        "area_is_physical": False,
        "area_estimate_um2": state.get("area_estimate_um2"),
        "area_estimate_method": state.get("area_estimate_method", "unknown"),
    }


def route_after_functional(state: AutoPPA) -> str:
    return "valid" if state.get("functional_valid", False) else "invalid"


def route_after_simulation(state: AutoPPA) -> str:
    if state.get("sim_failed", True):
        fix_attempts = int(state.get("fix_attempts", 0))
        if state.get("sim_error_pattern") in {"invalid_measure_node", "convergence", "missing_model"} and fix_attempts < 2:
            return "fixable"
        return "failed"
    return "success"


def route_convergence(state: AutoPPA) -> str:
    return "continue" if state.get("continue_loop", True) else "end"


graph = StateGraph(AutoPPA)

graph.add_node("input_loader", input_loader)
graph.add_node("structural_validator", structural_validator)
graph.add_node("functional_validator", functional_validator_node)
graph.add_node("electrical_validator", electrical_validator)
graph.add_node("llm_design_generator", llm_design_generator)
graph.add_node("simulation_runner", simulation_runner)
graph.add_node("error_analyzer", error_analyzer)
graph.add_node("deterministic_fixer", deterministic_fixer)
graph.add_node("llm_parse_results", llm_parse_results)
graph.add_node("acceptance_checker", acceptance_checker)
graph.add_node("cost_tracker", cost_tracker)
graph.add_node("convergence_checker", convergence_checker)
graph.add_node("history_logger", history_logger)
graph.add_node("final_optimizer", final_optimizer)

graph.add_edge(START, "input_loader")
graph.add_edge("input_loader", "structural_validator")
graph.add_edge("structural_validator", "functional_validator")

graph.add_conditional_edges(
    "functional_validator",
    route_after_functional,
    {"valid": "electrical_validator", "invalid": "llm_design_generator"},
)

graph.add_edge("electrical_validator", "llm_design_generator")
graph.add_edge("llm_design_generator", "simulation_runner")
graph.add_edge("simulation_runner", "error_analyzer")

graph.add_conditional_edges(
    "error_analyzer",
    route_after_simulation,
    {"fixable": "deterministic_fixer", "failed": "llm_design_generator", "success": "llm_parse_results"},
)

graph.add_edge("deterministic_fixer", "simulation_runner")
graph.add_edge("llm_parse_results", "acceptance_checker")
graph.add_edge("acceptance_checker", "cost_tracker")
graph.add_edge("cost_tracker", "convergence_checker")
graph.add_edge("convergence_checker", "history_logger")

graph.add_conditional_edges("history_logger", route_convergence, {"continue": "structural_validator", "end": "final_optimizer"})

graph.add_edge("final_optimizer", END)

app = graph.compile()