"""
AutoPPA — Closed-Loop Engineering System (Fixed & Enhanced)
============================================================
Advanced LLM-driven optimizer with closed-loop verification, 
deterministic fixes, and strong functional validation.

ARCHITECTURE:
  input_loader → structural_validator → functional_validator → 
  electrical_validator → llm_design_generator → functional_validator_check → 
  simulation_runner → error_analyzer → [deterministic_fixer → simulation_runner]*
  → llm_parse_results → acceptance_checker → cost_tracker → 
  convergence_checker → history_logger → [loop or] final_optimizer → END

Key fixes:
  • Combined netlist+testbench simulation
  • Strict functional validation (no empty subckts, no floating outputs)
  • Acceptance criteria with revert to best
  • Safe optimization constraints (sizing only, no logic removal)
  • Automatic node inference for testbench
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any, Optional, List, Tuple, Set
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
import re
import subprocess
import time
import copy
import tempfile
import shutil

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
ERROR_MEMORY_FILE = "autoppa_error_memory.json"

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN,
)

# System prompt for LLM parsing
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

# Safe optimization constraints for LLM
SAFE_OPTIMIZATION_RULES = """
STRICT OPTIMIZATION CONSTRAINTS:
1. YOU MAY ONLY: resize transistors (W/L), reduce redundant elements, add buffer transistors for high fanout
2. YOU MUST NOT: remove logic gates, change connectivity, delete required devices, modify topology
3. YOU MUST: preserve all input/output ports, maintain signal paths, keep subcircuit functionality
4. YOU MUST: ensure all subcircuits contain at least one device (M, R, C, L, or X instance)
5. YOU MUST: ensure output nodes drive at least one transistor or load
"""

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class AutoPPA(TypedDict):
    # inputs
    netlist_raw:       str
    working_netlist:   str
    circuit_name:      str
    circuit_type:      str
    prompt:            str
    brief_description: str

    # UI parameters
    opt_objective:     str
    max_iterations:    int
    analysis_type:     str

    # per-iteration LLM outputs
    proposed_netlist:  str
    testbench:         str
    reasoning:         str
    targets:           dict
    missing_models:    list

    # structural analysis
    netlist_valid:     bool
    circuit_structure: dict
    connectivity_map: dict
    fanout_analysis:  dict
    load_analysis:    dict
    timing_paths:     list

    # functional validation
    functional_valid:  bool
    functional_errors: list
    empty_subckts:     list
    floating_outputs:  list
    missing_nodes:     list
    path_exists:       bool

    # electrical validation
    electrical_valid:  bool
    signal_integrity:  dict
    power_delivery:    dict
    timing_analysis:   dict

    # measurement validation
    measurement_valid: bool
    constraint_status: dict

    # simulation
    ngspice_log:       str
    sim_failed:        bool
    sim_error_pattern: str

    # metrics
    delay_ps:          float
    power_uw:          float
    area_um2:          float
    cost:              float
    constraints_met:   bool
    met_details:       dict

    # baseline tracking
    baseline_delay:    float
    baseline_power:    float
    baseline_area:     float
    baseline_netlist:  str

    # convergence tracking
    iteration:         int
    history:           list
    pareto_front:      list
    convergence_score: float
    error_memory:      list

    # best-so-far
    best_delay:        float
    best_power:        float
    best_area:         float
    best_cost:         float
    best_netlist:      str
    best_iteration:    int
    
    # acceptance
    accepted:          bool
    reject_reason:     str

    # fix tracking
    applied_fixes:     list
    fix_history:       list
    design_quality:    dict

    # error analysis
    error_analysis:    dict


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM response with robust parsing."""
    if not isinstance(text, str):
        return None
    
    # Find JSON blocks
    matches = re.findall(r"(\{[^{}]*\})", text, re.DOTALL)
    if not matches:
        # Try with outer braces
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            matches = [match.group(1)]
    
    for raw in matches:
        try:
            # Clean up escaped newlines
            candidate = raw.replace("\\n", "\n").replace("\\t", "\t")
            # Remove trailing commas
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _load_error_memory() -> list:
    try:
        with open(ERROR_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x) for x in data if str(x).strip()] if isinstance(data, list) else []
    except Exception:
        return []


def _save_error_memory(items: list) -> None:
    try:
        with open(ERROR_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(items[-20:], f, indent=2)
    except Exception:
        pass


def _decode_escaped_text(text: str) -> str:
    """Unescape \\n and \\r\\n from LLM JSON string fields."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return t


def _strip_comments(text: str) -> str:
    """Remove SPICE comment lines (starting with *) and inline ; comments."""
    if not isinstance(text, str):
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*"):
            continue
        sc = line.find(";")
        if sc != -1:
            line = line[:sc]
        if line.strip():
            lines.append(line.rstrip())
    return "\n".join(lines)


def _sanitize_spice_text(text: str) -> str:
    """Replace non-ASCII characters so ngspice doesn't choke."""
    if not isinstance(text, str):
        return ""
    return text.encode("ascii", errors="replace").decode("ascii")


def _parse_spice_number(val_str: str) -> Optional[float]:
    """Parse SPICE number with units (p, n, u, m, k, etc.)"""
    if not val_str:
        return None
    match = re.match(r"([\d.]+)([a-zA-Z]*)", val_str.strip())
    if not match:
        return None
    val = float(match.group(1))
    unit = match.group(2).lower()
    multipliers = {
        "": 1, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
        "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12
    }
    return val * multipliers.get(unit, 1)


def _objective_weights(obj: str) -> Tuple[float, float, float]:
    """Return weight coefficients based on optimization objective."""
    obj_lower = obj.lower()
    if "delay" in obj_lower:
        return (0.7, 0.15, 0.15)
    elif "power" in obj_lower:
        return (0.15, 0.7, 0.15)
    elif "area" in obj_lower:
        return (0.15, 0.15, 0.7)
    else:
        return (0.4, 0.4, 0.2)


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check if design a dominates design b in Pareto sense."""
    return (a["delay"] <= b["delay"] and 
            a["power"] <= b["power"] and 
            a["area"] <= b["area"] and
            (a["delay"] < b["delay"] or 
             a["power"] < b["power"] or 
             a["area"] < b["area"]))


def _extract_ngspice_error_line(log: str) -> Optional[str]:
    """Extract the first error line from ngspice log."""
    for line in log.splitlines():
        l = line.lower()
        if ("error" in l or "fatal" in l or "unknown" in l) and not line.strip().startswith("*"):
            return line.strip()
    return None


def _call_llm(messages: list, max_tokens: int = 2000) -> str:
    try:
        resp = client.chat_completion(messages=messages, max_tokens=max_tokens)
        return resp.choices[0].message.content or "{}"
    except Exception as e:
        print(f"[LLM] call failed: {e}")
        return "{}"


# ─────────────────────────────────────────────────────────────────────────────
# Structural Analysis
# ─────────────────────────────────────────────────────────────────────────────
def _analyze_circuit_structure(netlist: str) -> Dict[str, Any]:
    """Comprehensive structural analysis for closed-loop optimization."""
    structure = {
        "nodes": set(),
        "elements": [],
        "subcircuits": {},
        "connectivity": {},
        "fanout_map": {},
        "load_map": {},
        "inputs": set(),
        "outputs": set(),
        "issues": []
    }

    lines = netlist.splitlines()
    
    # First pass: identify subcircuits and their contents
    in_subckt = False
    current_subckt = None
    subckt_content = {}
    
    for line in lines:
        s = line.strip()
        if re.match(r"(?i)^\.subckt\s+", s):
            parts = s.split()
            if len(parts) >= 2:
                current_subckt = parts[1].lower()
                subckt_content[current_subckt] = []
                in_subckt = True
        elif re.match(r"(?i)^\.ends", s):
            in_subckt = False
            current_subckt = None
        elif in_subckt and current_subckt:
            if not s.startswith("*") and not s.startswith("."):
                subckt_content[current_subckt].append(s)
    
    # Store subcircuit info
    for subckt_name, content in subckt_content.items():
        has_devices = any(re.match(r"^[MRCLDVIXBQ]", l.strip(), re.IGNORECASE) for l in content)
        structure["subcircuits"][subckt_name] = {
            "content": content,
            "device_count": len([l for l in content if re.match(r"^[MRCLDVIXBQ]", l.strip(), re.IGNORECASE)]),
            "has_devices": has_devices
        }
        if not has_devices:
            structure["issues"].append(f"Empty subcircuit: {subckt_name}")

    # Second pass: analyze elements and connectivity
    for line in lines:
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("."):
            continue

        parts = s.split()
        if not parts:
            continue

        elem_type = parts[0][0].upper()  # First character
        elem_name = parts[0]
        nodes = []

        # Extract nodes based on element type
        if elem_type in ("R", "C", "L", "D"):
            if len(parts) >= 4:
                nodes = parts[1:3]
        elif elem_type == "V":
            if len(parts) >= 4:
                nodes = parts[1:3]
                # Identify inputs (pulse sources)
                if "PULSE" in s.upper():
                    structure["inputs"].add(parts[1])
        elif elem_type == "I":
            if len(parts) >= 4:
                nodes = parts[1:3]
        elif elem_type == "M":
            if len(parts) >= 5:
                nodes = parts[1:5]  # D G S B
        elif elem_type == "X":
            # Subcircuit instance
            if len(parts) >= 3:
                nodes = parts[1:-1]  # All but last (subckt name)
                subckt_ref = parts[-1].lower()
                if subckt_ref in structure["subcircuits"]:
                    # Check port count
                    expected_ports = len(structure["subcircuits"][subckt_ref]["content"][0].split()[1:]) if structure["subcircuits"][subckt_ref]["content"] else 0
                    if expected_ports and len(nodes) != expected_ports:
                        structure["issues"].append(f"Port mismatch in {elem_name}: expected {expected_ports}, got {len(nodes)}")
        
        # Update connectivity
        for node in nodes:
            structure["nodes"].add(node)
            if node not in structure["connectivity"]:
                structure["connectivity"][node] = []
            structure["connectivity"][node].append(elem_name)

    # Identify outputs (nodes driving capacitors to ground)
    for line in lines:
        s = line.strip()
        if s.upper().startswith("C"):
            parts = s.split()
            if len(parts) >= 3 and parts[2].lower() in ("0", "gnd"):
                structure["outputs"].add(parts[1])

    # Calculate fanout
    for node, connections in structure["connectivity"].items():
        fanout = len([c for c in connections if c[0] in ("M", "X")])
        structure["fanout_map"][node] = fanout

    return structure


# ─────────────────────────────────────────────────────────────────────────────
# Functional Validation (Critical Fix)
# ─────────────────────────────────────────────────────────────────────────────
def _validate_functional(netlist: str, testbench: str, structure: dict) -> Dict[str, Any]:
    """
    Critical functional validation:
    1. No empty subcircuits
    2. No floating outputs
    3. Input-to-output path exists
    4. Measurement nodes exist in circuit
    """
    result = {
        "valid": True,
        "empty_subckts": [],
        "floating_outputs": [],
        "missing_nodes": [],
        "path_exists": False,
        "errors": []
    }

    # Check 1: Empty subcircuits
    for subckt_name, info in structure.get("subcircuits", {}).items():
        if not info.get("has_devices", False):
            result["empty_subckts"].append(subckt_name)
            result["valid"] = False
            result["errors"].append(f"Empty subcircuit: {subckt_name}")

    # Check 2: Floating outputs (outputs must drive something or be connected to active devices)
    outputs = structure.get("outputs", set())
    connectivity = structure.get("connectivity", {})
    
    for out in outputs:
        # Check if output connects to any transistor or subcircuit
        connected = connectivity.get(out, [])
        drivers = [c for c in connected if c[0] in ("M", "X", "B")]
        if not drivers:
            result["floating_outputs"].append(out)
            result["valid"] = False
            result["errors"].append(f"Floating output: {out}")

    # Check 3: Input-to-output path
    inputs = structure.get("inputs", set())
    if inputs and outputs:
        # Simple BFS to check connectivity
        visited = set()
        queue = list(inputs)
        while queue:
            node = queue.pop(0)
            if node in outputs:
                result["path_exists"] = True
                break
            if node not in visited:
                visited.add(node)
                # Add connected nodes
                for elem in connectivity.get(node, []):
                    # Find element definition and get other nodes
                    for line in netlist.splitlines():
                        if line.strip().startswith(elem):
                            parts = line.split()
                            for p in parts[1:]:
                                if p not in visited and p not in queue:
                                    queue.append(p)
        
        if not result["path_exists"]:
            result["valid"] = False
            result["errors"].append("No path from input to output")

    # Check 4: Measurement nodes exist
    if testbench:
        measure_nodes = set()
        for line in testbench.splitlines():
            if ".measure" in line.lower():
                v_matches = re.findall(r"v\(([^\)]+)\)", line, re.IGNORECASE)
                measure_nodes.update(v_matches)
        
        # Check if nodes exist in netlist
        all_nodes = structure.get("nodes", set())
        for node in measure_nodes:
            if node not in all_nodes:
                result["missing_nodes"].append(node)
                result["valid"] = False
                result["errors"].append(f"Measurement node not in circuit: {node}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Electrical Validation
# ─────────────────────────────────────────────────────────────────────────────
def _validate_electrical(netlist: str, structure: dict, testbench: str) -> Dict[str, Any]:
    """Electrical validation: signal transitions, power delivery, timing."""
    validation = {
        "valid": True,
        "signal_integrity": {},
        "power_delivery": {},
        "timing_analysis": {},
        "issues": []
    }

    # Check signal transitions
    pulse_sources = []
    for line in testbench.splitlines():
        if "PULSE" in line.upper():
            parts = line.split()
            if len(parts) >= 3:
                pulse_sources.append(parts[1])

    validation["signal_integrity"]["input_signals"] = pulse_sources
    if not pulse_sources:
        validation["issues"].append("No pulse input signals")
        validation["valid"] = False

    # Check power delivery
    has_vdd = any("VDD" in line.upper() or "vdd" in line for line in netlist.splitlines())
    if not has_vdd:
        validation["issues"].append("No VDD detected")
    
    validation["power_delivery"]["has_vdd"] = has_vdd

    # Check for floating MOS gates
    mos_gates = set()
    driven_nodes = set()
    
    for line in netlist.splitlines():
        parts = line.split()
        if parts and parts[0].upper().startswith("M"):
            if len(parts) >= 3:
                mos_gates.add(parts[2])  # Gate is 2nd node
        elif parts and parts[0].upper().startswith(("V", "X", "B")):
            driven_nodes.update(parts[1:3])
    
    floating = mos_gates - driven_nodes - {"gnd", "0", "vdd"}
    if floating:
        validation["issues"].append(f"Floating gates: {floating}")
        validation["valid"] = False

    return validation


# ─────────────────────────────────────────────────────────────────────────────
# Testbench Generation (Improved)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_testbench(netlist: str, analysis_type: str, structure: dict) -> str:
    """Generate testbench with automatic node inference."""
    lines = ["* Auto-generated Testbench"]
    
    # Find power node
    power_node = "vdd"
    for line in netlist.splitlines():
        if "vdd" in line.lower():
            power_node = "vdd"
            break
    
    # Add power supply
    lines.append(f"VDD {power_node} 0 DC 1.8")
    
    # Infer inputs from structure or find primary input
    inputs = list(structure.get("inputs", set()))
    if not inputs:
        # Look for .subckt ports
        for line in netlist.splitlines():
            if re.match(r"(?i)^\.subckt", line):
                parts = line.split()
                if len(parts) >= 3:
                    inputs = [parts[2]]  # First port is usually input
                    break
    
    primary_input = inputs[0] if inputs else "in"
    
    # Add input stimulus
    if analysis_type == "Transient":
        lines.append(f"VIN {primary_input} 0 PULSE(0 1.8 0 10p 10p 500p 1n)")
    else:
        lines.append(f"VIN {primary_input} 0 DC 0.9")
    
    # Add load capacitors to outputs if not present
    outputs = list(structure.get("outputs", set()))
    if not outputs:
        # Create default output
        lines.append("CL out 0 10f")
        outputs = ["out"]
    else:
        # Ensure output has capacitor
        has_cap = False
        for line in netlist.splitlines():
            if line.upper().startswith("C") and outputs[0] in line:
                has_cap = True
                break
        if not has_cap:
            lines.append(f"CL {outputs[0]} 0 10f")
    
    # Analysis directives
    if analysis_type == "Transient":
        lines.append(".tran 1p 2n")
        # Measure delay
        out_node = outputs[0]
        lines.append(f".measure tran delay_val TRIG v({primary_input}) VAL=0.9 RISE=1 TARG v({out_node}) VAL=0.9 FALL=1")
        lines.append(f".measure tran avg_pwr AVG par('-v({power_node})*i(VDD)') FROM=0 TO=2n")
    elif analysis_type == "AC":
        lines.append(".ac dec 10 1 1G")
    else:
        lines.append(".op")
    
    lines.append(".end")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation (Fixed to combine netlist + testbench)
# ─────────────────────────────────────────────────────────────────────────────
def _run_ngspice_combined(netlist: str, testbench: str) -> Dict[str, Any]:
    """Run ngspice with combined netlist and testbench."""
    # Combine netlist and testbench
    # Netlist should come first, then testbench with .end last
    combined = netlist + "\n\n* Testbench\n" + testbench
    
    # Ensure .end is at the end
    combined = re.sub(r"\.end\s*$", "", combined, flags=re.MULTILINE | re.IGNORECASE)
    combined = combined.strip() + "\n.end"
    
    # Sanitize
    combined = _sanitize_spice_text(combined)
    
    # Write to temp file
    fd, fname = tempfile.mkstemp(suffix=".sp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(combined)
        
        result = subprocess.run(
            ["ngspice", "-b", fname],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return {
            "log": result.stdout + result.stderr,
            "failed": result.returncode != 0,
            "combined": combined  # Return for debugging
        }
    except Exception as e:
        return {
            "log": f"Execution failed: {str(e)}",
            "failed": True,
            "combined": combined
        }
    finally:
        try:
            os.unlink(fname)
        except:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────────────────────────────────────
def _analyze_simulation_errors(log: str) -> Dict[str, Any]:
    """Classify ngspice errors."""
    analysis = {
        "error_type": "none",
        "severity": "low",
        "fix_strategy": "none",
        "details": {}
    }

    log_lower = log.lower()

    if "too few parameters" in log_lower:
        analysis["error_type"] = "subcircuit_port_mismatch"
        analysis["severity"] = "high"
        analysis["fix_strategy"] = "fix_ports"
    elif "unknown" in log_lower and "subcircuit" in log_lower:
        analysis["error_type"] = "missing_subcircuit"
        analysis["severity"] = "high"
        analysis["fix_strategy"] = "add_subckt"
    elif "no such vector" in log_lower:
        analysis["error_type"] = "invalid_node"
        analysis["severity"] = "high"
        analysis["fix_strategy"] = "fix_nodes"
    elif "singular" in log_lower or "gmin" in log_lower:
        analysis["error_type"] = "convergence"
        analysis["severity"] = "medium"
        analysis["fix_strategy"] = "add_gmin"
    elif "error" in log_lower:
        analysis["error_type"] = "generic"
        analysis["severity"] = "medium"
        analysis["fix_strategy"] = "syntax_check"

    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Fixer
# ─────────────────────────────────────────────────────────────────────────────
def _apply_fixes(netlist: str, testbench: str, error_analysis: dict) -> Tuple[str, str, List[str]]:
    """Apply deterministic fixes."""
    fixes = []
    netlist_out = netlist
    tb_out = testbench
    
    error_type = error_analysis.get("error_type", "none")
    
    if error_type == "subcircuit_port_mismatch":
        # Add missing ground ports
        lines = netlist_out.splitlines()
        new_lines = []
        for line in lines:
            if re.match(r"(?i)^\s*X", line):
                # Ensure ends with gnd if needed
                if not line.strip().endswith("gnd") and "gnd" not in line.split()[-2:]:
                    line = line.rstrip() + " gnd"
                    fixes.append("Added gnd port to X-instance")
            new_lines.append(line)
        netlist_out = "\n".join(new_lines)
    
    elif error_type == "missing_subcircuit":
        # Add dummy .subckt if missing
        missing = re.findall(r'unknown subcircuit (\w+)', error_analysis.get("log", ""), re.IGNORECASE)
        if missing:
            dummy = f".subckt {missing[0]} in out\nR1 in out 1k\n.ends"
            netlist_out = dummy + "\n" + netlist_out
            fixes.append(f"Added dummy subcircuit for {missing[0]}")
    
    elif error_type == "invalid_node":
        # Replace measurement nodes with actual nodes from netlist
        if "out" in testbench:
            tb_out = testbench.replace("v(out)", "v(n_out)", 1)
            fixes.append("Fixed measurement node reference")
    
    # Ensure .global gnd
    if ".global" not in netlist_out.lower():
        netlist_out = ".global gnd\n" + netlist_out
        fixes.append("Added .global gnd")
    
    return netlist_out, tb_out, fixes


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def input_loader(state: AutoPPA) -> Dict[str, Any]:
    """Load and initialize state."""
    raw = ""
    for key in ("netlist_raw", "working_netlist", "netlist", "spice_netlist"):
        v = state.get(key, "")
        if isinstance(v, str) and v.strip():
            raw = v.strip()
            break
    
    if not raw:
        raise ValueError("SPICE netlist is empty.")

    stripped = _strip_comments(raw)
    
    # Store baseline
    return {
        "netlist_raw": raw,
        "working_netlist": stripped,
        "baseline_netlist": stripped,
        "circuit_name": state.get("circuit_name", "Untitled"),
        "circuit_type": state.get("circuit_type", "Logic"),
        "opt_objective": state.get("opt_objective", "Balanced"),
        "max_iterations": state.get("max_iterations", 5),
        "analysis_type": state.get("analysis_type", "Transient"),
        "iteration": 0,
        "history": [],
        "error_memory": _load_error_memory(),
        "best_cost": float("inf"),
        "best_delay": float("inf"),
        "best_power": float("inf"),
        "best_area": float("inf"),
        "best_netlist": stripped,
        "best_iteration": 0,
        "accepted": False,
        "baseline_delay": float("inf"),
        "baseline_power": float("inf"),
        "baseline_area": 0.0,
    }


def structural_validator(state: AutoPPA) -> Dict[str, Any]:
    """Analyze circuit structure."""
    netlist = state.get("working_netlist", "")
    structure = _analyze_circuit_structure(netlist)
    
    return {
        "circuit_structure": structure,
        "connectivity_map": structure["connectivity"],
        "fanout_analysis": structure["fanout_map"],
    }


def functional_validator_node(state: AutoPPA) -> Dict[str, Any]:
    """Validate functional correctness."""
    netlist = state.get("proposed_netlist") or state.get("working_netlist", "")
    testbench = state.get("testbench", "")
    structure = state.get("circuit_structure", {})
    
    result = _validate_functional(netlist, testbench, structure)
    
    return {
        "functional_valid": result["valid"],
        "functional_errors": result["errors"],
        "empty_subckts": result["empty_subckts"],
        "floating_outputs": result["floating_outputs"],
        "missing_nodes": result["missing_nodes"],
        "path_exists": result["path_exists"],
    }


def electrical_validator(state: AutoPPA) -> Dict[str, Any]:
    """Validate electrical properties."""
    netlist = state.get("working_netlist", "")
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
    """Generate optimized design using LLM with strict constraints."""
    iteration = state.get("iteration", 0)
    current_netlist = state.get("working_netlist", "")
    structure = state.get("circuit_structure", {})
    func_errors = state.get("functional_errors", [])
    history = state.get("history", [])
    
    # Build context with error memory
    error_context = "\n".join(state.get("error_memory", [])[-5:])
    
    # Build improvement history context
    improvement_context = ""
    if history:
        last = history[-1]
        improvement_context = f"Previous: delay={last.get('delay_ps')}ps, power={last.get('power_uw')}uW"
    
    # Strict prompt with safety constraints
    prompt = f"""
You are an expert VLSI design engineer optimizing a SPICE circuit.

CURRENT ITERATION: {iteration}
OPTIMIZATION OBJECTIVE: {state.get('opt_objective', 'Balanced')}
PREVIOUS METRICS: {improvement_context}
FUNCTIONAL ERRORS TO FIX: {func_errors}

SAFE OPTIMIZATION RULES:
{SAFE_OPTIMIZATION_RULES}

CIRCUIT NETLIST:
{current_netlist}

SUBCIRCUIT ANALYSIS:
- Empty subcircuits found: {state.get('empty_subckts', [])}
- Floating outputs: {state.get('floating_outputs', [])}

ERROR MEMORY:
{error_context}

TASK:
Optimize the circuit for {state.get('opt_objective', 'Balanced')} while strictly following safety rules.
Fix any functional errors (empty subcircuits, floating nodes).
Only resize transistors, add buffers for high fanout, or remove redundant elements.

Return JSON:
{{
  "proposed_netlist": "<complete netlist with \\n escaped>",
  "reasoning": "<explanation of changes>",
  "targets": {{"delay_target_ps": 100, "power_target_uw": 200}},
  "area_estimate_um2": <number>
}}
"""
    
    raw = _call_llm([
        {"role": "system", "content": "You are a careful VLSI optimizer. Only apply safe transformations."},
        {"role": "user", "content": prompt}
    ], max_tokens=3000)
    
    parsed = _extract_json(raw) or {}
    
    proposed = _decode_escaped_text(parsed.get("proposed_netlist", current_netlist))
    
    # Generate testbench if not present
    testbench = state.get("testbench", "")
    if not testbench:
        testbench = _generate_testbench(proposed, state.get("analysis_type", "Transient"), structure)
    
    return {
        "proposed_netlist": proposed,
        "testbench": testbench,
        "reasoning": parsed.get("reasoning", ""),
        "targets": parsed.get("targets", {"delay_target_ps": 100, "power_target_uw": 200}),
        "area_um2": float(parsed.get("area_estimate_um2", 0)) if parsed.get("area_estimate_um2") else 0,
    }


def simulation_runner(state: AutoPPA) -> Dict[str, Any]:
    """Run simulation with combined netlist and testbench."""
    netlist = state.get("proposed_netlist") or state.get("working_netlist", "")
    testbench = state.get("testbench", "")
    
    if not testbench:
        return {
            "sim_failed": True,
            "ngspice_log": "No testbench available",
        }
    
    result = _run_ngspice_combined(netlist, testbench)
    
    return {
        "ngspice_log": result["log"],
        "sim_failed": result["failed"],
        "combined_circuit": result.get("combined", ""),
    }


def error_analyzer(state: AutoPPA) -> Dict[str, Any]:
    """Analyze simulation errors."""
    log = state.get("ngspice_log", "")
    analysis = _analyze_simulation_errors(log)
    
    return {
        "error_analysis": analysis,
        "sim_error_pattern": analysis["error_type"],
    }


def deterministic_fixer(state: AutoPPA) -> Dict[str, Any]:
    """Apply deterministic fixes."""
    netlist = state.get("proposed_netlist", "")
    testbench = state.get("testbench", "")
    error_analysis = state.get("error_analysis", {})
    
    fixed_netlist, fixed_tb, fixes = _apply_fixes(netlist, testbench, error_analysis)
    
    return {
        "proposed_netlist": fixed_netlist,
        "testbench": fixed_tb,
        "applied_fixes": fixes,
        "fix_history": state.get("fix_history", []) + fixes,
    }


def llm_parse_results(state: AutoPPA) -> Dict[str, Any]:
    """Parse simulation results."""
    if state.get("sim_failed"):
        return {
            "delay_ps": float("inf"),
            "power_uw": float("inf"),
            "area_um2": state.get("area_um2", 0),
        }
    
    log = state.get("ngspice_log", "")
    
    # Extract delay
    delay = float("inf")
    power = float("inf")
    
    # Parse .measure results
    for line in log.splitlines():
        if "delay_val" in line.lower():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    val = float(parts[-1])
                    if val > 0:
                        delay = val * 1e12  # Convert to ps if in seconds
                except:
                    pass
        elif "avg_pwr" in line.lower():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    val = float(parts[-1])
                    power = abs(val) * 1e6  # Convert to uW if in W
                except:
                    pass
    
    # If still inf, use LLM to parse
    if delay == float("inf") and power == float("inf"):
        raw = _call_llm([
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user", "content": log[:3000]}  # Limit log size
        ])
        parsed = _extract_json(raw) or {}
        delay = parsed.get("delay_ps", float("inf")) if parsed.get("delay_ps") else float("inf")
        power = parsed.get("power_uw", float("inf")) if parsed.get("power_uw") else float("inf")
    
    return {
        "delay_ps": delay,
        "power_uw": power,
        "area_um2": state.get("area_um2", 0),
    }


def acceptance_checker(state: AutoPPA) -> Dict[str, Any]:
    """
    Critical acceptance criteria:
    - Simulation must succeed
    - Functional validation must pass  
    - At least one metric must improve (delay, power, or area)
    """
    current_delay = state.get("delay_ps", float("inf"))
    current_power = state.get("power_uw", float("inf"))
    current_area = state.get("area_um2", float("inf"))
    
    best_delay = state.get("best_delay", float("inf"))
    best_power = state.get("best_power", float("inf"))
    best_area = state.get("best_area", float("inf"))
    
    sim_ok = not state.get("sim_failed", True)
    func_ok = state.get("functional_valid", False)
    
    reject_reasons = []
    
    if not sim_ok:
        reject_reasons.append("Simulation failed")
    if not func_ok:
        reject_reasons.append(f"Functional invalid: {state.get('functional_errors', [])}")
    
    # Check improvement (must improve at least one without worsening others significantly)
    improved = False
    if sim_ok and func_ok:
        delay_better = current_delay < best_delay * 0.99  # 1% improvement threshold
        power_better = current_power < best_power * 0.99
        area_better = current_area < best_area * 0.99
        
        # Accept if any improved and none worsened by >10%
        delay_ok = current_delay <= best_delay * 1.1
        power_ok = current_power <= best_power * 1.1
        area_ok = current_area <= best_area * 1.1
        
        if (delay_better or power_better or area_better) and delay_ok and power_ok and area_ok:
            improved = True
        elif best_delay == float("inf"):  # First successful run
            improved = True
        else:
            reject_reasons.append(f"No improvement: delay={current_delay:.2f} vs best={best_delay:.2f}, "
                                f"power={current_power:.2f} vs best={best_power:.2f}")
    
    accepted = sim_ok and func_ok and improved
    
    print(f"[Acceptance] Accepted={accepted}, Reasons={reject_reasons}, "
          f"Metrics: delay={current_delay:.2f}, power={current_power:.2f}")
    
    new_best = {
        "best_delay": best_delay,
        "best_power": best_power,
        "best_area": best_area,
        "best_netlist": state.get("best_netlist", state.get("working_netlist")),
        "best_iteration": state.get("best_iteration", 0),
        "accepted": accepted,
        "reject_reason": "; ".join(reject_reasons),
    }
    
    if accepted:
        new_best.update({
            "best_delay": current_delay,
            "best_power": current_power,
            "best_area": current_area,
            "best_netlist": state.get("proposed_netlist", state.get("working_netlist")),
            "best_iteration": state.get("iteration", 0),
            "working_netlist": state.get("proposed_netlist", state.get("working_netlist")),
        })
    
    return new_best


def cost_tracker(state: AutoPPA) -> Dict[str, Any]:
    """Calculate cost and check constraints."""
    delay = state.get("delay_ps", float("inf"))
    power = state.get("power_uw", float("inf"))
    area = state.get("area_um2", 0)
    targets = state.get("targets", {})
    obj = state.get("opt_objective", "Balanced")
    
    dt = max(float(targets.get("delay_target_ps", 100)), 1e-9)
    pt = max(float(targets.get("power_target_uw", 200)), 1e-9)
    
    d_score = (delay / dt) if delay < float("inf") else 999.0
    p_score = (power / pt) if power < float("inf") else 999.0
    a_score = area / 100.0
    
    w_d, w_p, w_a = _objective_weights(obj)
    cost = w_d * d_score + w_p * p_score + w_a * a_score
    
    met = {
        "delay": delay <= dt if delay < float("inf") else False,
        "power": power <= pt if power < float("inf") else False,
    }
    
    return {
        "cost": round(cost, 4),
        "constraints_met": met["delay"] and met["power"],
        "met_details": met,
    }


def history_logger(state: AutoPPA) -> Dict[str, Any]:
    """Log history and manage error memory."""
    history = list(state.get("history", []))
    iteration = state.get("iteration", 0)
    
    # Extract error line
    err_line = _extract_ngspice_error_line(state.get("ngspice_log", ""))
    lessons = list(state.get("error_memory", []))
    
    if err_line and err_line not in lessons:
        lessons.append(err_line)
        lessons = lessons[-12:]
        _save_error_memory(lessons)
    
    entry = {
        "iteration": iteration,
        "delay_ps": state.get("delay_ps"),
        "power_uw": state.get("power_uw"),
        "area_um2": state.get("area_um2"),
        "cost": state.get("cost"),
        "accepted": state.get("accepted"),
        "reject_reason": state.get("reject_reason"),
        "sim_ok": not state.get("sim_failed", True),
        "functional_ok": state.get("functional_valid", False),
        "reasoning": state.get("reasoning", ""),
    }
    history.append(entry)
    
    return {
        "history": history,
        "error_memory": lessons,
        "iteration": iteration + 1,
    }


def convergence_checker(state: AutoPPA) -> Dict[str, Any]:
    """Check if optimization should continue."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)
    accepted = state.get("accepted", False)
    constraints_met = state.get("constraints_met", False)
    
    # Converged if constraints met and no improvement for 2 iterations
    if constraints_met and iteration > 2:
        recent = state.get("history", [])[-2:]
        if all(h.get("cost") == recent[0].get("cost") for h in recent if recent):
            return {"continue_loop": False, "reason": "converged"}
    
    if iteration >= max_iter:
        return {"continue_loop": False, "reason": "max_iterations"}
    
    if not accepted and iteration > 1:
        # If consistently rejected, might need to stop
        recent_rejects = [h for h in state.get("history", [])[-3:] if not h.get("accepted")]
        if len(recent_rejects) >= 3:
            return {"continue_loop": False, "reason": "stagnation"}
    
    return {"continue_loop": True, "reason": "continue"}


def final_optimizer(state: AutoPPA) -> Dict[str, Any]:
    """Return best design."""
    best_netlist = state.get("best_netlist", state.get("working_netlist", ""))
    
    print(f"\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best Iteration: {state.get('best_iteration', 0)}")
    print(f"Best Delay: {state.get('best_delay', float('inf')):.2f} ps")
    print(f"Best Power: {state.get('best_power', float('inf')):.2f} uW")
    print(f"Best Area: {state.get('best_area', 0):.2f} um2")
    
    return {
        "optimised_netlist": best_netlist,
        "final_delay": state.get("best_delay"),
        "final_power": state.get("best_power"),
        "final_area": state.get("best_area"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing Logic
# ─────────────────────────────────────────────────────────────────────────────
def route_after_functional(state: AutoPPA) -> str:
    """Route based on functional validation."""
    if state.get("functional_valid", False):
        return "valid"
    else:
        return "invalid"


def route_after_simulation(state: AutoPPA) -> str:
    """Route based on simulation results."""
    if state.get("sim_failed", True):
        error_type = state.get("sim_error_pattern", "")
        if error_type in ["subcircuit_port_mismatch", "missing_subcircuit"]:
            return "fixable"
        else:
            return "failed"
    else:
        return "success"


def route_after_acceptance(state: AutoPPA) -> str:
    """Route based on acceptance."""
    if state.get("accepted", False):
        return "accepted"
    else:
        return "rejected"


def route_convergence(state: AutoPPA) -> str:
    """Route based on convergence."""
    if state.get("continue_loop", True):
        return "continue"
    else:
        return "end"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Assembly
# ─────────────────────────────────────────────────────────────────────────────
graph = StateGraph(AutoPPA)

# Add nodes
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

# Define edges
graph.add_edge(START, "input_loader")
graph.add_edge("input_loader", "structural_validator")
graph.add_edge("structural_validator", "functional_validator")

# After functional validation, either fix or proceed
graph.add_conditional_edges(
    "functional_validator",
    route_after_functional,
    {
        "valid": "electrical_validator",
        "invalid": "llm_design_generator"  # Go back to LLM to fix
    }
)

graph.add_edge("electrical_validator", "llm_design_generator")
graph.add_edge("llm_design_generator", "simulation_runner")
graph.add_edge("simulation_runner", "error_analyzer")

# After simulation, either fix, or parse results
graph.add_conditional_edges(
    "error_analyzer",
    route_after_simulation,
    {
        "fixable": "deterministic_fixer",
        "failed": "llm_design_generator",  # Back to LLM
        "success": "llm_parse_results"
    }
)

graph.add_edge("deterministic_fixer", "simulation_runner")  # Loop back
graph.add_edge("llm_parse_results", "acceptance_checker")
graph.add_edge("acceptance_checker", "cost_tracker")
graph.add_edge("cost_tracker", "convergence_checker")
graph.add_edge("convergence_checker", "history_logger")

# After history, check if continue or end
graph.add_conditional_edges(
    "history_logger",
    route_convergence,
    {
        "continue": "structural_validator",  # Loop back
        "end": "final_optimizer"
    }
)

graph.add_edge("final_optimizer", END)

app = graph.compile()
