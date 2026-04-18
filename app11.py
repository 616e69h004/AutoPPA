#!/usr/bin/env python3
"""
AutoPPA App for model11
=======================
Streamlit interface for the model11 closed-loop engineering optimizer.

Run with:
  streamlit run app11.py
"""

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from autoPPA.AutoPPA.model11 import app as model11_app


def _sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_result(item: Any) -> Any:
    if isinstance(item, str):
        return _sanitize_text(item)
    if isinstance(item, list):
        return [_sanitize_result(x) for x in item]
    if isinstance(item, dict):
        return {k: _sanitize_result(v) for k, v in item.items()}
    return item


def _save_results(output_file: Path, results: Dict[str, Any]) -> None:
    output = {
        "constraints_met": results.get("constraints_met", False),
        "circuit_name": _sanitize_text(results.get("circuit_name", "")),
        "circuit_type": _sanitize_text(results.get("circuit_type", "")),
        "analysis_type": _sanitize_text(results.get("analysis_type", "")),
        "best_delay_ps": results.get("best_delay"),
        "best_power_uw": results.get("best_power"),
        "best_area_um2": results.get("best_area"),
        "min_cost": results.get("min_cost"),
        "convergence_score": results.get("convergence_score"),
        "optimised_netlist": _sanitize_text(results.get("optimised_netlist", "")),
        "applied_fixes": _sanitize_result(results.get("applied_fixes", [])),
        "fix_history": _sanitize_result(results.get("fix_history", [])),
        "met_details": _sanitize_result(results.get("met_details", {})),
        "circuit_structure": _sanitize_result(results.get("circuit_structure", {})),
        "signal_integrity": _sanitize_result(results.get("signal_integrity", {})),
        "power_delivery": _sanitize_result(results.get("power_delivery", {})),
        "timing_analysis": _sanitize_result(results.get("timing_analysis", {})),
        "constraint_status": _sanitize_result(results.get("constraint_status", {})),
        "history": _sanitize_result(results.get("history", [])),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _save_netlist(output_file: Path, netlist: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(netlist)


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value == float("inf") or value > 1e9:
            return "N/A"
        return f"{value:.2f}"
    return str(value)


def run_model11(
    netlist: str,
    circuit_name: str,
    circuit_type: str,
    description: str,
    analysis_type: str,
    opt_objective: str,
    max_iterations: int,
) -> Dict[str, Any]:
    result = model11_app.invoke({
        "netlist_raw": netlist,
        "working_netlist": netlist,
        "circuit_name": circuit_name,
        "circuit_type": circuit_type,
        "prompt": "",
        "brief_description": description,
        "analysis_type": analysis_type,
        "opt_objective": opt_objective,
        "max_iterations": max_iterations,
    })
    return _sanitize_result(result)


def main() -> None:
    st.set_page_config(
        page_title="AutoPPA model11",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚡ AutoPPA model11")
    st.markdown("*Closed-loop SPICE optimization with structural, electrical, and functional validation.*")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")

        circuit_type = st.selectbox(
            "Circuit Type",
            ["Logic", "Amplifier", "Mixed", "Memory", "Analog"],
            index=0,
        )

        analysis_type = st.selectbox(
            "Analysis Type",
            ["Transient", "AC Analysis", "DC Analysis", "Operating Point"],
            index=0,
        )

        opt_objective = st.selectbox(
            "Optimization Objective",
            ["Balanced PPA", "Minimum Delay", "Maximum Performance", "Minimum Power"],
            index=0,
        )

        max_iterations = st.slider(
            "Max Iterations",
            min_value=1,
            max_value=15,
            value=5,
            help="Maximum number of closed-loop optimize/simulate/fix cycles.",
        )

        st.header("💾 Output Options")
        save_results = st.checkbox("Save Results to JSON", value=True)
        save_netlist = st.checkbox("Save Optimized Netlist", value=True)

        if save_results or save_netlist:
            output_dir = st.text_input(
                "Output Directory",
                value="./model11_output",
            )
        else:
            output_dir = "./model11_output"

    st.header("📝 Input")
    input_method = st.radio(
        "Input Method",
        ["Text Input", "File Upload"],
        horizontal=True,
    )

    netlist = ""
    circuit_name = "model11_circuit"

    if input_method == "Text Input":
        netlist = st.text_area(
            "SPICE Netlist",
            height=360,
            placeholder="Paste your SPICE netlist here...",
        )
        circuit_name = st.text_input("Circuit Name", value="My Circuit")
    else:
        uploaded_file = st.file_uploader(
            "Upload SPICE File",
            type=["sp", "spice", "cir", "txt"],
        )
        if uploaded_file is not None:
            netlist = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            circuit_name = Path(uploaded_file.name).stem
            st.success(f"Loaded {uploaded_file.name}")

    description = st.text_input(
        "Brief Description",
        value="",
        help="Optional description used when the optimizer needs more design intent.",
    )

    if st.button("🚀 Run model11 Optimization", type="primary"):
        if not netlist.strip():
            st.error("Please provide a valid SPICE netlist before running.")
        else:
            with st.spinner("Running model11 closed-loop optimization... please wait"):
                try:
                    result = run_model11(
                        netlist=netlist,
                        circuit_name=circuit_name,
                        circuit_type=circuit_type,
                        description=description,
                        analysis_type=analysis_type,
                        opt_objective=opt_objective,
                        max_iterations=max_iterations,
                    )
                    st.session_state["model11_result"] = result
                except Exception as exc:
                    st.error(f"Error running model11: {_sanitize_text(str(exc))}")
                    return
            st.success("model11 optimization complete!")
            st.rerun()

    if "model11_result" in st.session_state:
        result = st.session_state["model11_result"]

        st.markdown("---")
        st.header("📊 Results")

        cols = st.columns(4)
        cols[0].metric("Status", "✅ Success" if result.get("constraints_met") else "❌ Incomplete")
        cols[1].metric("Iterations", result.get("iteration", 0))
        cols[2].metric("Delay (ps)", _format_metric(result.get("best_delay")))
        cols[3].metric("Power (uW)", _format_metric(result.get("best_power")))

        cols = st.columns(4)
        cols[0].metric("Area (um²)", _format_metric(result.get("best_area")))
        cols[1].metric("Cost", _format_metric(result.get("min_cost")))
        cols[2].metric("Convergence", _format_metric(result.get("convergence_score")))
        cols[3].metric("Circuit Type", result.get("circuit_type", ""))

        st.subheader("Validation Summary")
        vcols = st.columns(3)
        structure = result.get("circuit_structure", {})
        structural_ok = len(structure.get("issues", [])) == 0
        electrical_ok = result.get("electrical_valid", False)
        functional_ok = result.get("functional_valid", False)
        vcols[0].metric("Structural", "✅ Valid" if structural_ok else "❌ Issues")
        vcols[1].metric("Electrical", "✅ Valid" if electrical_ok else "❌ Issues")
        vcols[2].metric("Functional", "✅ Valid" if functional_ok else "❌ Issues")

        if result.get("met_details"):
            with st.expander("Constraint Status", expanded=True):
                st.json(result.get("met_details", {}))

        if structure:
            with st.expander("Circuit Structure", expanded=False):
                st.json({
                    "issues": structure.get("issues", []),
                    "subcircuits": structure.get("subcircuits", {}),
                    "fanout_map": structure.get("fanout_map", {}),
                    "load_map": structure.get("load_map", {}),
                    "timing_paths": structure.get("timing_paths", []),
                })

        if result.get("applied_fixes") or result.get("fix_history"):
            with st.expander("Issues / Fixes Applied", expanded=False):
                if result.get("applied_fixes"):
                    st.write("Latest fixes:")
                    for item in result.get("applied_fixes", []):
                        st.write(f"- {item}")
                if result.get("fix_history"):
                    st.write("Full fix history:")
                    for item in result.get("fix_history", []):
                        st.write(f"- {item}")

        if result.get("optimised_netlist"):
            st.subheader("🔧 Optimized Netlist")
            st.code(result.get("optimised_netlist"), language="spice")
            st.download_button(
                label="Download Optimized Netlist",
                data=result.get("optimised_netlist"),
                file_name=f"{circuit_name}_model11_optimized.sp",
                mime="text/plain",
                use_container_width=True,
            )

        if result.get("history"):
            with st.expander("Iteration History", expanded=False):
                st.json(result.get("history"))

        if result.get("ngspice_log"):
            with st.expander("Simulation Log", expanded=False):
                st.code(result.get("ngspice_log"), language="text")

        if save_results or save_netlist:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            if save_results:
                _save_results(output_path / f"{circuit_name}_model11_results.json", result)
            if save_netlist and result.get("optimised_netlist"):
                _save_netlist(output_path / f"{circuit_name}_model11_optimized.sp", result.get("optimised_netlist", ""))
            st.success(f"Saved outputs to {output_path.resolve()}")


if __name__ == "__main__":
    main()
