import math

import pytest

import model11


def test_mos_gate_is_not_treated_as_conductive_path():
    graph = model11._build_node_graph("M1 out inp 0 0 nmos W=1u L=0.18u")
    assert not model11._has_path(graph, "inp", "out")


def test_passive_connection_is_detected():
    graph = model11._build_node_graph("R1 input output 1k")
    assert model11._has_path(graph, "input", "output")


def test_spice_suffix_area_is_reported_as_proxy_um2():
    area = model11._estimate_area_from_netlist(
        "M1 out inp 0 0 nmos W=1u L=0.18u"
    )
    assert area == pytest.approx(0.18)


def test_area_is_unavailable_when_dimensions_are_not_explicit():
    assert model11._estimate_area_from_netlist("M1 out inp 0 0 nmos") is None


def test_generic_mos_fallback_is_disabled():
    source = ".include 'process_models.lib'\nM1 out in 0 0 nch"
    updated, added = model11._inject_fallback_models(source, ["nmos"])
    assert updated == source
    assert added == []


def test_golden_measurements_pass_within_tolerance():
    tb = ".measure tran golden_out_high FIND v(out) AT=1.5n"
    result = model11._evaluate_golden_measurements(
        tb, "golden_out_high = 1.7995e+00",
        {"golden_out_high": 1.8},
    )
    assert result["passed"] is True
    assert result["comparisons"]["golden_out_high"]["passed"] is True


def test_golden_measurements_fail_when_output_mismatches():
    tb = ".measure tran golden_out_high FIND v(out) AT=1.5n"
    result = model11._evaluate_golden_measurements(
        tb, "golden_out_high = 1.2e+00",
        {"golden_out_high": 1.8},
    )
    assert result["passed"] is False


def test_golden_measurements_fail_when_ngspice_does_not_report_value():
    tb = ".measure tran golden_out_high FIND v(out) AT=1.5n"
    result = model11._evaluate_golden_measurements(
        tb, "transient simulation complete",
        {"golden_out_high": 1.8},
    )
    assert result["passed"] is False
    assert result["missing_measure_results"] == ["golden_out_high"]


def test_functional_topology_does_not_claim_transistor_equivalence():
    netlist = "M1 out inp 0 0 nch W=1u L=0.18u"
    tb = "\n".join([
        "VDD vdd 0 DC 1.8",
        "VIN inp 0 PULSE(0 1.8 0 10p 10p 500p 1n)",
        ".measure tran golden_out FIND v(out) AT=1.5n",
    ])
    structure = model11._analyze_circuit_structure(netlist)
    result = model11._validate_functional(netlist, tb, structure)
    assert result["validation_level"] == "structural_only"
    assert any("do not prove" in warning for warning in result["warnings"])


def test_minimum_delay_candidate_cannot_regress_power_beyond_tolerance():
    result = model11.acceptance_checker({
        "sim_failed": False,
        "functional_valid": True,
        "golden_validation_passed": True,
        "opt_objective": "Minimum Delay",
        "delay_ps": 90.0,
        "power_uw": 10.5,
        "area_um2": math.inf,
        "best_delay": 100.0,
        "best_power": 10.0,
        "best_area": math.inf,
        "working_netlist": "baseline",
        "proposed_netlist": "candidate",
    })
    assert result["accepted"] is False


def test_candidate_without_golden_validation_is_rejected():
    result = model11.acceptance_checker({
        "sim_failed": False,
        "functional_valid": True,
        "golden_validation_passed": False,
        "opt_objective": "Minimum Delay",
        "delay_ps": 90.0,
        "power_uw": 9.0,
        "area_um2": math.inf,
        "best_delay": 100.0,
        "best_power": 10.0,
        "best_area": math.inf,
        "working_netlist": "baseline",
        "proposed_netlist": "candidate",
    })
    assert result["accepted"] is False


def test_input_loader_requires_testbench_and_golden_reference():
    with pytest.raises(ValueError, match="testbench"):
        model11.input_loader({"netlist_raw": "M1 out in 0 0 nch"})


def test_input_loader_checks_named_golden_measure_declarations():
    tb = "\n".join([
        "VIN in 0 PULSE(0 1.8 0 10p 10p 500p 1n)",
        ".measure tran delay_val TRIG v(in) VAL=0.9 RISE=1 TARG v(out) VAL=0.9 FALL=1",
        ".measure tran avg_pwr AVG par('-v(vdd)*i(VDD)') FROM=0 TO=2n",
    ])
    with pytest.raises(ValueError, match="golden"):
        model11.input_loader({
            "netlist_raw": "M1 out in 0 0 nch W=1u L=0.18u",
            "testbench": tb,
            "golden_expectations": {"golden_out": 1.8},
        })
