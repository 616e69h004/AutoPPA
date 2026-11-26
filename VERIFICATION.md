# AutoPPA verification inputs

AutoPPA's PPA loop requires a reviewed, circuit-specific NGSpice testbench. It no longer guesses the supply voltage, input pins, output load, timing edge, or measurement window.

## Required inputs

1. The circuit netlist, including the correct PDK model-library `.include` or `.lib`.
2. A testbench that drives the actual inputs and defines:
   - `.measure tran delay_val ...`
   - `.measure tran avg_pwr ...`
   - one or more `.measure tran golden_<checkpoint> ...` statements.
3. A JSON object mapping each `golden_*` measurement name to its expected numeric result.

Example testbench structure (adapt the voltage levels, timing, loads, pin names and measurement points to the actual design and process):

```spice
VDD vdd 0 DC 1.8
VIN in 0 PULSE(0 1.8 0 10p 10p 500p 1n)
CLOAD out 0 10f

.tran 1p 2n
.measure tran delay_val TRIG v(in) VAL=0.9 RISE=1 TARG v(out) VAL=0.9 FALL=1
.measure tran avg_pwr AVG par('-v(vdd)*i(VDD)') FROM=0 TO=2n
.measure tran golden_out_high FIND v(out) AT=1.5n
.measure tran golden_out_low FIND v(out) AT=0.5n
.end
```

Corresponding golden reference JSON (values shown are illustrative; measure these on the trusted reference circuit under the exact same test conditions):

```json
{
  "golden_out_high": 1.8,
  "golden_out_low": 0.0
}
```

Run the reference/golden circuit first and use its reported `golden_*` measurement values. AutoPPA runs the same checkpoints against every candidate and rejects candidates whose results are missing or outside the configured default tolerance (relative 0.1%, with a 1e-6 absolute floor).

## PPA interpretation

- Delay and power are simulator measurements parsed from NGSpice output.
- The reported area proxy is `sum(W*L)`, converted to square micrometres when every MOS width/length is an explicit numeric SPICE value. It is not standard-cell area or layout area.
- Generic fallback MOS cards are disabled. Missing process models must be fixed by supplying the correct PDK models; a fabricated Level-1 model must not be used to claim PPA.
- Static graph checks are structural diagnostics, not formal equivalence. Golden-response simulation checks are mandatory for candidate acceptance.
- The current optimization mode is transient delay/power. Physical-area optimization should only be enabled after connecting a synthesis or layout area source.
