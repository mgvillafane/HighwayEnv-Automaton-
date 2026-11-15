# Training versus evaluation

The selected 53-input timeout/dropout run had 1 FRONT and 1 RIGHT training success. Its deterministic evaluations had zero successes for both goals. The updated graph separates these measurements; the evaluation-only graph is preserved.

The separate older 57-input variants had up to 18.2% RIGHT evaluation success, but no FRONT evaluation success. See timeout_dropout_audit.csv for the broader audit.

Regenerate the updated graph with plot_verified_success_rates.py. plot_success_rates.py regenerates the original evaluation-only view.
