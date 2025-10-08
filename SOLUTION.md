# EXO Labs: 2‑Day Work‑Task: Solution

This solution implements two functions as requested in the task.
It extends get_transition_events by adding three transitions: model activated/deactivated and model atomically replaced. It also adds UTs to test this behaviour.

It extends the function get_instance_placements by implementing NP-hard algorithm for multi-resource job allocation. This algorithm checks in which order on which machine we can schedule the model inference  get the lowest latency.
More details in the docstring of the function get_instance_placements_snapshot.
It should be noted that this model can be extended by accounting for timings of job computations (if not all the parts of the model are constantly in the compute phase) or if so, to proportionally divide compute/memory/network bandwidths with already active models.

Future work: several heuristics can be implemented to speed up the search.

The UTs have been extended to support this novel algorithm.