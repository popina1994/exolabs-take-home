# EXO Labs: 2‑Day Work‑Task (Vlad)

## Objective
Design and implement a distributed inference planner that allocates model shards to the heterogeneous nodes in an EXO cluster so that total end‑to‑end inference latency is minimised.

## Background (why this matters)
When EXO receives an inference request it must split the model into shards and run them in parallel. Nodes differ in compute power, memory and network bandwidth. Allocating the right shard to the right node – while respecting memory limits and noisy network links – is therefore critical to achieve interactive‑speed inference.

## Scope of Work
You will:

1. Implement a new `get_instance_placements` function.

    Here is the type signature of `get_instance_placements` (it should be self explanatory):
    ```python
    def get_instance_placements(
        command: CreateInstanceCommand,
        topology: TopologySnapshot,
        current_instances: dict[InstanceId, Instance],
    ) -> dict[InstanceId, Instance]:
    ```


2. Emit transition events via `get_transition_events` so workers know how to reach the new plan.

    Here is the type signature of `get_transition_events` (it should be self explanatory):
    ```python
    def get_transition_events(
        current_instances: dict[InstanceId, Instance],
        target_instances: dict[InstanceId, Instance],
    ) -> Sequence[Event]:
    ```

3. Write a short README describing your approach, trade‑offs and a 30‑second demo script.

## Suggested Steps

- There is already a basic implementation in [placement.py](/placement.py). You should start by understanding what it does and what its limitations are. You can run the tests in [test_placement.py](/tests/test_placement.py).
- The tests are intentionally very simple, and you will have to update them as you change the behaviour of these functions, don't take them as gospel.
