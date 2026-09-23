#!/usr/bin/env python3
"""Summarize P0 pad residence times; these are wall times, not GPU kernel times."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path


def distribution(values):
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {"n": len(values), "mean": sum(values) / len(values),
            **{f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)]
               for p in (50, 95, 99)}, "max": values[-1]}


def paired(stages, first, last, begin, end, end_field="start_ns"):
    def index(name, field):
        indexed = collections.defaultdict(list)
        for row in stages.get(name, []):
            if row["pts"] != 2**64 - 1:
                indexed[row["pts"]].append(row[field])
        return indexed
    starts, ends = index(first, "start_ns"), index(last, end_field)
    delays, missing, ambiguous, reversed_count = [], 0, 0, 0
    for pts, times in starts.items():
        if not any(begin <= t <= end for t in times):
            continue
        finish = ends.get(pts, [])
        if not finish:
            missing += 1
        elif len(times) != 1 or len(finish) != 1:
            ambiguous += 1
        elif finish[0] < times[0]:
            reversed_count += 1
        else:
            delays.append((finish[0] - times[0]) / 1e6)
    return {"from": first, "to": last, "wall_ms": distribution(delays),
            "unmatched": missing, "ambiguous": ambiguous, "reversed": reversed_count}


def analyze(directory, gpu_candidate=False, visual_only=False):
    with (directory / "trace.csv").open() as source:
        header = source.readline().strip()
        if header != "# dropped=0":
            raise ValueError(f"Trace is incomplete: {header}")
        stages = collections.defaultdict(list)
        for record in csv.DictReader(source):
            stages[record.pop("stage")].append({k: int(v) for k, v in record.items()})
    samples = [] if gpu_candidate or visual_only else [json.loads(line) for line in (directory / "snapshots.jsonl").read_text().splitlines()]
    if gpu_candidate or visual_only:
        results = stages.get("gpu_candidate/result" if gpu_candidate else "nvinfer/primary-infer/src", [])
        if not results:
            raise ValueError("No visual result timestamps")
        begin, end = results[0]["start_ns"] + 2_000_000_000, results[-1]["start_ns"]
    else:
        begin, end = samples[0]["observed_ns"] + 2_000_000_000, samples[-1]["observed_ns"]
    if end <= begin:
        raise ValueError("Not enough steady-state samples")
    stage_info = {}
    for name, rows in stages.items():
        selected = [r for r in rows if begin <= r["start_ns"] <= end]
        info = {"samples": len(selected), "events_per_second": len(selected) * 1e9 / (end - begin)}
        if selected and name.startswith("cpu_parser/"):
            info["cpu_parser_wall_ms"] = distribution([(r["end_ns"] - r["start_ns"]) / 1e6 for r in selected])
            info["candidate_count"] = distribution([r["auxiliary"] for r in selected])
        elif selected and name.startswith("nvinfer/") and name.endswith("/src"):
            info["detection_count"] = distribution([r["auxiliary"] for r in selected])
        elif selected and name == "gpu_candidate/result":
            info["detection_count"] = distribution([r["auxiliary"] for r in selected])
            info["image_ready_to_result_wall_ms"] = distribution([(r["end_ns"] - r["start_ns"]) / 1e6 for r in selected])
        elif selected and name.startswith("v4l2src/"):
            info["buffer_bytes"] = distribution([r["auxiliary"] for r in selected])
            info["zero_size_buffers"] = sum(r["auxiliary"] == 0 for r in selected)
        stage_info[name] = info
    # jpegparse retimestamps frames; nvstreammux uses different batch/frame PTS.
    # Report only joins within the same timestamp domain, never nearest-time joins.
    pairs = [("capture_to_detection_unresolved", "v4l2src/capture-source/src", "nvinfer/primary-infer/src"),
             ("jpeg_parsed_to_detection", "jpegparse/jpegparse0/src", "nvinfer/primary-infer/src"),
             ("infer_input_to_terminal_sink", "nvinfer/primary-infer/sink", "fakesink/deepstream-sink/sink")]
    if gpu_candidate:
        pairs = []
    for name in stages:
        if name.endswith("/sink") or name.endswith("/sink_0"):
            base = name.rsplit("/", 1)[0]
            if base + "/src" in stages:
                pairs.append((base, name, base + "/src"))
    segments = {label: paired(stages, first, last, begin, end) for label, first, last in pairs}
    if gpu_candidate:
        segments["jpeg_parsed_to_detection"] = paired(stages, "jpegparse/jpegparse0/src",
            "gpu_candidate/result", begin, end, end_field="end_ns")
    initial, final = (samples[0]["snapshot"], samples[-1]["snapshot"]) if samples else ({}, {})
    counters = {}
    for name, value in final.get("pipeline_metrics", {}).items():
        before = initial.get("pipeline_metrics", {}).get(name)
        if type(value) is int and type(before) is int:
            counters[name] = value - before
    return {"directory": directory.name, "gpu_capture_candidate": gpu_candidate,
            "visual_only_baseline": visual_only,
            "window_seconds": (end - begin) / 1e9,
            "window_monotonic_ns": [begin, end], "stages": stage_info,
            "segments": segments, "runtime_counter_delta": counters,
            "counter_window_seconds": (samples[-1]["observed_ns"] - samples[0]["observed_ns"]) / 1e9 if samples else None,
            "physical_output_closed": all(s["snapshot"]["pipeline_metrics"]["output_gate_open"] is False
                                          and s["snapshot"]["pipeline_metrics"]["device_receipts"] == 0 for s in samples) if samples else None}


def self_test():
    assert distribution([4, 1, 3, 2])["p50"] == 2
    def row(pts, ns):
        return {"pts": pts, "start_ns": ns}
    stages = {"a": [row(1, 1_000_000), row(2, 2_000_000), row(3, 3_000_000), row(4, 4_000_000)],
              "b": [row(1, 3_000_000), row(3, 4_000_000), row(3, 5_000_000), row(4, 1_000_000)]}
    result = paired(stages, "a", "b", 0, 10_000_000)
    assert result["wall_ms"]["n"] == 1 and result["wall_ms"]["p50"] == 2
    assert result["unmatched"] == result["ambiguous"] == result["reversed"] == 1
    span = {"a": [row(1, 1_000_000)], "b": [dict(row(1, 3_000_000), end_ns=5_000_000)]}
    assert paired(span, "a", "b", 0, 10_000_000, end_field="end_ns")["wall_ms"]["p50"] == 4
    print("P0_ANALYZER_SELF_TEST_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--self-test", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gpu-candidate", action="store_true")
    mode.add_argument("--visual-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.directory is None:
        parser.error("directory is required")
    else:
        report = analyze(args.directory, args.gpu_candidate, args.visual_only)
        output = args.directory / "analysis.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(output)
