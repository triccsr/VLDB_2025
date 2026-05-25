#!/usr/bin/env python3
import argparse
import heapq
import os
import tempfile


def event_key(parts: list[str]) -> tuple[int, int]:
    op_key = 0 if parts[2] == "+1" else 1
    return int(parts[3]), op_key


def write_sorted_chunk(events: list[list[str]], chunk_path: str) -> None:
    events.sort(key=event_key)
    with open(chunk_path, "w", encoding="utf-8") as fout:
        for u, v, op, ts in events:
            op_key = 0 if op == "+1" else 1
            fout.write(f"{ts} {op_key} {u} {v} {op} {ts}\n")


def iter_chunk(chunk_path: str):
    with open(chunk_path, "r", encoding="utf-8") as fin:
        for line in fin:
            ts, op_key, u, v, op, orig_ts = line.split()
            yield int(ts), int(op_key), u, v, op, orig_ts


def sort_dynamic_graph(input_path: str, output_path: str, chunk_size: int) -> None:
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
        header = fin.readline()
        if header == "":
            raise ValueError("input file is empty")

    with tempfile.TemporaryDirectory(prefix="sort_dynamic_graph_") as tmpdir:
        chunk_paths: list[str] = []

        with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
            fin.readline()
            events: list[list[str]] = []
            for line in fin:
                parts = line.split()
                if len(parts) != 4:
                    continue

                if parts[2] not in {"+1", "-1"}:
                    continue

                events.append(parts)
                if len(events) >= chunk_size:
                    chunk_path = os.path.join(tmpdir, f"chunk_{len(chunk_paths)}.txt")
                    write_sorted_chunk(events, chunk_path)
                    chunk_paths.append(chunk_path)
                    events = []

            if events:
                chunk_path = os.path.join(tmpdir, f"chunk_{len(chunk_paths)}.txt")
                write_sorted_chunk(events, chunk_path)
                chunk_paths.append(chunk_path)

        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write(header)
            if header and not header.endswith("\n"):
                fout.write("\n")

            iterators = [iter_chunk(path) for path in chunk_paths]
            for _, _, u, v, op, ts in heapq.merge(*iterators):
                fout.write(f"{u} {v} {op} {ts}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sort dynamic graph events by timestamp, with +1 before -1 for equal timestamps."
    )
    parser.add_argument("input", help="Input dynamic graph file")
    parser.add_argument("output", help="Output sorted dynamic graph file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of events sorted in memory per chunk",
    )
    args = parser.parse_args()
    sort_dynamic_graph(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
