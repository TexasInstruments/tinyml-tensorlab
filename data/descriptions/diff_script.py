#!/usr/bin/env python3
"""
Rigorous structural diff between OLD (2026.08.13 release, timeseries-only flat dict)
and NEW['timeseries'] (current descriptions.json) for the COMMON top-level keys.

stdlib only. Writes full structured result to diff_report.json.
"""
import json
import difflib
from collections import defaultdict

OLD_PATH = "/home/a0484689/PycharmProjects/tinyml_docker_images/2026.08.13/description_timeseries.json"
NEW_PATH = "/home/a0484689/PycharmProjects/py310_tinyml/tinyml-modelmaker/data/descriptions/descriptions.json"
OUT_PATH = "/home/a0484689/PycharmProjects/py310_tinyml/tinyml-modelmaker/data/descriptions/diff_report.json"

CLEANUP_KEYS = {"context_help", "help_url"}


def load():
    with open(OLD_PATH, "r") as f:
        old = json.load(f)
    with open(NEW_PATH, "r") as f:
        new_full = json.load(f)
    new = new_full["timeseries"]
    return old, new


def path_join(base, key):
    if base == "":
        return str(key)
    return f"{base}.{key}"


def is_cleanup_path(path):
    """True if the differing path is (or is nested under) context_help/help_url."""
    parts = path.replace("[]", "").split(".")
    return any(p in CLEANUP_KEYS for p in parts)


def recursive_diff(old_val, new_val, path, changes):
    """
    Populate `changes` (a list of dicts) with field-path-level diffs between
    old_val and new_val, rooted at `path`.
    Each change entry: {"path": str, "kind": "added"|"removed"|"changed",
                         "old": ..., "new": ...}
    """
    if isinstance(old_val, dict) and isinstance(new_val, dict):
        old_keys = set(old_val.keys())
        new_keys = set(new_val.keys())
        for k in sorted(old_keys - new_keys):
            changes.append({
                "path": path_join(path, k),
                "kind": "removed",
                "old": old_val[k],
                "new": None,
            })
        for k in sorted(new_keys - old_keys):
            changes.append({
                "path": path_join(path, k),
                "kind": "added",
                "old": None,
                "new": new_val[k],
            })
        for k in sorted(old_keys & new_keys):
            recursive_diff(old_val[k], new_val[k], path_join(path, k), changes)
    elif isinstance(old_val, list) and isinstance(new_val, list):
        if old_val == new_val:
            return
        # Try element-wise recursive diff if lists are dicts of same length;
        # otherwise, if lists of scalars, diff as a set; otherwise report whole-list change.
        if len(old_val) == len(new_val) and all(
            isinstance(a, dict) and isinstance(b, dict) for a, b in zip(old_val, new_val)
        ):
            for idx, (a, b) in enumerate(zip(old_val, new_val)):
                recursive_diff(a, b, path_join(path, f"[{idx}]"), changes)
        elif all(not isinstance(x, (dict, list)) for x in old_val + new_val):
            old_set = old_val
            new_set = new_val
            if old_set != new_set:
                changes.append({
                    "path": path_join(path, "[]"),
                    "kind": "changed",
                    "old": old_val,
                    "new": new_val,
                })
        else:
            changes.append({
                "path": path_join(path, "[]"),
                "kind": "changed",
                "old": old_val,
                "new": new_val,
            })
    else:
        if old_val != new_val:
            changes.append({
                "path": path,
                "kind": "changed",
                "old": old_val,
                "new": new_val,
            })


def diff_common_entries(old_section, new_section, common_keys):
    """
    For each common key (e.g. model name), recursively diff old_section[key] vs
    new_section[key]. Aggregate by field-path across all keys.

    Returns:
      per_entry: {entry_key: [change_dict, ...]}
      aggregated: {field_path: {"kind": set, "count": int, "entries": [entry_key,...],
                                 "tag": "expected (context_help/help_url cleanup)" | "unexpected - needs review",
                                 "examples": [ {entry, kind, old, new} up to 3 ]}}
    """
    per_entry = {}
    aggregated = defaultdict(lambda: {"kinds": set(), "count": 0, "entries": [], "examples": []})

    for key in sorted(common_keys):
        changes = []
        recursive_diff(old_section[key], new_section[key], "", changes)
        per_entry[key] = changes
        for ch in changes:
            agg = aggregated[ch["path"]]
            agg["kinds"].add(ch["kind"])
            agg["count"] += 1
            agg["entries"].append(key)
            if len(agg["examples"]) < 3:
                agg["examples"].append({
                    "entry": key,
                    "kind": ch["kind"],
                    "old": ch["old"],
                    "new": ch["new"],
                })

    aggregated_out = {}
    for field_path, agg in aggregated.items():
        tag = "expected (context_help/help_url cleanup)" if is_cleanup_path(field_path) else "unexpected - needs review"
        aggregated_out[field_path] = {
            "kinds": sorted(agg["kinds"]),
            "count": agg["count"],
            "affected_entries": sorted(agg["entries"]),
            "tag": tag,
            "examples": agg["examples"],
        }

    return per_entry, aggregated_out


def section_report(section_name, old, new):
    old_sec = old[section_name]
    new_sec = new[section_name]
    old_keys = set(old_sec.keys())
    new_keys = set(new_sec.keys())
    common = old_keys & new_keys
    per_entry, aggregated = diff_common_entries(old_sec, new_sec, common)

    n_identical = sum(1 for k in common if not per_entry[k])
    n_differing = len(common) - n_identical

    return {
        "old_key_count": len(old_keys),
        "new_key_count": len(new_keys),
        "common_key_count": len(common),
        "keys_only_in_old": sorted(old_keys - new_keys),
        "keys_only_in_new": sorted(new_keys - old_keys),
        "entries_identical_count": n_identical,
        "entries_differing_count": n_differing,
        "aggregated_field_path_diffs": aggregated,
        "per_entry_changes": per_entry,
    }


def target_device_report(old, new):
    old_sec = old["target_device_descriptions"]
    new_sec = new["target_device_descriptions"]

    assert isinstance(old_sec, dict), f"expected OLD target_device_descriptions to be dict, got {type(old_sec)}"
    assert isinstance(new_sec, list), f"expected NEW target_device_descriptions to be list, got {type(new_sec)}"

    old_names = set(old_sec.keys())
    new_names = set(new_sec)

    old_not_in_new = sorted(old_names - new_names)
    new_not_in_old = sorted(new_names - old_names)

    # Confirm metadata fields are entirely gone: new entries are bare strings (no dict/metadata).
    all_new_are_strings = all(isinstance(x, str) for x in new_sec)

    # Sample of what a metadata record used to look like (for documentation).
    sample_old_record = None
    if old_sec:
        sample_key = sorted(old_sec.keys())[0]
        sample_old_record = {sample_key: old_sec[sample_key]}

    return {
        "old_type": "dict",
        "new_type": "list",
        "old_device_count": len(old_sec),
        "new_device_count": len(new_sec),
        "old_device_names": sorted(old_names),
        "new_device_names": sorted(new_names),
        "old_names_not_in_new": old_not_in_new,
        "new_names_not_in_old": new_not_in_old,
        "all_old_names_present_in_new": len(old_not_in_new) == 0,
        "all_new_entries_are_bare_strings_no_metadata": all_new_are_strings,
        "sample_old_metadata_record": sample_old_record,
        "conclusion": (
            "CONFIRMED REGRESSION: NEW target_device_descriptions is a flat list of bare "
            "device-name strings; all per-device metadata (device_type, device_selection_factor, "
            "device_details, sdk_version, sdk_release) has been lost."
            if all_new_are_strings else
            "NEW target_device_descriptions contains non-string entries; regression not fully confirmed as described."
        ),
    }


def exact_equality_report(section_name, old, new):
    old_sec = old[section_name]
    new_sec = new[section_name]
    old_keys = set(old_sec.keys())
    new_keys = set(new_sec.keys())
    common = old_keys & new_keys

    differing = {}
    identical_count = 0
    for k in sorted(common):
        if old_sec[k] == new_sec[k]:
            identical_count += 1
        else:
            differing[k] = {"old": old_sec[k], "new": new_sec[k]}

    return {
        "old_key_count": len(old_keys),
        "new_key_count": len(new_keys),
        "keys_only_in_old": sorted(old_keys - new_keys),
        "keys_only_in_new": sorted(new_keys - old_keys),
        "identical_count": identical_count,
        "differing_count": len(differing),
        "differing_values": differing,
    }


def help_descriptions_report(old, new):
    old_str = old["help_descriptions"]
    new_str = new["help_descriptions"]
    assert isinstance(old_str, str) and isinstance(new_str, str)

    old_lines = old_str.splitlines(keepends=True)
    new_lines = new_str.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, fromfile="OLD/help_descriptions",
                                      tofile="NEW/help_descriptions", lineterm=""))

    return {
        "old_length_chars": len(old_str),
        "new_length_chars": len(new_str),
        "old_length_lines": len(old_lines),
        "new_length_lines": len(new_lines),
        "identical": old_str == new_str,
        "unified_diff": diff,
    }


def main():
    old, new = load()

    report = {}

    print("Diffing model_descriptions ...")
    report["model_descriptions"] = section_report("model_descriptions", old, new)

    print("Diffing preset_descriptions ...")
    report["preset_descriptions"] = section_report("preset_descriptions", old, new)

    print("Diffing sample_dataset_descriptions ...")
    report["sample_dataset_descriptions"] = section_report("sample_dataset_descriptions", old, new)

    print("Diffing target_device_descriptions ...")
    report["target_device_descriptions"] = target_device_report(old, new)

    print("Diffing task_descriptions / tooltip_descriptions / version_descriptions ...")
    report["task_descriptions"] = exact_equality_report("task_descriptions", old, new)
    report["tooltip_descriptions"] = exact_equality_report("tooltip_descriptions", old, new)
    report["version_descriptions"] = exact_equality_report("version_descriptions", old, new)

    print("Diffing help_descriptions ...")
    report["help_descriptions"] = help_descriptions_report(old, new)

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)

    # ---- Print a verification summary to stdout ----
    print("\n=== VERIFICATION SUMMARY ===")

    for sec in ["model_descriptions", "preset_descriptions", "sample_dataset_descriptions"]:
        r = report[sec]
        n_unexpected = sum(1 for v in r["aggregated_field_path_diffs"].values()
                            if v["tag"] == "unexpected - needs review")
        n_expected = sum(1 for v in r["aggregated_field_path_diffs"].values()
                          if v["tag"] == "expected (context_help/help_url cleanup)")
        print(f"{sec}: common_keys={r['common_key_count']} identical_entries={r['entries_identical_count']} "
              f"differing_entries={r['entries_differing_count']} "
              f"distinct_field_paths={len(r['aggregated_field_path_diffs'])} "
              f"(expected_cleanup={n_expected}, unexpected={n_unexpected})")

    td = report["target_device_descriptions"]
    print(f"target_device_descriptions: old_count={td['old_device_count']} new_count={td['new_device_count']} "
          f"old_names_not_in_new={td['old_names_not_in_new']} new_names_not_in_old={td['new_names_not_in_old']} "
          f"all_metadata_gone={td['all_new_entries_are_bare_strings_no_metadata']}")

    for sec in ["task_descriptions", "tooltip_descriptions", "version_descriptions"]:
        r = report[sec]
        print(f"{sec}: identical={r['identical_count']} differing={r['differing_count']} "
              f"keys={r['differing_values'] and list(r['differing_values'].keys())}")

    hd = report["help_descriptions"]
    print(f"help_descriptions: old_len={hd['old_length_chars']} new_len={hd['new_length_chars']} "
          f"identical={hd['identical']} diff_lines={len(hd['unified_diff'])}")

    print(f"\nFull structured report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
