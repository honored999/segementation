"""Full-case report table. No model/checkpoint imports or metric recomputation."""
from __future__ import annotations

import csv
import math
import re
import textwrap
from pathlib import Path

METRICS = ("dice", "iou", "f2", "avd_percent", "lcd", "recall", "hd95_mm")
COUNTS = ("tp", "fp", "fn")
TITLES = dict(zip(METRICS, ("Dice (0..1)", "IoU (0..1)", "F2 (0..1)", "AVD (%)", "LCD", "Recall (0..1)", "HD95 (mm)")))
MISSING = ("", "none", "null", "n/a", "na")


def metric_value(value):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid numeric metric: {value!r}")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid numeric metric: {value!r}") from error
    return number if math.isfinite(number) else None


def integer_value(value, cid, key):
    """Never pass counts through float, including integers above 2**53."""
    if value is None or isinstance(value, str) and value.strip().lower() in MISSING:
        return None
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{cid}: {key} requires an exact nonnegative integer, got {value!r}")


def source_counts(row, cid):
    result = {}
    for key in COUNTS:
        values = [integer_value(row[k], cid, k) for k in (key, key.upper()) if k in row]
        # Both aliases present must agree, including missing vs nonmissing.
        if len(values) == 2 and values[0] != values[1]:
            raise ValueError(f"{cid}: conflicting {key}/{key.upper()} columns")
        result[key] = values[0] if values else None
    return result


def count_coverage(rows):
    pending = []
    for cid, row in rows.items():
        if any(value is None for value in source_counts(row, cid).values()):
            pending.append(cid)
    return {"source_complete_cases": len(rows)-len(pending), "pending_count_cases": sorted(pending),
            "verification": "metadata only; voxel counts/geometry not verified"}


def binary_labels(*documents, standalone=False):
    """Only explicit binary class-index contracts are supported for fallback."""
    found = []
    def visit(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if key in ("ignore_label", "ignore_index") and value is not None:
                raise ValueError("ignore label contracts are unsupported for counting")
            if key in ("regions_class_order", "regions", "region_based") and value not in (None, False, [], {}):
                raise ValueError("region contracts are unsupported for counting")
            if key == "labels":
                if not isinstance(value, dict) or any(type(v) is not int for v in value.values()):
                    raise ValueError("unknown/region label contract")
                if "ignore" in value or value.get("background") != 0 or len(value) != 2:
                    raise ValueError("only background=0 and one foreground label supported")
                positive = [v for k,v in value.items() if k != "background"]
                if positive[0] <= 0:
                    raise ValueError("foreground label must be positive")
                found.append(positive[0])
            if key == "num_classes" and (type(value) is not int or value != 2):
                raise ValueError("only binary num_classes=2 supported")
            if isinstance(value, dict):
                visit(value)
    for doc in documents:
        visit(doc)
    if standalone:
        # Existing standalone class-index training/inference contract: 0/1.
        config = documents[0] if documents else {}
        if not isinstance(config.get("model"), dict) or config["model"].get("num_classes") != 2:
            raise ValueError("unknown standalone binary label contract")
        found.append(1)
    if not found or len(set(found)) != 1:
        raise ValueError("unknown or conflicting binary label contract")
    return found[0]


def voxel_counts(pred, gt, foreground=1):
    import numpy as np
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError("mask shape mismatch")
    for kind, array in (("prediction", pred), ("GT", gt)):
        if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == foreground)):
            raise ValueError(f"{kind}: invalid binary mask labels (expected 0/{foreground})")
    p, g = pred == foreground, gt == foreground
    return dict(tp=int(np.count_nonzero(p & g)), fp=int(np.count_nonzero(p & ~g)),
                fn=int(np.count_nonzero(~p & g)))


def saved_mask_counts(prediction, label, cid, foreground, check_geometry):
    import SimpleITK as sitk
    pred = sitk.ReadImage(str(prediction))
    gt = sitk.ReadImage(str(label))
    check_geometry(pred, gt, cid)
    return voxel_counts(sitk.GetArrayFromImage(pred), sitk.GetArrayFromImage(gt), foreground)


def _validate_counts(row, counts, cid, summary):
    tp, fp, fn = (counts[k] for k in COUNTS)
    for key, expected in (("pred_voxels", tp+fp), ("gt_voxels", tp+fn)):
        if key in row:
            observed = integer_value(row[key], cid, key)
            if observed is not None and observed != expected:
                raise ValueError(f"{cid}: {key} conflicts with TP/FP/FN")
    # Aggregation and f2_mode alone are not case-definition evidence.
    definitions = summary.get("metric_definitions", {})
    if not isinstance(definitions, dict) or set(definitions) - {"dice", "iou", "recall", "f2"}:
        raise ValueError("unsupported metric_definitions")
    formulas = {
        "dice": {"2tp/(2tp+fp+fn)": (2*tp, 2*tp+fp+fn)},
        "iou": {"tp/(tp+fp+fn)": (tp, tp+fp+fn)},
        "recall": {"tp/(tp+fn)": (tp, tp+fn)},
        "f2": {"5tp/(5tp+4fp+fn)": (5*tp, 5*tp+4*fp+fn),
               "5tp/(5tp+fp+4fn)": (5*tp, 5*tp+fp+4*fn)},
    }
    for key, definition in definitions.items():
        if not isinstance(definition, dict) or set(definition) - {"formula", "zero_denominator"}:
            raise ValueError(f"unsupported metric definition: {key}")
        formula = definition.get("formula", "unknown")
        empty = definition.get("zero_denominator", "unknown")
        if not isinstance(formula, str) or formula not in (*formulas[key], "unknown") or empty not in ("unknown", "missing"):
            raise ValueError(f"unsupported metric definition: {key}")
        if formula == "unknown":
            if empty != "unknown":
                raise ValueError(f"{key}: zero denominator requires a known formula")
            continue
        if key == "f2":
            mode = "paper" if formula == "5tp/(5tp+4fp+fn)" else "standard"
            if summary.get("f2_mode") in ("paper", "standard") and summary["f2_mode"] != mode:
                raise ValueError(f"{cid}: f2 conflicts with declared formula/f2_mode")
        num, den = formulas[key][formula]
        observed = metric_value(row[key])
        if den == 0:
            conflict = empty == "missing" and observed is not None
        else:
            conflict = observed is not None and not math.isclose(observed, num/den, rel_tol=1e-7, abs_tol=1e-10)
        if conflict:
            raise ValueError(f"{cid}: {key} conflicts with TP/FP/FN (caller-declared formula={formula}, zero_denominator={empty})")


def macro_summary(rows, summary):
    if type(summary.get("n_cases")) is not int or summary["n_cases"] != len(rows):
        raise ValueError("summary n_cases mismatch")
    if "case_ids" in summary:
        ids = summary["case_ids"]
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(rows):
            raise ValueError("summary case_ids mismatch")
    if summary.get("aggregation") != "macro average over cases":
        raise ValueError("unknown summary aggregation; cannot label means as case macro averages")
    means, evidence = {}, {}
    for key in METRICS:
        values = [metric_value(row[key]) for row in rows.values()]
        valid = [v for v in values if v is not None]
        item = summary.get("metrics", {}).get(key)
        if not isinstance(item, dict) or type(item.get("valid_cases")) is not int or item["valid_cases"] != len(valid):
            raise ValueError(f"summary valid_cases mismatch: {key} (finite-value policy)")
        calculated = math.fsum(valid)/len(valid) if valid else None
        if "mean" in item:
            source = metric_value(item["mean"])
            if (source is None) != (calculated is None) or source is not None and not math.isclose(source, calculated, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"summary mean mismatch: {key}")
            means[key] = item["mean"] if source is not None else "NA"
            origin = "source_summary_verified_against_full_rows"
        else:
            means[key] = calculated if calculated is not None else "NA"
            origin = "fallback_macro_from_full_rows"
        evidence[key] = {"valid_cases": len(valid), "source": origin}
    return means, evidence


def build_table(rows, summary, predictions, labels, *, label_contract=None, check_geometry=None,
                loader=None, geometry_verified=False):
    """Resolve one missing case at a time; complete CSV path never reads masks."""
    if not rows or set(rows) != set(predictions) or not set(rows) <= set(labels):
        raise ValueError("table case coverage mismatch")
    means, evidence = macro_summary(rows, summary)
    foreground_definition = summary.get("count_foreground", "unknown")
    if foreground_definition not in ("unknown", "positive"):
        raise ValueError("unsupported count_foreground")
    records = []
    extras = []
    reserved = {"row_type", "count_label_definition", "count_geometry", *(k+"_source" for k in COUNTS)}
    for cid in sorted(rows):
        row = rows[cid]
        if row.get("case_id", cid) != cid:
            raise ValueError(f"{cid}: case_id mismatch")
        if reserved.intersection(row):
            raise ValueError(f"{cid}: source columns collide with table metadata")
        counts = source_counts(row, cid)
        origins = {key: "source_csv" for key in COUNTS}
        needs = any(v is None for v in counts.values())
        if needs:
            if label_contract is None or check_geometry is None:
                raise ValueError(f"{cid}: missing counts require verified binary labels and geometry")
            foreground = label_contract()  # lazy: no added contract requirement for source reuse
            calculated = (loader or saved_mask_counts)(predictions[cid], labels[cid], cid, foreground, check_geometry)
            for key in COUNTS:
                value = integer_value(calculated.get(key), cid, key)
                if value is None:
                    raise ValueError(f"{cid}: recomputation missing {key}")
                if counts[key] is not None and counts[key] != value:
                    raise ValueError(f"{cid}: source {key} conflicts with saved masks")
                if counts[key] is None:
                    counts[key] = value
                    origins[key] = "recomputed_from_saved_masks"
        _validate_counts(row, counts, cid, summary)
        record = dict(row, row_type="case", case_id=cid, **counts)
        # Canonical count aliases retain exact integer formatting.
        for key in COUNTS:
            if key.upper() in record:
                record[key.upper()] = counts[key]
            record[key+"_source"] = origins[key]
        record["count_label_definition"] = (f"saved-mask binary background=0 foreground={foreground}; source foreground {foreground_definition}" if needs
                                            else f"source foreground {foreground_definition}" +
                                            (" (caller-declared >0)" if foreground_definition == "positive" else "") +
                                            "; masks not inspected by count service")
        record["count_geometry"] = "verified" if needs or geometry_verified else "not_verified_by_count_service"
        records.append(record)
        for key in row:
            if key not in ("case_id", *METRICS, *COUNTS) and key not in extras:
                extras.append(key)
    columns = ["row_type", "case_id", *METRICS, *COUNTS, *extras, *(k+"_source" for k in COUNTS),
               "count_label_definition", "count_geometry"]
    mean = dict(row_type="case_macro_mean", case_id="病例平均指标", **means)
    total = dict(row_type="all_case_voxel_total", case_id="全病例 TP/FP/FN 总计",
                 **{k: sum(r[k] for r in records) for k in COUNTS})
    notes = ["Dice/IoU/F2/Recall: 0..1; AVD already percent (no x100); HD95 mm; LCD lesion-count difference.",
             "TP/FP/FN: original full-volume voxel counts; no spacing scaling, lesion-count or slice aggregation.",
             "Metrics retained from source; nonfinite/missing: CSV NA, display N/A. Unknown extra columns are not averaged.",
             "Count provenance per case and per item: source_csv or recomputed_from_saved_masks; no model/HD95 evaluation for counts.",
             "Geometry 'verified': existing full-report header checks or fallback saved-mask checks; source counts are not voxel-validated.",
             "Case metric definitions (caller-declared source summary; not independently verified): " +
             "; ".join(f"{k}: formula={summary.get('metric_definitions', {}).get(k, {}).get('formula', 'unknown')}, "
                       f"zero_denominator={summary.get('metric_definitions', {}).get(k, {}).get('zero_denominator', 'unknown')}"
                       for k in ("dice", "iou", "recall", "f2")),
             "Source count foreground: " + foreground_definition + "; independent of saved-mask label contract.",
             "F2 mode: " + str(summary.get("f2_mode", "unknown (not assumed)")),
             "Aggregation: macro average over cases; source means verified against full rows (case IDs checked when supplied).",
             "Mean evidence: " + "; ".join(f"{k}: valid N={v['valid_cases']}, {v['source']}" for k,v in evidence.items())]
    return {"columns": columns, "records": [*records, mean, total], "notes": notes}


def _cell(record, key, display=False):
    value = record.get(key, "")
    if key in METRICS and record.get("row_type") in ("case", "case_macro_mean"):
        number = metric_value(value)
        if number is None:
            return "N/A" if display else "NA"
        return f"{number:.5g}" if display else str(value)
    # Additional source columns are preserved, with explicit nonfinite display
    # tokens; no aggregation is inferred for those columns.
    if isinstance(value, float) and not math.isfinite(value) or isinstance(value, str) and value.strip().lower() in ("nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
        return "N/A" if display else "NA"
    return str(value)


def export_table(table, output):
    """CSV, TXT and PNG all consume exactly the same ordered records."""
    output = Path(output)
    cols, records = table["columns"], table["records"]
    with (output/"metrics_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols)
        writer.writerows([_cell(r,k) for k in cols] for r in records)
    headers = [TITLES.get(k,k.upper() if k in COUNTS else k) for k in cols]
    cells = [[_cell(r,k,True) for k in cols] for r in records]
    txt = "\nFull evaluation metrics table\n" + "\t".join(headers)+"\n"
    txt += "\n".join("\t".join(row) for row in cells)+"\n"+"\n".join(table["notes"])+"\n"
    render_table(headers, cells, table["notes"], output/"metrics_table.png")
    return txt


def render_table(headers, cells, notes, output):
    """Keep full ordered data; render all rows and columns without truncation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report_visuals import BODY, TITLE, setup_font, wrap, save_figure
    setup_font(); output=Path(output)
    def draw(indices, row_indices, path):
        limits=[28 if headers[i]=='case_id' else 18 for i in indices]
        wrapped=[[wrap(str(v),limits[j]) for j,v in enumerate(row)] for row in
            [[headers[i] for i in indices],*[[cells[r][i] for i in indices] for r in row_indices]]]
        heights=[.27*max(len(v.split('\n')) for v in row)+.18 for row in wrapped]
        size=(max(18,2.7*len(indices)),sum(heights)+2.2)
        fig=plt.figure(figsize=size)
        top=.84; bottom=.12
        ax=fig.add_axes((.025,bottom,.95,top-bottom)); ax.axis('off')
        # Case IDs get more room, without suppressing source/provenance columns.
        widths=[1.8 if headers[i]=='case_id' else 1.1 for i in indices]; total=sum(widths)
        tab=ax.table(cellText=wrapped[1:],colLabels=wrapped[0],colWidths=[w/total for w in widths],cellLoc='left',bbox=(0,0,1,1))
        tab.auto_set_font_size(False); tab.set_fontsize(BODY)
        for (r,c),cell in tab.get_celld().items():
            cell.set_height(heights[r]/sum(heights)); cell.PAD=.045; cell.set_edgecolor('#c8d2dc')
            row_type=cells[row_indices[r-1]][headers.index('row_type')] if r and 'row_type' in headers else ''
            cell.set_facecolor('#dae7f2' if r==0 else '#e6f0e5' if row_type in ('case_macro_mean','voxel_total') else '#f4f7fa' if r%2 else 'white')
        fig.suptitle('Full evaluation set | metrics and voxel counts',fontsize=TITLE,y=.975)
        fig.text(.025,.885,'All source columns and cases; ordered macro and voxel summaries retained',fontsize=BODY)
        fig.text(.025,.04,'Missing: N/A; blank: not applicable. Source definitions, units and full provenance: report.txt / CSV',fontsize=BODY)
        try: save_figure(fig,path,family='metrics_table',coverage={'column_indices':indices,'columns':[headers[i] for i in indices],
            'row_indices':row_indices,'case_ids':[cells[r][headers.index('case_id')] for r in row_indices]})
        finally: plt.close(fig)
    draw(list(range(len(headers))),list(range(len(cells))),output)
