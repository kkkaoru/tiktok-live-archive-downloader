#!/usr/bin/env bash
# Consume llvm-cov report from the package directory, including every executable object.
set -euo pipefail
awk '
/^Sources\// {
    files++
    invalid = (NF != 13)
    split("2 3 5 6 8 9 11 12", columns, " ")
    for (column in columns) {
        if ($(columns[column]) !~ /^[0-9]+$/) invalid = 1
    }
    if (invalid || $5 <= 0 || $8 <= 0 || $6 > $5 || $9 > $8 || $12 > $11) {
        print "FAIL invalid coverage row: " $0
        failed = 1
        next
    }
    functions = ($5 - $6) * 100 / $5
    lines = ($8 - $9) * 100 / $8
    printf "%s lines=%.2f%% functions=%.2f%%", $1, lines, functions
    if (($5 - $6) * 100 < $5 * 95 || ($8 - $9) * 100 < $8 * 95) failed = 1
    if ($11 > 0) {
        branches += $11
        printf " branches=%.2f%%", ($11 - $12) * 100 / $11
        if (($11 - $12) * 100 < $11 * 95) failed = 1
    }
    print ""
}
END {
    if (!files) { print "FAIL no instrumented production sources"; exit 1 }
    if (!branches) print "Branch counters not emitted; branch coverage is not measured."
    print failed ? "FAIL per-file coverage requires >=95%" : "PASS per-file coverage >=95%"
    exit failed
}' "${1:?Usage: check-swift-coverage.sh LLVM_REPORT}"
