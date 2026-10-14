#!/usr/bin/env bash
# Benchmarks a git ref, then the working tree, and reports what changed between them.
#
#   scripts/bench-compare.sh                                  # main against the working tree
#   scripts/bench-compare.sh v0.5.0                           # any branch, tag or commit
#   scripts/bench-compare.sh main --bench inference           # cargo options after the ref
#   scripts/bench-compare.sh main -- feed_forward --quick     # Criterion options after --
#
# The ref is checked out into a temporary git worktree and benchmarked there, so the
# working tree, uncommitted changes included, is never touched. Both runs share the
# target directory: dependencies build once, and both results land in
# target/criterion. The ref's results are saved as a Criterion baseline named after
# it, so `cargo bench -- --baseline-lenient <name>` compares against it again later
# without re-running the ref.
#
# Benchmarks that only exist on one side are listed as not compared, and `--bench`
# targets the ref does not have are skipped there. Run on AC power with other
# programs closed; the two runs are sequential, so anything that slows the machine
# down during one of them shows up as a change. `scripts/bench-compare.sh HEAD`
# compares the code with itself, which shows how noisy the machine is.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
ref=main
if [[ $# -gt 0 && $1 != -* ]]; then
    ref=$1
    shift
fi
commit=$(git -C "$repo" rev-parse --verify --quiet "$ref^{commit}") || {
    echo "error: $ref is not a commit" >&2
    exit 1
}
baseline=${ref//[^A-Za-z0-9._-]/_}
# Criterion keeps every run in `new` and plain `cargo bench` runs in `base`.
if [[ $baseline == base || $baseline == new ]]; then
    echo "error: Criterion uses the baseline name '$baseline' itself; pass the commit instead" >&2
    exit 1
fi

# Options before `--` go to cargo, the rest to Criterion.
cargo_args=()
criterion_args=()
while [[ $# -gt 0 && $1 != -- ]]; do
    cargo_args+=("$1")
    shift
done
if [[ $# -gt 0 ]]; then
    shift
    criterion_args=("$@")
fi

# Without a target selection, only the Criterion targets: older refs still benchmark
# the library with libtest, which rejects Criterion's options.
selects_targets=false
for arg in ${cargo_args[@]+"${cargo_args[@]}"}; do
    case $arg in
        --bench | --bench=* | --benches | --lib | --all-targets) selects_targets=true ;;
    esac
done
$selects_targets || cargo_args+=(--bench '*')

# The ref gets the same options, minus the `--bench <name>` targets it does not have
# yet. If that leaves it no target at all, it runs every Criterion target it has, and
# the Criterion filter decides what runs.
has_target() {
    git -C "$repo" cat-file -e "$commit:benches/$1.rs" 2>/dev/null ||
        git -C "$repo" cat-file -e "$commit:benches/$1/main.rs" 2>/dev/null
}
ref_args=()
ref_selected=false
set -- ${cargo_args[@]+"${cargo_args[@]}"}
while [[ $# -gt 0 ]]; do
    case $1 in
        --bench | --bench=*)
            if [[ $1 == --bench ]]; then
                target=${2-}
                shift
            else
                target=${1#--bench=}
            fi
            shift
            if [[ $target == *[*?[]* ]] || has_target "$target"; then
                ref_args+=(--bench "$target")
                ref_selected=true
            else
                echo "note: $ref has no bench target '$target', skipping it there" >&2
            fi
            ;;
        --benches | --lib | --all-targets)
            ref_args+=("$1")
            ref_selected=true
            shift
            ;;
        *)
            ref_args+=("$1")
            shift
            ;;
    esac
done
$ref_selected || ref_args+=(--bench '*')

# Absolute, since the two runs start in different directories.
target_dir=${CARGO_TARGET_DIR:-$repo/target}
[[ $target_dir == /* ]] || target_dir=$PWD/$target_dir
export CARGO_TARGET_DIR=$target_dir
log_dir=$(mktemp -d)
worktree="$log_dir/worktree"
trap 'git -C "$repo" worktree remove --force "$worktree" 2>/dev/null; rm -rf "$log_dir"' EXIT

echo "==> Benchmarking $ref ($(git -C "$repo" rev-parse --short "$commit")) as baseline '$baseline'"
git -C "$repo" worktree add --quiet --detach "$worktree" "$commit"
(cd "$worktree" && cargo bench "${ref_args[@]}" -- \
    --save-baseline "$baseline" ${criterion_args[@]+"${criterion_args[@]}"})

echo
echo "==> Benchmarking the working tree against '$baseline'"
(cd "$repo" && cargo bench "${cargo_args[@]}" -- \
    --baseline-lenient "$baseline" ${criterion_args[@]+"${criterion_args[@]}"}) \
    2>&1 | tee "$log_dir/candidate.log"

# Criterion prints each benchmark's name, then, if it has a baseline, its change in
# time (the middle figure is the estimate; benchmarks with a throughput print it on
# the line after `change:`), then a verdict when the change is significant and
# beyond the noise threshold. This relies on Criterion's default output: `--quiet`
# or another `--format` breaks it.
echo
echo "==> Changes against $ref"
awk '
    /^Benchmarking [^:]+$/ { name = $2; names[++n] = name }
    /change:/ { in_change = 1; compared[name] = 1; c++ }
    in_change && /\[/ {
        in_change = 0
        match($0, /\[[^]]*\]/)
        split(substr($0, RSTART + 1, RLENGTH - 2), figures, " ")
        change = figures[2]
    }
    /Performance has regressed/ { regressed[++r] = sprintf("  %-60s %s", name, change) }
    /Performance has improved/  { improved[++i] = sprintf("  %-60s %s", name, change) }
    END {
        print "Regressed:"; if (!r) print "  none"; for (k = 1; k <= r; k++) print regressed[k]
        print "Improved:";  if (!i) print "  none"; for (k = 1; k <= i; k++) print improved[k]
        print "Not compared, no baseline for them:"
        for (k = 1; k <= n; k++) if (!(names[k] in compared)) { print "  " names[k]; u++ }
        if (!u) print "  none"
        printf "Within noise: the other %d of the %d compared.\n", c - r - i, c
    }
' "$log_dir/candidate.log"
echo "The reports are in $CARGO_TARGET_DIR/criterion/report/index.html."
