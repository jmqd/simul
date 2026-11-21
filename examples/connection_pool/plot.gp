# Publication-quality gnuplot 6.0 script for examples/connection_pool/
#
# Usage:
#   gnuplot -e "DATA_DIR='/path/to/tsvs'; OUT_DIR='examples/connection_pool/assets'" \
#       examples/connection_pool/plot.gp

if (!exists("DATA_DIR")) DATA_DIR = "data"
if (!exists("OUT_DIR"))  OUT_DIR  = "examples/connection_pool/assets"

# Colorblind-safe high-contrast palette (Okabe-Ito / Tol inspired)
NAVY    = "#1f4e79"
BLUE    = "#2b7bba"
ORANGE  = "#d95f02"
TEAL    = "#1b9e77"
CRIMSON = "#c0392b"
PURPLE  = "#6a3d9a"
AMBER   = "#d4a017"
SLATE   = "#475569"
MUTED   = "#94a3b8"
GRIDC   = "#e2e8f0"

set encoding utf8
set datafile separator "\t"
set border 3 lw 1.2 lc rgb "#334155"
set tics nomirror out scale 0.6 tc rgb "#334155" font "Helvetica,10"
set grid back lc rgb GRIDC lt 0 lw 1

# ==============================================================================
# CHART 1: HikariCP / USL Pool-Size Knee & U-Shaped Latency Breakdown
# ==============================================================================
set terminal pngcairo enhanced truecolor font "Helvetica,11" size 1360,560 rounded
set output sprintf("%s/pool_size_knee.png", OUT_DIR)

set multiplot layout 1,2 title "{/:Bold Database Connection Pool Sizing: HikariCP / USL Knee \\& Latency Crossover}" \
    font "Helvetica,14" tc rgb "#0f172a" margins 0.06, 0.98, 0.13, 0.87 spacing 0.08, 0.0

# --- Panel 1A: Effective Goodput vs Pool Size ---
set title "A. Effective Goodput vs. Pool Size (8 DB Cores)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set xlabel "Connection Pool Size (P, log_2 scale)" font "Helvetica,11" tc rgb "#1e293b"
set ylabel "Effective Goodput (req/s)" font "Helvetica,11" tc rgb "#1e293b"
set logscale x 2
set xrange [1.8:110]
set xtics (2, 4, 8, 12, 16, 24, 32, 48, 64, 96)
set yrange [0:575]
set ytics 0,100,500
set key bottom right box lc rgb "#cbd5e1" opaque width 1.0 spacing 1.25 font "Helvetica,10"

set object 1 rect from 13, 0 to 21, 575 fc rgb TEAL fs transparent solid 0.12 noborder behind
set object 2 rect from 44, 0 to 110, 575 fc rgb CRIMSON fs transparent solid 0.09 noborder behind
set arrow 1 from 13.9, 0 to 13.9, 535 nohead dt 2 lw 1.3 lc rgb TEAL front
set arrow 2 from 1.8, 504.5 to 110, 504.5 nohead dt 3 lw 1.1 lc rgb SLATE back

set label 1 "Goldilocks Knee\n(n* = 13.9)" at 17.5, 255 center font "Helvetica,9.5" tc rgb "#115e59" front
set label 2 "USL Thrashing\n(-30% at P=96)" at 69, 235 center font "Helvetica,9.5" tc rgb "#991b1b" front
set label 3 "Offered load (504.5 req/s)" at 2.1, 524 left font "Helvetica,9" tc rgb SLATE
set label 4 "RPC-hold penalty\n(-94 req/s at P=12)" at 2.15, 422 left font "Helvetica,9" tc rgb ORANGE
set arrow 3 from 5.6, 416 to 11.2, 405 lw 1.15 lc rgb ORANGE head filled size screen 0.008,20

plot sprintf("%s/sweep.tsv", DATA_DIR) skip 1 using 1:2:3 with yerrorbars lc rgb NAVY lw 1.3 pt 7 ps 0.9 notitle, \
     "" skip 1 using 1:2 with linespoints lc rgb NAVY lw 2.4 pt 7 ps 1.0 title "Transaction pooling (release before RPC)", \
     "" skip 1 using 1:12 with linespoints lc rgb ORANGE lw 2.1 dt 2 pt 4 ps 1.0 title "Session pooling (hold across RPC)"

unset object 1
unset object 2
unset arrow 1
unset arrow 2
unset arrow 3
unset label 1
unset label 2
unset label 3
unset label 4

# --- Panel 1B: Latency Breakdown (Log-Y U-Curve) ---
set title "B. Latency Crossover: Pool Wait vs. USL DB Execution (Txn Mode)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set xlabel "Connection Pool Size (P, log_2 scale)" font "Helvetica,11" tc rgb "#1e293b"
set ylabel "Latency (ms, log_{10} scale)" font "Helvetica,11" tc rgb "#1e293b"
set logscale y 10
set yrange [12:5800]
set ytics (20, 50, 100, 200, 500, 1000, 2000, 4000)
set key top left box lc rgb "#cbd5e1" opaque width 0.5 spacing 1.18 font "Helvetica,9.5"

set object 3 rect from 13, 12 to 21, 5800 fc rgb TEAL fs transparent solid 0.12 noborder behind
set object 4 rect from 44, 12 to 110, 5800 fc rgb CRIMSON fs transparent solid 0.09 noborder behind

set label 5 "Pool-wait\ndominated" at 3.8, 36 center font "Helvetica,9.5" tc rgb PURPLE
set label 6 "USL DB-exec\ndominated" at 70, 92 center font "Helvetica,9.5" tc rgb "#9a3412"
set label 7 "Min P99: 475 ms\nat P = 16" at 16, 2650 center font "Helvetica,9.5" tc rgb "#115e59"
set arrow 4 from 16, 1750 to 16, 560 lw 1.2 lc rgb TEAL head filled size screen 0.008,20

plot sprintf("%s/sweep.tsv", DATA_DIR) skip 1 using 1:6 with linespoints lc rgb CRIMSON lw 2.4 pt 5 ps 0.9 title "End-to-end P99", \
     "" skip 1 using 1:7 with lines lc rgb ORANGE lw 1.8 dt 2 title "End-to-end CVaR95", \
     "" skip 1 using 1:5 with linespoints lc rgb BLUE lw 2.0 pt 7 ps 0.85 title "End-to-end P50", \
     "" skip 1 using 1:9 with linespoints lc rgb PURPLE lw 1.8 dt 4 pt 9 ps 0.9 title "Pool Wait P95", \
     "" skip 1 using 1:10 with linespoints lc rgb AMBER lw 2.1 pt 13 ps 0.95 title "DB Exec P95 (USL)"

unset multiplot
reset

# ==============================================================================
# CHART 2: Paired CRN Comparison of 5 Production Pooling Architectures
# ==============================================================================
set terminal pngcairo enhanced truecolor font "Helvetica,11" size 1360,550 rounded
set output sprintf("%s/patterns_comparison.png", OUT_DIR)

set encoding utf8
set datafile separator "\t"
set border 3 lw 1.2 lc rgb "#334155"
set tics nomirror out scale 0.6 tc rgb "#334155" font "Helvetica,9.5"
set grid y back lc rgb GRIDC lt 0 lw 1

set multiplot layout 1,2 title "{/:Bold Incremental Architectural Fixes Under Paired Common Random Numbers (32 Replications)}" \
    font "Helvetica,14" tc rgb "#0f172a" margins 0.06, 0.98, 0.17, 0.87 spacing 0.08, 0.0

# --- Panel 2A: Fast-OLTP P99 vs Overall CVaR95 (Log-Y Grouped Bars) ---
set title "A. Tail Latency Collapse (Log Scale, 95% Student-t CIs)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set ylabel "Latency (ms, log_{10} scale)" font "Helvetica,11" tc rgb "#1e293b"
set logscale y 10
set xrange [-0.6:4.6]
set yrange [10:6500]
set ytics (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
set boxwidth 0.34
set style fill solid 0.88 border lc rgb "#1e293b"
set key top right box lc rgb "#cbd5e1" opaque spacing 1.2 font "Helvetica,10"

set label 1 "97× Fast-P99\nreduction!" at 2.82, 760 center font "Helvetica,10" tc rgb "#115e59" front
set arrow 1 from 2.82, 460 to 2.82, 36 lw 1.4 lc rgb TEAL head filled size screen 0.009,20 front

plot sprintf("%s/patterns.tsv", DATA_DIR) skip 1 using ($1-0.18):7:xtic(2) with boxes fc rgb NAVY title "Fast-OLTP P99 (ms)", \
     "" skip 1 using ($1+0.18):10 with boxes fc rgb ORANGE title "All-Request CVaR95 (ms)", \
     "" skip 1 using ($1-0.18):7:8 with yerrorbars lc rgb "#0f172a" lw 1.2 pt 0 notitle, \
     "" skip 1 using ($1-0.18):($7*1.32):(sprintf("%.0f", $7)) with labels font "Helvetica,9" tc rgb NAVY notitle, \
     "" skip 1 using ($1+0.18):($10*1.32):(sprintf("%.0f", $10)) with labels font "Helvetica,9" tc rgb "#9a3412" notitle

unset label 1
unset arrow 1
unset logscale y

# --- Panel 2B: Request Success Rate (%) & Effective Goodput ---
set title "B. Request Success Rate (%) \\& Effective Goodput (req/s)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set ylabel "Success Rate (%)" font "Helvetica,11" tc rgb "#1e293b"
set xrange [-0.6:4.6]
set yrange [72:107.5]
set ytics 75,5,100
set boxwidth 0.52
set style fill solid 0.88 border lc rgb "#1e293b"
set key off

set arrow 2 from -0.6, 99.0 to 4.6, 99.0 nohead dt 2 lw 1.3 lc rgb CRIMSON front
set label 2 "99.0% Success SLO" at -0.52, 100.3 left font "Helvetica,9.5" tc rgb CRIMSON

plot sprintf("%s/patterns.tsv", DATA_DIR) skip 1 using 1:5:xtic(2) with boxes fc rgb TEAL, \
     "" skip 1 using 1:5:6 with yerrorbars lc rgb "#0f172a" lw 1.3 pt 0, \
     "" skip 1 using 1:($5+2.65):(sprintf("%.1f%%\n(%.0f/s)", $5, $3)) with labels font "Helvetica,9" tc rgb "#0f172a"

unset multiplot
reset

# ==============================================================================
# CHART 3: Nested-Pool Deadlock Phase Cliff & Time-to-Deadlock
# ==============================================================================
set terminal pngcairo enhanced truecolor font "Helvetica,11" size 1360,550 rounded
set output sprintf("%s/deadlock_phase_cliff.png", OUT_DIR)

set encoding utf8
set datafile separator "\t"
set border 3 lw 1.2 lc rgb "#334155"
set tics nomirror out scale 0.6 tc rgb "#334155" font "Helvetica,10"
set grid back lc rgb GRIDC lt 0 lw 1

set multiplot layout 1,2 title "{/:Bold Nested-Pool Deadlock Phase Transition \\& Burst-Triggered Failure Horizon}" \
    font "Helvetica,14" tc rgb "#0f172a" margins 0.06, 0.98, 0.13, 0.87 spacing 0.08, 0.0

# --- Panel 3A: Deadlock Probability Phase Boundary ---
set title "A. Exact Deadlock Probability vs. Pool Size (95% Wilson CIs)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set xlabel "Connection Pool Size (P)" font "Helvetica,11" tc rgb "#1e293b"
set ylabel "Deadlock Probability (%)" font "Helvetica,11" tc rgb "#1e293b"
set xrange [3:33.5]
set xtics (4, 6, 8, 10, 12, 16, 20, 24, 32)
set yrange [-4:117]
set ytics 0,20,100
set key bottom left box lc rgb "#cbd5e1" opaque spacing 1.25 font "Helvetica,10"

set object 1 rect from 14, -4 to 20, 117 fc rgb TEAL fs transparent solid 0.14 noborder behind
set label 1 "USL Sweet Spot\n(P = 14–20)" at 17, 109 center font "Helvetica,9" tc rgb "#115e59" front
set label 2 "Catch-22: 15–30% nested\ncalls deadlock inside\nthe USL sweet spot!" at 27.2, 92 center font "Helvetica,9" tc rgb CRIMSON front

plot sprintf("%s/deadlock.tsv", DATA_DIR) skip 1 using 1:2:3:4 with yerrorbars lc rgb AMBER lw 1.3 pt 9 ps 0.9 notitle, \
     "" skip 1 using 1:2 with linespoints lc rgb AMBER lw 2.3 pt 9 ps 1.0 title "5% nested subcalls", \
     "" skip 1 using 1:5:6:7 with yerrorbars lc rgb PURPLE lw 1.3 pt 5 ps 0.9 notitle, \
     "" skip 1 using 1:5 with linespoints lc rgb PURPLE lw 2.3 pt 5 ps 0.95 title "15% nested subcalls", \
     "" skip 1 using 1:8:9:10 with yerrorbars lc rgb CRIMSON lw 1.3 pt 7 ps 0.9 notitle, \
     "" skip 1 using 1:8 with linespoints lc rgb CRIMSON lw 2.5 pt 7 ps 1.0 title "30% nested subcalls"

unset object 1
unset label 1
unset label 2

# --- Panel 3B: Mean Time-to-Deadlock @ 30% Nested ---
set title "B. Mean Time-to-Deadlock at 30% Nested Calls (ms)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set xlabel "Connection Pool Size (P)" font "Helvetica,11" tc rgb "#1e293b"
set ylabel "Mean Time-to-Deadlock (ms)" font "Helvetica,11" tc rgb "#1e293b"
set xrange [3:33.5]
set xtics (4, 6, 8, 10, 12, 16, 20, 24, 32)
set yrange [0:8600]
set ytics 0,1000,8000
set key off

set object 2 rect from 3, 4000 to 33.5, 6000 fc rgb ORANGE fs transparent solid 0.14 noborder behind
set label 3 "Traffic Burst Window (4,000–6,000 ms)\nLarger pools pass steady-state tests,\nthen deadlock mid-burst!" at 3.8, 5300 left font "Helvetica,9" tc rgb "#9a3412" front
set label 4 "Instant in dev\n(< 0.5 s)" at 5.8, 2250 center font "Helvetica,9" tc rgb SLATE
set arrow 1 from 5.8, 1820 to 6.0, 680 lw 1.1 lc rgb SLATE head filled size screen 0.008,20

plot sprintf("%s/deadlock.tsv", DATA_DIR) skip 1 using 1:($11 >= 0 ? $11 : 1/0) with linespoints lc rgb CRIMSON lw 2.4 pt 7 ps 1.05, \
     "" skip 1 using 1:($11 >= 0 ? $11 + 340 : 1/0):(sprintf("%.0f", $11)) with labels font "Helvetica,8.5" tc rgb "#7f1d1d"

unset multiplot
reset

# ==============================================================================
# CHART 4: Mixed-CEM Pareto Frontier & 6D Sobol Global Sensitivity
# ==============================================================================
set terminal pngcairo enhanced truecolor font "Helvetica,11" size 1380,560 rounded
set output sprintf("%s/pareto_and_sensitivity.png", OUT_DIR)

set encoding utf8
set datafile separator "\t"
set border 3 lw 1.2 lc rgb "#334155"
set tics nomirror out scale 0.6 tc rgb "#334155" font "Helvetica,10"
set grid back lc rgb GRIDC lt 0 lw 1

set multiplot layout 1,2 title "{/:Bold Automated Policy Optimization (Mixed-CEM Pareto) \\& 6D Sobol Parameter Sensitivity}" \
    font "Helvetica,14" tc rgb "#0f172a" margins 0.06, 0.98, 0.13, 0.87 spacing 0.14, 0.0

# --- Panel 4A: Pareto Frontier (Fast-OLTP P99 vs Overall Success Rate) ---
set title "A. Mixed-CEM Search \\& Non-Dominated Pareto Frontier (80 Evals)" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set xlabel "Fast-OLTP P99 Latency (ms, log_{10} scale)" font "Helvetica,11" tc rgb "#1e293b"
set ylabel "Overall Request Success Rate (%)" font "Helvetica,11" tc rgb "#1e293b"
set logscale x 10
set xrange [14:600]
set xtics (15, 20, 30, 50, 100, 200, 400)
set yrange [93.5:100.85]
set ytics 94,1,100
set key bottom right box lc rgb "#cbd5e1" opaque spacing 1.25 font "Helvetica,9.5"

set arrow 1 from 14, 99.0 to 600, 99.0 nohead dt 2 lw 1.2 lc rgb CRIMSON back
set arrow 2 from 30, 93.5 to 30, 100.85 nohead dt 2 lw 1.2 lc rgb TEAL back
set object 1 rect from 14, 99.0 to 30, 100.85 fc rgb TEAL fs transparent solid 0.12 noborder behind

set label 1 "Target SLO Zone\n(<30ms, ≥99%)" at 14.8, 100.38 left font "Helvetica,9" tc rgb "#115e59" front
set label 2 "split(3), P=20\n23.2ms, 99.2%" at 56, 97.6 left font "Helvetica,9" tc rgb "#0f172a"
set arrow 3 from 54, 97.85 to 24.8, 99.10 lw 1.15 lc rgb NAVY head filled size screen 0.008,20
set label 3 "split(4), P=27\n29.3ms, 99.6%" at 48, 100.42 left font "Helvetica,9" tc rgb "#0f172a"
set arrow 4 from 46, 100.28 to 30.8, 99.65 lw 1.15 lc rgb NAVY head filled size screen 0.008,20
set label 4 "shared, P=27\n230ms, 100%" at 225, 99.32 center font "Helvetica,9" tc rgb SLATE
set arrow 5 from 225, 99.58 to 230, 99.94 lw 1.05 lc rgb SLATE head filled size screen 0.008,20

plot sprintf("%s/pareto_all.tsv", DATA_DIR) skip 1 using ($11 == 0 ? $7 : 1/0):8 with points pt 7 ps 0.9 lc rgb MUTED title "Dominated CEM candidates", \
     sprintf("%s/pareto_frontier.tsv", DATA_DIR) skip 1 using 7:8 with linespoints lc rgb TEAL lw 2.4 pt 7 ps 1.15 title "Pareto frontier (HV = 381.3)"

unset logscale x
unset object 1
unset arrow 1
unset arrow 2
unset arrow 3
unset arrow 4
unset arrow 5
unset label 1
unset label 2
unset label 3
unset label 4

# --- Panel 4B: 6D Sobol Global Sensitivity (eta vs |rho|) ---
set title "B. 6D Sobol Sensitivity on P99: Nonlinear η vs. Monotonic |ρ|" font "Helvetica,12" tc rgb "#1e293b" offset 0,0.4
set xlabel "Sensitivity Magnitude (0 = none, 1 = dominant)" font "Helvetica,11" tc rgb "#1e293b"
unset ylabel
set xrange [0:0.78]
set xtics 0,0.1,0.7
set yrange [0.3:6.7]
set ytics ("Arrival Rate (RPS)" 6, "Slow Query Frac" 5, "USL Coherency (κ)" 4, "Pool Size (P)" 3, "Acquire Timeout" 2, "RPC-Hold Frac" 1) font "Helvetica,9.5"
set style fill solid 0.88 border lc rgb "#1e293b"
set key bottom right box lc rgb "#cbd5e1" opaque spacing 1.2 font "Helvetica,9.5"

set label 5 "U-shaped! Monotonic |ρ| misses\nthe pool-size knee, while η catches it" at 0.36, 2.25 left font "Helvetica,8.5" tc rgb CRIMSON
set arrow 6 from 0.36, 2.45 to 0.33, 3.08 lw 1.15 lc rgb CRIMSON head filled size screen 0.008,20

plot sprintf("%s/sensitivity.tsv", DATA_DIR) skip 1 using ($4/2.0):(7-$1+0.17):(0):($4):(7-$1+0.02):(7-$1+0.33) with boxxyerror fc rgb NAVY title "Sobol quartile η (nonlinear)", \
     "" skip 1 using ($6/2.0):(7-$1-0.17):(0):($6):(7-$1-0.33):(7-$1-0.02) with boxxyerror fc rgb ORANGE title "Spearman |ρ| (monotonic)", \
     "" skip 1 using ($4+0.015):(7-$1+0.17):(sprintf("%.3f", $4)) with labels left font "Helvetica,8.5" tc rgb NAVY notitle, \
     "" skip 1 using ($6+0.015):(7-$1-0.17):(sprintf("%+.3f", $5)) with labels left font "Helvetica,8.5" tc rgb "#9a3412" notitle

unset multiplot
reset
