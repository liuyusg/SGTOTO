"""
Pattern analysis for Draw 4193  —  T-1 (draw 4192): [2, 7, 11, 19, 20, 42]
                                    T-2 (draw 4191): [9, 16, 27, 28, 32, 49]
                                    T-3 (draw 4190): [14, 16, 21, 22, 35, 38]

Patterns:
  A. Same last digit as a T-1 number
  B. Adjacent ±1 or ±2 to a T-1 number
  C. Pair co-occurrence in same draw as T-1 number
  D. Sum of two T-1 numbers
  E. Draws with a pair summing to 30 or 40 → elevated next-draw rates
  F. Numbers whose pair-sum in T-1 also appeared as a pair-sum in T-2 (shared sum)
  G. Draws where two numbers differ by 30 → elevated next-draw rates
  H. Draws where two numbers differ by 40 → elevated next-draw rates
"""
import sqlite3
from collections import Counter
from itertools import combinations

T1 = [2, 7, 11, 19, 20, 42]
T2 = [9, 16, 27, 28, 32, 49]
T3 = [14, 16, 21, 22, 35, 38]
T1_SET = set(T1)
BASELINE = 6 / 49
MIN_LIFT = 1.30
MIN_DRAWS = 15

conn = sqlite3.connect('toto.sqlite')
cur  = conn.cursor()

cur.execute("""
    SELECT d.draw_no, GROUP_CONCAT(j.number, ',')
    FROM draws d
    JOIN jackpot_no j ON d.draw_no=j.draw_no AND j.no_type='normal'
    WHERE d.scanned=1
    GROUP BY d.draw_no ORDER BY d.draw_no
""")
rows      = cur.fetchall()
all_draws = [(r[0], [int(n) for n in r[1].split(',')]) for r in rows]
N         = len(all_draws)
draw_map  = {dn: set(nums) for dn, nums in all_draws}

def transitions_from(src):
    counts = Counter()
    total  = 0
    for i in range(N - 1):
        if src in draw_map[all_draws[i][0]]:
            for n in all_draws[i + 1][1]:
                counts[n] += 1
            total += 1
    return counts, total

score_A  = Counter()
score_B1 = Counter()
score_B2 = Counter()
score_C  = Counter()
score_D  = Counter()
score_E  = Counter()
score_F  = Counter()
score_G  = Counter()   # gap = 30
score_H  = Counter()   # gap = 40

def next_draw_counts_when(condition_fn):
    """Returns (counter_of_next_nums, n_qualifying_draws)."""
    counts = Counter()
    total  = 0
    for i in range(N - 1):
        dn, nums = all_draws[i]
        if condition_fn(draw_map[dn]):
            total += 1
            for n in all_draws[i + 1][1]:
                counts[n] += 1
    return counts, total

# ── A. SAME LAST DIGIT ───────────────────────────────────────────────────────
print('=' * 70)
print('A. SAME LAST DIGIT')
for src in T1:
    d = src % 10
    counts, total = transitions_from(src)
    if total < MIN_DRAWS:
        continue
    hits = [(n, counts[n]/total, counts[n]/total/BASELINE)
            for n in range(1, 50) if n % 10 == d and n != src
            and counts[n]/total >= BASELINE * MIN_LIFT]
    if hits:
        hits.sort(key=lambda x: -x[2])
        print(f'\n  {src} (last digit {d})  [n={total}]')
        for n, rate, lift in hits:
            print(f'    → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
            score_A[n] += lift
if not score_A:
    print('  (none above threshold)')

# ── B. ADJACENT ±1 / ±2 ──────────────────────────────────────────────────────
print()
print('=' * 70)
print('B. ADJACENT ±1 / ±2')
for src in T1:
    counts, total = transitions_from(src)
    if total < MIN_DRAWS:
        continue
    printed = False
    for delta, sbucket in [(1, score_B1), (2, score_B2)]:
        for sign in [-1, 1]:
            n = src + sign * delta
            if 1 <= n <= 49 and n not in T1_SET:
                rate = counts[n] / total
                lift = rate / BASELINE
                if lift >= MIN_LIFT:
                    if not printed:
                        print(f'\n  {src}  [n={total}]')
                        printed = True
                    print(f'    → {n:3d}  Δ={n-src:+d}  rate={rate:.1%}  lift={lift:.2f}x')
                    sbucket[n] += lift
if not score_B1 and not score_B2:
    print('  (none above threshold)')

# ── C. PAIR CO-OCCURRENCE IN SAME DRAW ──────────────────────────────────────
print()
print('=' * 70)
print('C. PAIR CO-OCCURRENCE  (in same draw as T-1 number — historical)')
for src in T1:
    same_draw = Counter()
    n_src = 0
    for dn, nums in all_draws:
        if src in set(nums):
            n_src += 1
            for n in nums:
                if n != src:
                    same_draw[n] += 1
    hits = [(n, same_draw[n]/n_src, same_draw[n]/n_src/BASELINE)
            for n in range(1, 50) if n != src
            and same_draw[n]/n_src >= BASELINE * MIN_LIFT]
    hits.sort(key=lambda x: -x[2])
    if hits:
        print(f'\n  {src} (appeared {n_src}x):')
        for n, rate, lift in hits[:8]:
            print(f'    → {n:3d}  co-rate={rate:.1%}  lift={lift:.2f}x')
            score_C[n] += lift

# ── D. SUM OF TWO T-1 NUMBERS ───────────────────────────────────────────────
t1_sums = {(a + b): (a, b) for a, b in combinations(T1, 2) if 1 <= a + b <= 49}
print()
print('=' * 70)
print(f'D. SUMS OF TWO T-1 NUMBERS  →  {sorted(t1_sums)}')
print('   (checking: when those two appeared together, how often did their sum appear next?)')
for s, (a, b) in sorted(t1_sums.items()):
    counts = Counter()
    total  = 0
    for i in range(N - 1):
        nums_i = draw_map[all_draws[i][0]]
        if a in nums_i and b in nums_i:
            for n in all_draws[i + 1][1]:
                counts[n] += 1
            total += 1
    if total < MIN_DRAWS:
        print(f'  {s:3d} = {a}+{b}  (only {total} co-draws, skip)')
        continue
    rate = counts[s] / total
    lift = rate / BASELINE
    mark = '  ★' if lift >= MIN_LIFT else ''
    print(f'  {s:3d} = {a}+{b}  next_rate={rate:.1%}  lift={lift:.2f}x  [n={total}]{mark}')
    if lift >= MIN_LIFT:
        score_D[s] += lift

# overall frequency of sum numbers
print('\n  Overall historical frequency:')
for s in sorted(t1_sums):
    freq = sum(1 for _, nums in all_draws if s in nums) / N
    lift = freq / BASELINE
    mark = ' ★' if lift >= MIN_LIFT else ''
    print(f'  {s:3d}  freq={freq:.1%}  lift={lift:.2f}x{mark}')
    if lift >= MIN_LIFT:
        score_D[s] += lift * 0.5

# ── E. PAIR SUM = 30 OR 40 ───────────────────────────────────────────────────
print()
print('=' * 70)
print('E. DRAWS WITH A PAIR SUMMING TO 30 OR 40  → next-draw elevated numbers')

# Which of our T-1 / T-2 / T-3 pairs hit these sums?
for label, nums in [('T-1', T1), ('T-2', T2), ('T-3', T3)]:
    sums30 = [(a, b) for a, b in combinations(nums, 2) if a + b == 30]
    sums40 = [(a, b) for a, b in combinations(nums, 2) if a + b == 40]
    if sums30 or sums40:
        print(f'  {label}: sum=30 pairs {sums30}  sum=40 pairs {sums40}')
    else:
        print(f'  {label}: no pair sums to 30 or 40')

for target_sum in [30, 40]:
    counts, total = next_draw_counts_when(
        lambda nums, ts=target_sum: any(a + b == ts for a, b in combinations(nums, 2))
    )
    if total < MIN_DRAWS:
        print(f'\n  sum={target_sum}: only {total} qualifying draws, skip')
        continue
    hits = [(n, counts[n]/total, counts[n]/total/BASELINE)
            for n in range(1, 50)
            if counts[n]/total >= BASELINE * MIN_LIFT]
    hits.sort(key=lambda x: -x[2])
    print(f'\n  Draws with a pair summing to {target_sum}  [n={total}]:')
    if hits:
        for n, rate, lift in hits[:10]:
            print(f'    → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
            score_E[n] += lift
    else:
        print('    (none above threshold)')

# ── F. SHARED PAIR-SUM BETWEEN T-1 AND T-2 ───────────────────────────────────
print()
print('=' * 70)
print('F. SHARED PAIR-SUM IN CONSECUTIVE DRAWS (same sum in T-1 AND T-2)')

t1_pair_sums = {a + b for a, b in combinations(T1, 2) if 1 <= a + b <= 49}
t2_pair_sums = {a + b for a, b in combinations(T2, 2) if 1 <= a + b <= 49}
t3_pair_sums = {a + b for a, b in combinations(T3, 2) if 1 <= a + b <= 49}
shared_t1_t2 = t1_pair_sums & t2_pair_sums
shared_t1_t3 = t1_pair_sums & t3_pair_sums

print(f'  T-1 pair sums (1-49): {sorted(t1_pair_sums)}')
print(f'  T-2 pair sums (1-49): {sorted(t2_pair_sums)}')
print(f'  T-3 pair sums (1-49): {sorted(t3_pair_sums)}')
print(f'  Shared T-1 ∩ T-2: {sorted(shared_t1_t2)}')
print(f'  Shared T-1 ∩ T-3: {sorted(shared_t1_t3)}')

# Historically: when draw i and draw i-1 share a pair sum, what appears in draw i+1?
counts_shared = Counter()
total_shared  = 0
for i in range(1, N - 1):
    sums_cur  = {a + b for a, b in combinations(all_draws[i][1],   2) if 1 <= a+b <= 49}
    sums_prev = {a + b for a, b in combinations(all_draws[i-1][1], 2) if 1 <= a+b <= 49}
    if sums_cur & sums_prev:
        total_shared += 1
        for n in all_draws[i + 1][1]:
            counts_shared[n] += 1

print(f'\n  Draws where consecutive pair-sums overlap  [n={total_shared}]:')
if total_shared >= MIN_DRAWS:
    hits = [(n, counts_shared[n]/total_shared, counts_shared[n]/total_shared/BASELINE)
            for n in range(1, 50)
            if counts_shared[n]/total_shared >= BASELINE * MIN_LIFT]
    hits.sort(key=lambda x: -x[2])
    if hits:
        for n, rate, lift in hits[:10]:
            print(f'    → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
            score_F[n] += lift
    else:
        print('    (none above threshold)')

# Specifically for shared sums from T-1 ∩ T-2 — which sum, and what number predictions?
if shared_t1_t2:
    for s in sorted(shared_t1_t2):
        t1_pairs = [(a,b) for a,b in combinations(T1,2) if a+b==s]
        t2_pairs = [(a,b) for a,b in combinations(T2,2) if a+b==s]
        print(f'\n  Shared sum {s}: T-1 via {t1_pairs}, T-2 via {t2_pairs}')
        # When both current and previous draw had a pair summing to s, what came next?
        c2 = Counter()
        t2 = 0
        for i in range(1, N - 1):
            s_cur  = {a+b for a,b in combinations(all_draws[i][1],   2) if a+b==s}
            s_prev = {a+b for a,b in combinations(all_draws[i-1][1], 2) if a+b==s}
            if s_cur and s_prev:
                t2 += 1
                for n in all_draws[i + 1][1]:
                    c2[n] += 1
        if t2 >= MIN_DRAWS:
            print(f'    Both draws had sum={s}  [n={t2}]:')
            h2 = [(n, c2[n]/t2, c2[n]/t2/BASELINE) for n in range(1,50)
                  if c2[n]/t2 >= BASELINE * MIN_LIFT]
            for n, rate, lift in sorted(h2, key=lambda x: -x[2])[:8]:
                print(f'      → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
                score_F[n] += lift * 1.5
        else:
            print(f'    (only {t2} draws with both having sum={s})')

# ── G. SUBTRACTION = 30 (gap between two numbers) ───────────────────────────
print()
print('=' * 70)
print('G. TWO NUMBERS WITH DIFFERENCE = 30  → next-draw elevated numbers')

for label, nums in [('T-1', T1), ('T-2', T2), ('T-3', T3)]:
    g30 = [(min(a,b), max(a,b)) for a, b in combinations(nums, 2) if abs(a - b) == 30]
    print(f'  {label}: diff-30 pairs {g30 if g30 else "(none)"}')
print('  Numbers 30 away from each T-1 number:')
for src in T1:
    for delta in [-30, 30]:
        n = src + delta
        if 1 <= n <= 49 and n not in T1_SET:
            print(f'    {src} ± 30 → {n}')

counts_g, total_g = next_draw_counts_when(
    lambda nums: any(abs(a - b) == 30 for a, b in combinations(nums, 2))
)
print(f'\n  Draws with two numbers 30 apart  [n={total_g}]:')
if total_g >= MIN_DRAWS:
    hits = [(n, counts_g[n]/total_g, counts_g[n]/total_g/BASELINE)
            for n in range(1, 50)
            if counts_g[n]/total_g >= BASELINE * MIN_LIFT]
    hits.sort(key=lambda x: -x[2])
    if hits:
        for n, rate, lift in hits[:10]:
            print(f'    → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
            score_G[n] += lift
    else:
        print('    (none above threshold)')

# ── H. SUBTRACTION = 40 (gap between two numbers) ────────────────────────────
print()
print('=' * 70)
print('H. TWO NUMBERS WITH DIFFERENCE = 40  → next-draw elevated numbers')

for label, nums in [('T-1', T1), ('T-2', T2), ('T-3', T3)]:
    g40 = [(min(a,b), max(a,b)) for a, b in combinations(nums, 2) if abs(a - b) == 40]
    print(f'  {label}: diff-40 pairs {g40 if g40 else "(none)"}')
print('  Numbers 40 away from each T-1 number:')
for src in T1:
    for delta in [-40, 40]:
        n = src + delta
        if 1 <= n <= 49 and n not in T1_SET:
            print(f'    {src} ± 40 → {n}')

counts_h, total_h = next_draw_counts_when(
    lambda nums: any(abs(a - b) == 40 for a, b in combinations(nums, 2))
)
print(f'\n  Draws with two numbers 40 apart  [n={total_h}]:')
if total_h >= MIN_DRAWS:
    hits = [(n, counts_h[n]/total_h, counts_h[n]/total_h/BASELINE)
            for n in range(1, 50)
            if counts_h[n]/total_h >= BASELINE * MIN_LIFT]
    hits.sort(key=lambda x: -x[2])
    if hits:
        for n, rate, lift in hits[:10]:
            print(f'    → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
            score_H[n] += lift
    else:
        print('    (none above threshold)')

# Bonus: when T-1 AND T-2 both have a diff-40 pair, what appears elevated next?
t1_has_g40 = any(abs(a-b)==40 for a,b in combinations(T1,2))
t2_has_g40 = any(abs(a-b)==40 for a,b in combinations(T2,2))
if t1_has_g40 and t2_has_g40:
    print('\n  ★ Both T-1 AND T-2 have a diff-40 pair — checking consecutive-draw bonus:')
    counts_hh = Counter()
    total_hh  = 0
    for i in range(1, N - 1):
        cur_g40  = any(abs(a-b)==40 for a,b in combinations(all_draws[i][1],   2))
        prev_g40 = any(abs(a-b)==40 for a,b in combinations(all_draws[i-1][1], 2))
        if cur_g40 and prev_g40:
            total_hh += 1
            for n in all_draws[i+1][1]:
                counts_hh[n] += 1
    print(f'  Both consecutive draws had diff-40 pair  [n={total_hh}]:')
    if total_hh >= MIN_DRAWS:
        hhits = [(n, counts_hh[n]/total_hh, counts_hh[n]/total_hh/BASELINE)
                 for n in range(1, 50)
                 if counts_hh[n]/total_hh >= BASELINE * MIN_LIFT]
        for n, rate, lift in sorted(hhits, key=lambda x: -x[2])[:10]:
            print(f'      → {n:3d}  rate={rate:.1%}  lift={lift:.2f}x')
            score_H[n] += lift * 1.5
    else:
        print(f'    (only {total_hh} qualifying consecutive pairs, skip)')

# ── COMBINED RANKING ─────────────────────────────────────────────────────────
print()
print('=' * 70)
print('COMBINED RANKING  (weights: A×1, B1×1, B2×0.8, C×1, D×0.7, E×0.9, F×1.1, G×0.9, H×1.0)')
print('=' * 70)
all_nums = (set(score_A) | set(score_B1) | set(score_B2) | set(score_C) |
            set(score_D) | set(score_E) | set(score_F) | set(score_G) | set(score_H))

def row(n):
    a  = score_A.get(n, 0)
    b1 = score_B1.get(n, 0)
    b2 = score_B2.get(n, 0) * 0.8
    c  = score_C.get(n, 0)
    d  = score_D.get(n, 0) * 0.7
    e  = score_E.get(n, 0) * 0.9
    f  = score_F.get(n, 0) * 1.1
    g  = score_G.get(n, 0) * 0.9
    h  = score_H.get(n, 0) * 1.0
    return (n, a, b1, b2, c, d, e, f, g, h, a+b1+b2+c+d+e+f+g+h)

combined = sorted(
    [row(n) for n in all_nums if n not in T1_SET],
    key=lambda x: -x[10]
)
hdr = (f'{"#":<4}{"Num":<5}{"A:dig":>6}{"B1":>5}{"B2":>5}{"C:co":>6}'
       f'{"D:sum":>6}{"E:s30/40":>9}{"F:shd":>6}{"G:d30":>6}{"H:d40":>6}{"TOTAL":>7}')
print(hdr)
print('-' * 72)
for i, (n, a, b1, b2, c, d, e, f, g, h, tot) in enumerate(combined[:15], 1):
    print(f'{i:<4}{n:<5}{a:>6.2f}{b1:>5.2f}{b2:>5.2f}{c:>6.2f}{d:>6.2f}{e:>9.2f}{f:>6.2f}{g:>6.2f}{h:>6.2f}{tot:>7.2f}')

top6 = [n for n, *_ in combined][:6]
print(f'\nTop 6: {top6}')
conn.close()
