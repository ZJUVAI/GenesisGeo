# Known issues in `larger_imo_eval.txt`

## Three statements appear to be false as written

Reported by an independent symbolic prover. While running `larger_imo_eval.txt` through an independent symbolic geometry prover, three statements came back refuted numerically: the constructions all draw fine, every fact they assert holds, and the **goal is false in every figure we draw**. I've included explicit coordinates so you can check each one without running any of our code.

File: `benchmarks/larger_imo_eval.txt`, sha256 `932f755f2d7ea8158d388296c5c2752cea37bd0bab7a2c435118ee5ebceb6db9`

**Method.** For each statement we draw the configuration **without** aiming at the goal (a sketcher that retries until the goal holds would hide exactly this), confirm that every fact the constructions assert holds in the drawn figure, then evaluate the goal. 200 attempts each.

**Important caveat up front.** We evaluate against AlphaGeometry's `defs.txt`, not Newclid's. If Newclid's definition of any construction below differs from AlphaGeometry's, that would explain the disagreement and this report is wrong — so I've listed the exact asserted facts we used for every construction involved. None of these three uses `eq_triangle`, which is the one definition we know differs between the two.

---

#### 1. `translated_imo_2005_p1` (lines 353–354)

```
a b c = ieq_triangle a b c; a1 = on_line a1 b c; a2 = on_line a2 b c; b1 = eqdistance b1 a2 a1 a2, on_line b1 c a; b2 = eqdistance b2 b1 b1 a2, on_line b2 c a; c1 = eqdistance c1 b2 b2 b1, on_line c1 a b; c2 = eqdistance c2 c1 c1 b2, on_line c2 a b; o = on_line o a1 b2, on_line o b1 c2 ? coll o c1 a2
```

200 of 200 figures built; **the goal holds in 0 of them.** All 16 asserted facts hold in the figure below; `coll o c1 a2` has scale-free residual `0.0188`.

```
   a = (-0.527903820525, -0.793667931539)
  a1 = (-0.106643037415, -0.216184228379)
  a2 = (-0.218299270028, -0.738807962139)
   b = (-0.207883514779, -0.690055458395)
  b1 = (-0.033112021779, -1.240114619243)
  b2 = (-0.429889335529, -0.882105618429)
   c = (-0.278162633761, -1.019007409470)
  c1 = (+0.031557110248, -0.612532142267)
  c2 = (-0.476876436116, -0.777146875213)
   o = (-0.411864357617, -0.844972267121)
```

#### 2. `translated_imo_2007_p2` (lines 355–356)

```
a b c = triangle a b c; d = parallelogram a b c d; e = on_circum e b c d; g = eqdistance g e e c, on_line g b c; f = eqdistance f e e g, on_line f d c; x = angle_bisector x d a b ? coll x f g
```

200 of 200 built; **goal holds in 0.** All 10 asserted facts hold below; `coll x f g` residual `0.496`.

```
   a = (+0.567597178070, -0.393374547842)
   b = (-0.046806091695, +0.166764078910)
   c = (+0.816225770391, +0.009373711635)
   d = (+1.430629040156, -0.550764915117)
   e = (-0.189581230301, +0.144690564624)
   f = (-0.417061640925, +1.133736082346)
   g = (-1.178405037967, +0.373132790874)
   x = (+0.925421393245, +0.329195632040)
```

#### 3. `translated_imo_2018_sl_g2` (lines 407–408)

```
a b c = iso_triangle a b c; m = midpoint m b c; p = on_pline p a b c; x = on_line x p b; y = on_line y p c, eqangle3 y p m x p m ? cyclic a p x y
```

200 of 200 built; **goal holds in 0.** All 8 asserted facts hold below; `cyclic a p x y` residual `0.790`.

```
   a = (+0.496739772050, +0.702372064539)
   b = (-0.046806091695, +0.166764078910)
   c = (+0.816225770391, +0.009373711635)
   m = (+0.384709839348, +0.088068895272)
   p = (-0.552282109920, +0.893681269374)
   x = (-0.098259646942, +0.240758635468)
   y = (+0.659003124525, +0.110968430954)
```

---

**The construction semantics we used** (AlphaGeometry `defs.txt`, line 4 of each block — the facts each construction asserts):

| construction | asserted facts |
|---|---|
| `ieq_triangle a b c` | `cong a b b c`, `cong b c c a`, `eqangle a b a c c a c b`, `eqangle c a c b b c b a` |
| `iso_triangle a b c` | `eqangle b a b c c b c a`, `cong a b a c` |
| `on_line x a b` | `coll x a b` |
| `on_pline x a b c` | `para x a b c` |
| `on_circum x a b c` | `cyclic a b c x` |
| `eqdistance x a b c` | `cong x a b c` |
| `midpoint x a b` | `coll x a b`, `cong x a x b` |
| `parallelogram a b c x` | `para a b c x`, `para a x b c`, `cong a b c x`, `cong a x b c` |
| `angle_bisector x a b c` | `eqangle b a b x b x b c` |
| `eqangle3 x a b d e f` | `eqangle x a x b d e d f` |

**Why this might matter to you.** An execution-based metric — "does the statement build a figure" — scores all three of these as fine, because they do build. Only testing the goal in a figure drawn *without* the goal as a target separates them. If these are intended to be theorems, the formalisations look like they've lost a configuration constraint (a side vs a line, an arc, an orientation) somewhere in translation.

Happy to be told I've mis-read a definition — the coordinates above should make that quick to settle either way.

---

## Open question: is `translated_imo_2013_p3` constructible? `o1` looks over-determined

File: `benchmarks/larger_imo_eval.txt`, lines 357–358, sha256 `932f755f2d7ea8158d388296c5c2752cea37bd0bab7a2c435118ee5ebceb6db9`

```
a b c = triangle a b c; a1 b1 c1 e = excenter2 a1 b1 c1 e a b c; o1 = circumcenter o1 a1 b1 c1, on_circum o1 a b c ? perp a b a c
```

**This is a question, not a bug report** — I think the limitation may well be on our side and I'd like to know which.

Our sketcher fails to build a figure for this in 200 attempts, reporting `o1: constructions disagree`. Reading the statement, that looks structurally expected rather than accidental:

- `circumcenter o1 a1 b1 c1` asserts `cong o1 a1 o1 b1, cong o1 b1 o1 c1`, which determines `o1` **uniquely** from `a1 b1 c1`.
- `on_circum o1 a b c` then asserts `cyclic a b c o1` — one further equation, but `o1` has no freedom left.

So the extra condition is really a constraint on `a b c`, which a left-to-right sketcher has already placed. Satisfying it needs solving backwards for the triangle whose excentral-triangle circumcentre lands on its own circumcircle — which is presumably the actual content of IMO 2013 P3, and the goal `perp a b a c` is the conclusion drawn *from* that condition.

Two questions:

1. **Does Newclid build this one?** If yes, the limitation is ours and I'd be glad to know how the constraint is discharged — that would be useful to us well beyond this problem.
2. **If not, is the intended reading a conditional** ("for triangles where this circumcentre lies on the circumcircle, the angle at A is right")? If so, the DSL as written does not seem able to state the antecedent, since every construction places a new point and none constrains points already placed. That would be a limitation of the language rather than of this line, and worth recording somewhere for the benchmark's other users.

I'm filing this separately from the three statements in the other issue because those are refuted by explicit counterexample figures, whereas this one we simply cannot construct — a weaker and different claim, and I didn't want to present them as the same kind of finding.
