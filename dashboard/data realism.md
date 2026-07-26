# Data Realism Audit — Round 2

The first audit round removed six artifacts that were inflating results. This
round removes the three that survived it, and they were the largest.

## The finding

Every injected attack was defined by a signal the benign baseline never
produced. Measured on the previous dataset:

| Rule | Rows | Anomaly purity | Share of all anomalies |
|---|---|---|---|
| `auth_result == "failure"` | 975 | **100.00%** | 62.2% |
| `distinct_entities_per_ip_1h >= 5` | 301 | **100.00%** | 19.2% |
| `is_new_device_for_entity == 1` | 645 | 89.92% | 37.0% |
| `is_new_resource_for_entity == 1` | 1,122 | 49.29% | 35.3% |

`generate_normal_sessions()` hardcoded `auth_result="success"` on all 72,526
benign rows, drew a fresh random public IP for every session, and only ever
selected resources and devices from the entity's known set. Consequently:

- A two-condition rule with no model at all filled the **entire top-1% analyst
  alert budget** with guaranteed true positives (292 qualifying rows in the test
  window against a 185-alert budget).
- The reported "0 false positives at a realistic alert budget" measured the
  simulator, not the detector.
- The anomaly-score distribution was degenerate: the top-1% threshold sat at
  1.0000 and the 5% threshold at 0.0003, with nothing in between.

The generator docstring also claimed that "rare legitimate travel is modeled
separately… legitimate travel respects plausible flight speeds." No legitimate
travel existed in the code, so any non-home city was an attack by construction.

## The fix

Four benign behaviours were added, each with a real-world referent rather than
a value chosen to move a metric.

| Behaviour | Implementation | Breaks |
|---|---|---|
| Password typos, expired certs | 2–6% of user sessions preceded by a burst of 1–3 failures, **from the entity's own device and IP** | failure = attack |
| Shared office egress IPs | One NAT egress address per city; 65% of on-site sessions use it | IP fan-out = credential stuffing |
| Second devices + hardware refresh | Occasional personal phone (5% of sessions); 15% of entities issued new kit mid-timeline | new device = spoofing |
| Genuine business travel | 18% of users take 1–2 trips of 2–5 days; outbound and return days carry **no** sessions, guaranteeing a ≥24h gap and therefore <900 km/h implied velocity for every city pair | foreign city = attack |
| Legitimate scope change | 3% of sessions touch an out-of-profile resource; 30% of those become permanent | novel resource = lateral movement |

Post-fix purity of the same four rules: **31.4%**, **5.4%**, 67.8%, 30.8%. The
detector now has to learn a *threshold* — how many failures, how much fan-out,
how fast the implied travel — instead of a boolean.

## Effect on every reported metric

| Metric | Before | After |
|---|---|---|
| PR-AUC | 0.9995 | **0.9849** |
| Macro-F1 (7 classes) | 0.964 | **0.873** |
| Correct attack type given detected | 98.4% | **95.8%** |
| Recall at top-1% budget | 49.9% | **39.3%** |
| False positives at top-1% budget | 0 / 185 | **0 / 186** |
| Cold-start FP rate | 0.00% | **0.00%** |
| Drift FP rate | 0.00% | **0.00%** |
| Class imbalance | 1:48 | 1:38 |

Every headline number fell except the false-positive results, which held. That
matters: before the fix, "0 false positives" was explained by the label oracles
and told you nothing. After the fix it is earned.

Threshold sweep on the held-out window, which is now informative:

| Alert budget | Precision | Recall | Cold-start FP | Drift FP |
|---|---|---|---|---|
| 0.5% | 100.0% | 19.6% | 0.00% | 0.00% |
| **1%** | **100.0%** | **39.3%** | **0.00%** | **0.00%** |
| **2%** | **99.7%** | **78.1%** | **0.00%** | **0.00%** |
| 5% | 51.0% | 100.0% | 5.5% | 7.9% |
| 10% | 25.5% | 100.0% | 15.4% | 16.9% |

A 2% budget — roughly 29 alerts/day for this population — is the recommended
operating point: 78% recall at 99.7% precision with no cold-start or drift
false positives.

## What broke, and what that revealed

Two classes collapsed when benign confusability was introduced:

- **Low-and-slow exfiltration: F1 0.98 → 0.59** (recall 0.49). Benign off-hours
  work and travel now looked identical to it.
- **Lateral movement: F1 0.95 → 0.56** (precision 0.41). A colleague hand-off
  produces a never-before-seen resource, so novelty alone stopped meaning
  anything.

Both were previously riding on the oracles. Two causal features were added in
response, each targeting the property that actually defines the attack rather
than a proxy for it:

- `offhours_access_count_entity_7d` — low-and-slow is defined by
  **accumulation** over days, not by any single off-hours access. This restored
  the class to **F1 0.90 (recall 0.94)**.
- `novel_resources_entity_1h` — lateral movement is defined by **breadth in a
  short window**: 5–10 unseen resources in under an hour, versus one for a
  hand-off. Mean value is 2.63 for lateral movement against 0.016 for
  everything else, and the feature reaches **85.7% purity at ≥4**.

Lateral movement nonetheless remains the weakest class at **F1 0.50 (precision
0.34, recall 0.91)**, and this is an operating-point consequence rather than a
feature failure. Inverse-frequency class weighting on a class with n=34 in test,
competing against 1,023 benign novel-resource events, optimises recall at the
cost of precision. Precision is recoverable by thresholding on
`novel_resources_entity_1h`, at the cost of the ~18% of incidents that stay
below 4 hops per hour. **Lateral movement is the honest open problem in this
system**, and it is the one a real SOC would tune hardest.

## Limitation this does not fix

The remaining attacks are still rule-generated and therefore still more
separable than real intrusions — the score distribution remains
bimodal, just no longer degenerate. An adaptive adversary that deliberately
stays under each threshold (3 failed attempts per hour rather than 15–60, two
lateral hops per hour rather than 5–10) is not represented in this data and
would substantially reduce recall. Validation against a real access-log corpus
or a red-team exercise remains the necessary next step before any production
claim.
