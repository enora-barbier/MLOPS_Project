"""
Validate a trained Paris 15e valuation model before deployment.
"""

import argparse
import json
import math
import sys
 
# Running a few smoke test predictions and verifying output is in a plausible EUR range
SMOKE_TESTS = [
    ("Studio  (25 m², 1 room, apt)",      25,  1, 1),
    ("2-room  (50 m², 2 rooms, apt)",     50,  2, 1),
    ("3-room  (75 m², 3 rooms, apt)",     75,  3, 1),
    ("Large   (120 m², 5 rooms, apt)",   120,  5, 1),
    ("House   (150 m², 6 rooms)",        150,  6, 0),
]
 
 
def predict(coefs, global_mean, surface, rooms, is_apt):
    log_price = (
        coefs["intercept"]
        + coefs["surface_reelle_bati"] * surface
        + coefs["nombre_pieces_principales"] * rooms
        + coefs["is_appartement"] * is_apt
        + coefs["section_location"] * global_mean
    )
    return math.exp(log_price)
 
 
def validate(model_path):
    with open(model_path) as f:
        m = json.load(f)
 
    coefs       = m["coefficients"]
    global_mean = m["global_mean_log_price"]
    failures    = []
 
    # Checking R² is in a reasonable range
    if m["r_squared"] < 0.30:
        failures.append(f"R² = {m['r_squared']:.4f} is below 0.30")
 
    # Checking coefficient signs (as all should be positive)
    for name, expected in [
        ("surface_reelle_bati", +1),
        ("nombre_pieces_principales", +1),
        ("is_appartement", +1),
        ("section_location", +1),
    ]:
        if math.copysign(1, coefs[name]) != expected:
            failures.append(f"Unexpected sign for '{name}': {coefs[name]:+.6f}")
 
    # Smoke-test predictions in plausible range
    for label, surface, rooms, is_apt in SMOKE_TESTS:
        price = predict(coefs, global_mean, surface, rooms, is_apt)
        # Large range as this is just a sanity check
        if not (50_000 <= price <= 5_000_000):
            failures.append(f"Implausible prediction for {label}: {price:,.0f} €")
 
    # Report
    print(f"Model: {model_path}")
    print(f"R²: {m['r_squared']:.4f}   sigma: {m['sigma']:.4f}   n: {m['n_train']:,}")
    print(f"\n Smoke tests:")
    for label, surface, rooms, is_apt in SMOKE_TESTS:
        price = predict(coefs, global_mean, surface, rooms, is_apt)
        print(f"    {label:<38s} {price:>12,.0f} €")
    print()
    if failures:
        for err in failures:
            print(f"  ✗ {err}")
        print(f"\n  RESULT: Test failed\n")
        return False
    print(f"  RESULT: Test Passed\n")
    return True
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/model.json")
    ok = validate(parser.parse_args().model)
    sys.exit(0 if ok else 1)