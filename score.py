"""
Inference module for the Paris 15e property valuation model.

Usage:
    from score import load_model, score_listing
    model = load_model("artifacts/model.json")
    result = score_listing(model, surface=50, nb_room=2, is_appartement=1,
                           parcel_id="75115000CG0058", price=350000)
"""

import json
import math


def load_model(path="artifacts/model.json"):
    with open(path) as f:
        return json.load(f)


def predict(model, surface, nb_room, is_appartement, parcel_id):
    coefs = model["coefficients"]
    section_id = parcel_id[:-4]
    section_mean = model["section_means"].get(section_id, model["global_mean_log_price"])

    log_price = (
        coefs["intercept"]
        + coefs["surface_reelle_bati"] * surface
        + coefs["nombre_pieces_principales"] * nb_room
        + coefs["is_appartement"] * is_appartement
        + coefs["section_location"] * section_mean
    )
    return math.exp(log_price)


def score_listing(model, surface, nb_room, is_appartement, parcel_id, price):
    expected = predict(model, surface, nb_room, is_appartement, parcel_id)

    z = (math.log(price) - math.log(expected)) / model["sigma"]

    if z < -2:
        label = "strongly_underpriced"
    elif z < -1:
        label = "slightly_underpriced"
    elif z <= 1:
        label = "fair"
    elif z <= 2:
        label = "slightly_overpriced"
    else:
        label = "strongly_overpriced"

    return {
        "expected_price": round(expected),
        "gap_eur":        round(price - expected),
        "gap_pct":        round((price - expected) / expected * 100, 1),
        "z_score":        round(z, 3),
        "label":          label,
        "score":          round(max(0.0, 1 - abs(z) / 4), 3),
    }