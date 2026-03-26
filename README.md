# Paris 15 — Property Valuation & Anomaly Detection Tool

A lightweight decision-support tool for estimating the fair market value of apartments and houses in the **15th arrondissement of Paris**, and flagging listings that deviate significantly from expected prices.

Built as part of the CentraleSupélec MLOps course, referencing the [CESAR](https://github.com/JoachimZ/cesar/tree/v1) baseline.

---

## Objective

Help potential buyers, sellers, and investors answer a simple but critical question:

> *"Is this listing fairly priced given its characteristics and location within the 15th arrondissement?"*

The tool provides a predicted market value, a deviation score, and a human-readable verdict.

---

## Target Users

Any individual considering buying or selling a property in the 15th arrondissement of Paris.

---

## Features

| Feature | Description |
|---|---|
| **Property valuation** | Predicts the market value of an apartment or house using a regression model trained on recent DVF transactions |
| **Anomaly detection** | Flags listings that deviate significantly from expected prices using rule-based thresholds derived from market statistics |
| **Filtering** | Allow users to input surface area, number of rooms, property type, and asking price to get a property valuation estimate |

---

## Data

**Source:** [Explorateur de données de valeurs foncières (DVF)](https://app.dvf.etalab.gouv.fr/) — official French property transaction records.

**Geographic scope:** 15th arrondissement of Paris (`code_departement = 75`), filtered by cadastral section to our zone of interest. We chose this area arbtrarily because we both live here and are curious about the pricing dynamics of this specific part of the market.

**Temporal scope:** Currently using data from 2020 onwards. We plan to restrict this to post-2023 transactions only, to avoid distortions from the COVID period and post-2022 market shifts and ensure the model reflects current conditions.

**Property types:** `Appartement` and `Maison`.

### Features used for training

| Column | Description |
|---|---|
| `surface_reelle_bati` | Living area in m² |
| `nombre_pieces_principales` | Number of main rooms |
| `is_appartement` | Binary flag: 1 = Appartement, 0 = Maison |
| `id_parcelle` | Cadastral parcel ID (proxy for location) |

**Note on location encoding:** Prices in the northern part of the 15th arrondissement tend to be higher than in the south. We initially considered using GPS distance to the centre of Paris as a location feature, but this is impractical for end users (who cannot be expected to provide coordinates). We instead use `id_parcelle`, which uniquely identifies each land parcel and implicitly encodes spatial information. This is a deliberate design choice that preserves usability without sacrificing location signal.

**Target variable:** `valeur_fonciere` (transaction price in €)

---

## Method (Planned)

### Model 

We fit a **log-linear regression** on `log(valeur_fonciere)` using the ln of price rather than the raw price as housing prices tend to vary proportionally rather than by constant euro amounts. using the log of price helps the model capture these proportional effects and reduces the influence of very high-priced observations.

Formally (may be subject to change):
```
log(valeur_fonciere) = b0 + b1 × surface_reelle_bati + b2 × nombre_pieces_principales + b3 × is_appartement + b4 × id_parcelle
```

Parameters `b0 … b4` and the residual standard deviation `σ` are estimated from DVF transactions via ordinary least squares.

We use **ordinary least squares (OLS)** to estimate the coefficients beacause it is simple, interpretable, and well-suited to a dataset of this size (n ≈ 16,000). We also have prior experience with it, making implementation and debugging easier than with more complex models.

### Anomaly detection

Anomalies are flagged based on how far the listed price deviates from the model’s expected price, relative to the usual prediction error observed in the training data. After fitting the model, we compute:

$$
z = \frac{\log(\text{price}) - \log(\text{expected\_price})}{\sigma}
$$

where `σ` is the residual standard deviation estimated on the training data. This measures how unusually high or low a listed price is relative to comparable properties in the dataset.

| Label | Condition |
|---|---|
| `strongly_underpriced` | `z < -2` |
| `slightly_underpriced` | `-2 ≤ z < -1` |
| `fair` | `-1 ≤ z ≤ 1` |
| `slightly_overpriced` | `1 < z ≤ 2` |
| `strongly_overpriced` | `z > 2` |

A **score ∈ [0, 1]** is also returned, where values closer to 1 indicate that the listed price is close to the model’s expected price. The score decreases as the absolute value of the standardised residual increases. This anomaly signal should be interpreted as a measure of **unusual pricing**, not as direct evidence of fraud.

---

## API

The tool is exposed as a **FastAPI** service.

The API takes the key characteristics of a property together with its listed price, computes the model’s expected market value, and returns both a pricing gap and an anomaly assessment.

**Planned endpoint:** `GET /score`

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `surface` | float | Living area in m² |
| `nb_room` | int | Number of main rooms |
| `property_type` | string | Property type (`Appartement` or `Maison`) |
| `section_id` | string | Cadastral section identifier used as a proxy for location |
| `price` | float | Listed price in € |

**Response:**
```json
{
  "expected_price": 320000,
  "gap_eur": 18000,
  "gap_pct": 5.9,
  "z_score": 0.74,
  "label": "fair",
  "score": 0.87
}
```

---

## Web UI

A minimal interface allows users to enter a property’s characteristics and instantly view:

- The **predicted market price**
- The **pricing gap** in € and %
- The **anomaly label** (`strongly_underpriced` / `slightly_underpriced` / `fair` / `slightly_overpriced` / `strongly_overpriced`)
- The **anomaly score** ∈ [0, 1], indicating how closely the listed price aligns with the model’s expected price
- The **methodology and data source** (explained briefly)

---

## Repository Structure

---

## How to Run

---

## Scope and Limitations

---

## Authors

Built by [Enora] and [Yann] as part of the CentraleSupélec MLOps course.  
Reference: [CESAR by Joachim Z](https://github.com/JoachimZ/cesar/tree/v1)
