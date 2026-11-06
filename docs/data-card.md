# Data card

**Source:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), specifically `bank-additional/bank-additional-full.csv` inside `bank-additional.zip`, inside the repository's combined download. Do not substitute the older 45,211-row `bank-full.csv` or the randomly sampled 4,119-row file.

**Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306). UCI lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as the license. Redistribution or adaptation must retain attribution.

**Provenance:** 41,188 rows, 20 source inputs plus `y`. The source describes date ordering from May 2008 through November 2010. The dataset reflects telephone marketing at a Portuguese banking institution. The label records whether a term deposit was subscribed. Client contact history, demographics, calendar fields, and economic context are included. This is observational historical data; it does not identify the causal effect of contacting a customer.

| File | SHA-256 |
|---|---|
| Combined UCI source ZIP | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` |
| Exact CSV, 5,834,924 bytes | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` |

The download command verifies both hashes before writing the selected exact member. It does not unpack arbitrary archive paths. A source revision fails closed and requires explicit review. The downloaded names file and a provenance JSON are retained in `data/`; this directory is excluded from source control. The recorded training report also embeds provenance. No resume or saved application data enters this project.

**Feature availability:** The names file explicitly warns that `duration` is unknown before a call and should be discarded for realistic predictive modeling. ModelGate discards it for every candidate, even when present in the raw input. `campaign` includes the current contact; the experiment assumes the contact count and planned contact channel/calendar fields are available immediately before the modeled call. This assumption should be checked against any real operational event schema. Economic variables have publication and revision timing risks that the dataset does not resolve; the winning ablation removes them.

**Missingness:** UCI's webpage and original descriptions differ in how they characterize missing data. The names file documents categorical `unknown` values. ModelGate preserves `unknown` as a category, maps explicit categorical null to it, imputes numeric null with training medians, and keeps `pdays=999` as the original “not previously contacted” sentinel. Missing values are never inferred from an absent JSON key. Unknown inference-time categories are encoded as all-zero one-hot vectors and counted separately in the drift profile.

**Split limitations:** The source provides date order but no exact timestamps or stable client identifiers. Row-contiguous partitions approximate temporal deployment. A client may have multiple contacts across partitions, so temporal separation does not prove client-disjoint separation. Row order cannot support a precise per-day forecast. Chronological changes in label prevalence, customer mix, campaigns, and macroeconomic conditions are major confounders.

**Use limitations:** Ages, occupations, marital status, and education can encode sensitive distinctions and proxies. Public availability does not establish suitability for targeting, fairness, representativeness, consent in a new context, or current financial use. This project uses the data only for a local historical modeling exercise. Slices are descriptive and cannot certify fairness. It is not a lending/credit score and must not be relabeled as one.
