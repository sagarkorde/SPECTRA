# Dataset

SPECTRA is trained and evaluated on a Bitcoin on-chain transaction corpus in
Apache Parquet format.

## Obtaining the data

The dataset is publicly released on IEEE DataPort:

- **Title:** Bitcoin Blockchain Transaction Dataset for Wallet Address
  Profiling and Behavioral Analysis (Parquet Format)
- **DOI:** [10.21227/bxmt-mn56](https://doi.org/10.21227/bxmt-mn56)
- **Mirror:** https://github.com/sagarkorde/Bitcoin-Blockchain-Transaction-Dataset-for-Wallet-Address-Profiling-and-Behavioral-Analysis

Download the Parquet file and place it at the path referenced by
`configs/config.yaml` (`data.path`), or update that path to point to your
local copy.

## Schema

Rows: 5,884,387 &nbsp;|&nbsp; Columns: 53 &nbsp;|&nbsp; Format: Parquet
(51 row groups)

| # | Column | Type | # | Column | Type |
|---|--------|------|---|--------|------|
| 1 | `txid` | VARCHAR | 28 | `value_difference` | FLOAT |
| 2 | `block_height` | INTEGER | 29 | `fee_rate_sat_per_byte` | FLOAT |
| 3 | `block_time` | BIGINT | 30 | `fee_rate_sat_per_vbyte` | FLOAT |
| 4 | `version` | INTEGER | 31 | `input_address_count` | INTEGER |
| 5 | `locktime` | INTEGER | 32 | `output_address_count` | INTEGER |
| 6 | `size` | INTEGER | 33 | `total_addresses` | INTEGER |
| 7 | `vsize` | INTEGER | 34 | `input_script_count` | INTEGER |
| 8 | `weight` | INTEGER | 35 | `output_script_count` | INTEGER |
| 9 | `input_count` | INTEGER | 36 | `address_reuse` | INTEGER |
| 10 | `output_count` | INTEGER | 37 | `is_self_transfer` | BOOLEAN |
| 11 | `total_input_value` | FLOAT | 38 | `is_consolidation` | BOOLEAN |
| 12 | `total_output_value` | FLOAT | 39 | `is_distribution` | BOOLEAN |
| 13 | `fee` | FLOAT | 40 | `is_peer_to_peer` | BOOLEAN |
| 14 | `input_addresses` | VARCHAR[] | 41 | `is_batch_payment` | BOOLEAN |
| 15 | `output_addresses` | VARCHAR[] | 42 | `is_coinjoin_like` | BOOLEAN |
| 16 | `input_script_types` | VARCHAR[] | 43 | `has_p2pk` | BOOLEAN |
| 17 | `output_script_types` | VARCHAR[] | 44 | `has_p2pkh` | BOOLEAN |
| 18 | `has_coinbase` | BOOLEAN | 45 | `has_p2sh` | BOOLEAN |
| 19 | `has_op_return` | BOOLEAN | 46 | `has_p2wpkh` | BOOLEAN |
| 20 | `op_return_data` | VARCHAR | 47 | `has_p2wsh` | BOOLEAN |
| 21 | `rbf_enabled` | BOOLEAN | 48 | `has_taproot` | BOOLEAN |
| 22 | `timestamp` | TIMESTAMP_NS | 49 | `avg_input_value` | FLOAT |
| 23 | `hour` | INTEGER | 50 | `avg_output_value` | FLOAT |
| 24 | `day_of_week` | INTEGER | 51 | `value_concentration_ratio` | FLOAT |
| 25 | `week_of_year` | INTEGER | 52 | `month_1` | VARCHAR |
| 26 | `year` | INTEGER | 53 | `sample_size` | INTEGER |
| 27 | `input_output_ratio` | FLOAT | | | |

## Descriptive statistics (selected columns)

All 53 columns have `null_count = 0`. Timestamp coverage: `2022-07-13` to
`2025-07-01`.

| Column | Mean | Stddev | Min | Max |
|---|---|---|---|---|
| `block_height` | 815166.3 | 30875.3 | 744837 | 903456 |
| `size` | 548.4 | 4097.6 | 137 | 3,991,717 |
| `vsize` | 325.1 | 1862.5 | 86 | 998,000 |
| `weight` | 1298.8 | 7449.9 | 344 | 3,991,999 |
| `input_count` | 2.28 | 16.73 | 1 | 1,794 |
| `output_count` | 2.79 | 16.76 | 1 | 3,199 |
| `total_input_value` (BTC) | 2.575 | 98.10 | 0.0 | 87,051.0 |
| `fee` (BTC) | 1.29e-4 | 8.22e-3 | ~0.0 | 19.82 |
| `input_output_ratio` | 5.06 | 19.79 | 0.0 | 18,176.7 |

## Class distribution (transaction type)

| Class | Count | % of dataset |
|---|---|---|
| Standard | ~2,869,307 | ~48.8% |
| P2P | 1,842,344 | 31.3% |
| Consolidation | 369,919 | 6.3% |
| Distribution | 554,722 | 9.4% |
| BatchPayment | 137,743 | 2.3% |
| CoinJoin-like | 110,352 | 1.9% |

## Data-quality notes

- `has_taproot` is always `False` despite `input_script_types` /
  `output_script_types` containing `witness_v1_taproot` for the same rows.
  This is believed to be a feature-engineering artifact upstream of this
  release; the boolean column is dropped in preprocessing and script-type
  strings are used directly for address parsing instead (see `spectra/graph.py`).
- `input_address_count`, `output_address_count` are binary (0/1) indicators,
  not raw counts.
- `is_self_transfer`, `has_p2pk`, `has_p2pkh`, `has_p2sh`, `has_p2wpkh`,
  `has_p2wsh`, `address_reuse`, `output_script_count` are constant
  (always 0/`False`) across the full corpus and are dropped before
  modeling (see `configs/config.yaml` → `features.always_zero_cols`).
- The spectral graph module (Module S) is built on a subsampled graph for
  computational tractability; the full transaction table is used for all
  tabular ML/DL models. See `configs/config.yaml` for the exact sampling
  parameters (`sample_n`, `graph_sample_n`, `max_graph_nodes`).
