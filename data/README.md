# Data sources and local file layout

Data files are not distributed in this public repository pending confirmation of the source providers' redistribution terms. Processed files will be provided to editors and reviewers upon request. The paths below document the local layout expected by the notebooks.

## Processed data used by the code

| Path | Purpose |
|---|---|
| `clean_dataframes/1960-2021/*.csv` | Interpolated county demographic and socioeconomic series |
| `clean_dataframes/urban_metrics_interpolated_and_cleaned_1960_2020.csv` | Combined county covariates used by notebook 06 |
| `clean_dataframes/crime_county_data.csv` | County homicide data through 2014 |
| `clean_dataframes/crime_county_data_extended.csv` | Extended county homicide data through 2020 |
| `clean_dataframes/full_merged_data_1960_2020.csv` | County-level merged analysis input |
| `log_df.csv` | Final log-transformed MSA analysis table: 344 MSAs and 1960–2020 |
| `log_df_centered.csv` | Centered version of the final analysis table |
| `cbsatocountycrosswalk.csv` | County-to-metropolitan-area crosswalk |
| `state_abbreviations.csv` | State-name lookup |
| `historical_trend/homicide_rate_cdc_wonder.tsv` | CDC WONDER national homicide series used in Figure 1 |
| `homicide_trend.dat` | Compiled historical US homicide series used in Figure 1 |
| `homicide-rates-across-western-europe.csv` | Cross-national homicide series used in Figure S1 |

The final analysis table used for the manuscript contains 20,914 unique MSA-year rows, 344 MSAs, and all 61 years from 1960 through 2020.

## Original source inventory

### FBI Uniform Crime Reporting data

- Provider: Federal Bureau of Investigation, Uniform Crime Reporting Program.
- Current access point: [FBI Crime Data Explorer](https://cde.ucr.cjis.gov/LATEST/webapp/).
- Historical study used for the 2015–2020 extension: *Uniform Crime Reporting Program Data: Offenses Known and Clearances by Arrest, United States, 2015–2020*, [ICPSR 37911, version 1](https://doi.org/10.3886/ICPSR37911.v1).
- Manuscript access dates: 1 January 2022 for the FBI source and 5 October 2022 for ICPSR 37911.
- Expected local inputs:

```text
data/raw_crime_data/<year>.dat
data/clean_crime_data/<year>.csv
data/agencies_county_map/*.dat
data/agencies_county_map/ori_to_fips.csv
data/laucnty16.csv
data/clean_dataframes/crime.csv
```

The `<year>.dat` and `<year>.csv` names represent the annual files consumed by notebooks 01–03. ICPSR downloads may require accepting the provider's terms or signing into an ICPSR account. Do not rename provider files until their correspondence with the notebook year is recorded.

### IPUMS NHGIS demographic and socioeconomic data

- Provider: IPUMS National Historical Geographic Information System.
- Access: [NHGIS Data Finder](https://www.nhgis.org/data) and [NHGIS documentation](https://www.nhgis.org/data-documentation).
- Manuscript access date: 1 January 2022.
- Access condition: a free IPUMS account and a custom extract request may be required.
- Expected extracts and paths:

```text
data/ipumns_raw_data/population_1960/*.csv
data/ipumns_raw_data/population_1970-2010/nhgis0003_ts_nominal_county.csv
data/ipumns_raw_data/race_1960/*.csv
data/ipumns_raw_data/poverty/*.csv
data/ipumns_raw_data/education/nhgis0014_ts_nominal_county.csv
data/ipumns_raw_data/labor_force_and_employment/nhgis0015_ts_nominal_county.csv
```

`ipumns_raw_data` is the historical local directory spelling used by the notebooks; it refers to IPUMS NHGIS. When recreating the extracts, select county geography and preserve the NHGIS codebook delivered with each extract. Extract numbers such as `nhgis0003`, `nhgis0014`, and `nhgis0015` are account-specific historical filenames, not globally reproducible dataset identifiers. Record the selected tables, years, geographic level, extract creation date, and downloaded codebook.

### US Census Bureau county population and poverty estimates

- Provider: US Census Bureau.
- General access: [Census Bureau data portal](https://data.census.gov/) and [Population Estimates](https://www.census.gov/programs-surveys/popest.html).
- Manuscript access date: 1 January 2022.
- Expected local inputs:

```text
data/county_population_1970_1989_us_census/*.xls
data/census_poverty_data_1989_2019/<year>.xls
data/census_poverty_data_1989_2019/<year>.dat
data/census_poverty_data_1989_2019/<year>.csv
data/Poverty-Rates-by-County-1960-2010.xlsm
```

The workbook and converted annual files should be preserved together. Record the original Census table or download URL for every year before public archival deposit.

### CDC WONDER bridged-race population estimates

- Provider: Centers for Disease Control and Prevention, CDC WONDER.
- Dataset cited in the manuscript: [Bridged-Race Population Estimates 1990–2020](https://wonder.cdc.gov/Bridged-Race-v2020.HTML).
- Manuscript access date: 1 January 2022.
- Expected local inputs:

```text
data/cdc_county_population_1990_2019/*.txt
```

Save the WONDER query criteria or exported request description with the downloaded text files so another researcher can reconstruct the query.

### Bureau of Labor Statistics unemployment estimates

- Provider: US Bureau of Labor Statistics, Local Area Unemployment Statistics.
- Access: [LAUS tables and county annual averages](https://www.bls.gov/lau/tables.htm) and [LAUS data tools](https://www.bls.gov/lau/data.htm).
- Manuscript access date: 1 January 2022.
- Expected local inputs:

```text
data/annual_unemployment_1990-2020/laucnty<yy>.txt
data/annual_unemployment_1990-2020/laucnty<yy>.csv
```

The notebook uses annual county files for 1990–2020. Preserve the original BLS text or workbook downloads before producing the local CSV conversions.

## Recommended local archive record

For every downloaded provider file, add one row to a local manifest with:

```text
provider,dataset,source_url,version_or_year,access_date,local_path,sha256,redistribution_note
```

[`raw_sources_manifest_template.csv`](raw_sources_manifest_template.csv) provides a prefilled starting point. Expand each placeholder row to one row per actual downloaded file.

Generate a checksum with:

```bash
shasum -a 256 path/to/file
```

Do not commit account credentials, download cookies, API tokens, agency contact-address fields, or files whose provider terms prohibit redistribution. If redistribution is permitted, deposit large raw files and their manifest in Zenodo, Dryad, OSF, or another durable data repository rather than GitHub.

## Reproduction boundary

- Notebooks 05–07 and S1 can run after the processed data are placed in the paths listed above.
- Notebooks 01–04 document the raw-to-processed construction but require the provider files listed above.
- Editors and reviewers will be given access to processed files on request and to preserved raw inputs where legally permitted.

