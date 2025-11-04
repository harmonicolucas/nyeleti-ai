# Data sources

All fire data are NASA products obtained from the Fire Information for
Resource Management System (FIRMS). Nothing is simulated or synthesised.

## Active fire detections

| Key | Product | Sensor / platform | Resolution | Years used | Records (raw) |
|---|---|---|---|---|---|
| `modis` | MCD14ML, Collection 6.1, standard processing | MODIS on Terra and Aqua | 1 km | 2001-2025 | 3,421,484 |
| `viirs_snpp` | VNP14IMGTML, standard processing | VIIRS on Suomi NPP | 375 m | 2012-2025 | 11,258,891 |
| `viirs_noaa20` | VJ114IMGTML, standard processing | VIIRS on NOAA-20 | 375 m | 2020-2025 | 4,733,923 |

Source: FIRMS country yearly summaries,
<https://firms.modaps.eosdis.nasa.gov/country/>. File pattern:

    https://firms.modaps.eosdis.nasa.gov/data/country/<sensor>/<year>/<sensor>_<year>_Mozambique.csv

with `<sensor>` one of `modis`, `viirs-snpp`, `viirs-jpss1`. These files hold
the standard-processing (science quality) record and need no access key.

The role of each sensor:

- **MODIS** is the input of the harmonization and the only source before 2012.
- **VIIRS S-NPP** is the reference scale and the training target.
- **VIIRS NOAA-20** is not used for training. It is shown on the map and used
  in the evaluation as an independent second VIIRS.

### Fields used

`latitude`, `longitude`, `acq_date`, `acq_time` (UTC), `satellite`,
`instrument`, `confidence`, `frp` (MW), `brightness` and `bright_t31` (MODIS)
or `bright_ti4` and `bright_ti5` (VIIRS), `scan`, `track`, `daynight`,
`version`, `type`.

`confidence` is a percentage for MODIS and a class (low, nominal, high) for
VIIRS; both are kept, and the cleaned files add a common three-level class.
`type` separates presumed vegetation fires (0) from active volcanoes (1), other
static land sources (2) and offshore detections (3).

### Period

- MODIS: 1 January 2001 to 30 December 2025. 2001 and the first half of 2002
  are Terra only.
- VIIRS S-NPP: 20 January 2012 to 31 December 2025.
- VIIRS NOAA-20: 1 January 2020 to 31 December 2025 (first year in the
  country archive).

The year 2026 was not in the archive when the data were downloaded
(15 November 2026). Recent months exist only as near-real-time data, which
have a different processing level and require a FIRMS MAP_KEY; they are not
mixed into this record. `NYELETI_LAST_YEAR` extends the download when a new
yearly file is published.

### Provenance

`ml/data/raw/manifest.json` lists, for every file: URL, product, size,
SHA-256, number of records, first and last acquisition date, satellites and
versions found, and the download time. `docs/data_quality.md` reproduces the
checksums and the result of the validation, file by file.

## Administrative boundaries

geoBoundaries, gbOpen release, simplified geometries, fetched through the
geoBoundaries API: <https://www.geoboundaries.org/api/current/gbOpen/MOZ/>.

| Level | Units | Original source | Licence | Year represented |
|---|---|---|---|---|
| ADM0 (country) | 1 | OpenStreetMap, Wambacher | ODbL 1.0 | 2017 |
| ADM1 (provinces) | 10 | OpenStreetMap, Wambacher | ODbL 1.0 | 2017 |
| ADM2 (districts) | 159 | Mozambique INE, OCHA ROSEA | CC BY 3.0 IGO | 2019 |

The exact files and download times are in `ml/data/boundaries/sources.json`.
The ADM1 layer has 10 units; Maputo city is not a separate unit in it. One
small ADM2 unit (Ilha Risunodo) contains no grid cell centre at 5 km and has
no series of its own. Boundaries are used to aggregate and to display results and carry no
official endorsement.

## Base maps

- NASA GIBS, Blue Marble Shaded Relief and Bathymetry, and MODIS Terra
  Corrected Reflectance (true colour) for the selected day:
  <https://gibs.earthdata.nasa.gov>.
- OpenStreetMap standard tiles, © OpenStreetMap contributors.

Base maps are fetched by the browser at display time; no imagery is stored.

## Things to keep in mind when reading the data

- An active fire detection is a pixel flagged as containing fire at the time
  of the overpass. It is not a fire event and not a burned area. Counts depend
  on pixel size, overpass time, cloud cover and viewing angle.
- Both instruments miss fires under cloud or smoke and fires that burn between
  overpasses.
- Days with no detection can be true absence of fire, cloud, or missing data.
  The archive does not say which. The pipeline handles them explicitly (see
  `methodology.md`, section 3) and the quality report counts them.
- MODIS on Aqua has no detection in the archive from 16 August to 1 September
  2020, during the fire season.

## Citations

- MODIS Collection 6.1 active fire product: Giglio, L., Schroeder, W., Justice,
  C. O. (2016). The collection 6 MODIS active fire detection algorithm and
  fire products. *Remote Sensing of Environment*, 178, 31-41.
- VIIRS 375 m active fire product: Schroeder, W., Oliva, P., Giglio, L.,
  Csiszar, I. A. (2014). The New VIIRS 375 m active fire detection data
  product: Algorithm description and initial assessment. *Remote Sensing of
  Environment*, 143, 85-96.
- NASA FIRMS: "We acknowledge the use of data from NASA's Fire Information for
  Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part
  of NASA's Earth Science Data and Information System (ESDIS)."
- geoBoundaries: Runfola, D. et al. (2020). geoBoundaries: A global database
  of political administrative boundaries. *PLoS ONE*, 15(4), e0231866.
- NASA Global Imagery Browse Services (GIBS), part of NASA's Earth Science
  Data and Information System (ESDIS).
