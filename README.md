# neon_aniso
A collection of scripts and tools designed to evaluate NEON turbulence with a specific eye towards turbulent anisotropy and surface layer scaling.
Intended for the use of Dr. Waterman, collaborators, and other interested parties
Please reach out at tyswater{at}gmail.com if you have any questions.

## Repository Scripts and Structure:
#### download_prep/ 
download scripts for various pieces of NEON data; tweaks likely needed
#### turb_tw/
Scripts to calculate information directly from turbulence timeseries
#### testing/
Testing and development scripts
#### templates/
Template scripts for various data processing workflows


## Variable List
A list of variables available for L1 processing and description 
- TIME: time in seconds since 1970 UTC
- q[VARIABLE]: quality flag for variable from NEON. 0 is good.
- qs[VARIABLE]: science review quality flag (not always available)
- qUVW: compilation of quality flags for U,V and W
- qprofile\_[X]\_upper: quality flag for only top 3 pointsin profile
- U: horizontal velocity (m/s)
- V: horizontal velocity (m/s)
- W: vertical velocity (m/s)
- Us: streamline velocity (m/s)
- Vs: spanwise velocity (m/s)
- UU/UsUs: velocity variance (if Us, is streamwise)
- VV/VsVs: spanwise velocity variance
- WW: vertical velocity variance
- THETA: virtual temperature
- THETATHETA: sonic temperature (virtual temperature) variance
- Q: water vapor molar ratio with dry air
- QQ: water vapor variance
- C: carbon dioxide molar ratio with dry air
- CC: carbon dioxide variance
- T: air temperature
- TT: temperature variance
- UV, UW, VW: velocity covariances
- QT, TC, THETAC, THETAQ, QC: scalar covariances
- DMOLAIRDRY: molar density of dry air
- PA: Pressure
- P: precipitation in mm/minute
- ST\_ZH\_23_[XX]: stationarity metric for variance [XX]
- ANI\_YB: anisotropy invariant metric yb
- ANI\_XB: anisotropy invariant metric xb
- ANID\_YB: anisotropy invariant metric yb if only energetics (variances) used
- ANID\_XB: anisotropy invariant metric xb if only energetics (variances) used
- RH: relative humidity
- TD: dewpoint temperature
- RHO: air density
- USTAR: friction velocity
- H: sensible heat flux (W/m2)
- LE: latent heat flux (W/m2)
- G: ground heat flux (W/m2)
- G\_full: ground heat flux; 3 sensor average even if qaqc flag 1 from sensor
- L\_MOST: monin-obukhov stability parameter
- canopy height: canopy height as defined by the NEON network personnel
- elevation: m
- lat: latitude
- lon: longitude
- lvls: vertical sensor levels for profiles of scalars (Q,C,T)
- lvls\_u: vertical sensor levels for profiles of velocity
- nlcd\_dom: dominant nlcd landcover type within a 2km box centered on the tower
- tow\_height: tower height (3D sonic height)
- utc\_off: utc offset 
- zd: zero plane displacement height
- nlcd[XX]: percent coverage of the [XX] nlcd landcover type within a 2km box
- profile_[X]: profile of quantity [X]
    - [X]\_[N]: timeseries of [X] at index N. the value of N corresponds to an
                index in lvls or lvls\_u for the actual height of measurement
- SW\_IN: shortwave radiation downwards [W/m2]
- SW\_OUT: shortwave radiation upwards [W/m2]
- LW\_IN: longwave radiation downwards [W/m2]
- LW\_OUT: longwave radiation upwards [W/m2]
- NETRAD: net incoming radiation at top of tower
- GCC90: 90th percentile green chromatic coordinate from phenocam image
- GCC90\_E: GCC90 for evergreen plants only
- GCC90\_D: GCC90 for non-evergreen plants only
- GCC90\_C: compilation (average across plant types) GCC90
- NDVI90: NDVI vegetation index from phenocam
- SOLAR\_ALTITUDE: solar altitude angle
- GROWING: 0 not growing season, 1 growing season deciduous
           2 growing season evergreen, 3 growing season all vegetation

