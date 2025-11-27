Dataset Name: operational_metrics.csv
Number of Rows: ~695,000
Number of Columns: 25

Description:
This dataset contains time-series operational data recorded from an industrial energy production plant 
(possibly geothermal). Measurements are taken every few minutes and track injection/production flows, 
temperatures, pressures, and energy metrics. The goal is to monitor performance, identify trends, 
and link operational behavior with seismic events.

Key columns:

1. recorded_at (datetime)
   → Timestamp of the recorded operational data.

2. phase (float)
   → Numeric code identifying the plant’s operational phase (e.g., startup, production, shutdown).

3. inj_flow (float)
   → Injection flow rate — amount of fluid pumped into the reservoir (m³/h).

4. inj_whp (float)
   → Injection wellhead pressure — pressure at the injection wellhead (bar or MPa).

5. inj_temp (float)
   → Temperature of injected fluid (°C).

6. inj_ap (float)
   → Injection annulus pressure — secondary pressure in the injection casing (bar).

7. prod_temp (float)
   → Production wellhead temperature (°C).

8. prod_whp (float)
   → Production wellhead pressure (bar).

9. gt03_whp (float)
   → Another wellhead pressure sensor reading (possibly backup sensor or specific well).

10. hedh_thpwr (float)
    → Thermal power output (heat energy produced) in MW or MJ.

11. basin_flow (float)
    → Flow measurement from the basin/reservoir system (m³/h).

12. prod_flow (float)
    → Flow rate of produced (extracted) fluid (m³/h).

13. source (string)
    → Origin or source of the measurement (e.g., sensor group or system name).

14. is_producing (boolean)
    → Indicates if the plant was producing energy at the time of measurement (True = producing).

15. phase_started_at (datetime)
    → Timestamp when the current operational phase started.

16. phase_production_ended_at (datetime)
    → Timestamp when production in this phase ended.

17. phase_ended_at (datetime)
    → Timestamp when the entire operational phase ended.

18. volume (float)
    → Instantaneous processed fluid volume (m³).

19. cum_volume (float)
    → Cumulative total processed volume since the start of the phase (m³).

20. inj_energy (float)
    → Energy injected into the system (MJ or MWh).

21. cum_inj_energy (float)
    → Cumulative injected energy over time.

22. cooling_energy (float)
    → Cooling energy used at this timestamp.

23. cum_cooling_energy (float)
    → Total cumulative cooling energy.

24. heat_exch_energy (float)
    → Energy transferred through the heat exchanger at this time.

25. cum_heat_exch_energy (float)
    → Total cumulative energy exchanged through the heat exchanger.

Purpose:
- To analyze the plant’s performance and correlate operational conditions with seismic activity.
- To understand how variables like injection flow, pressure, and temperature relate to energy output and stability.
- Common tasks include data cleaning, resampling (e.g., to hourly averages), time-based merging with seismic event data, and correlation analysis.

Data Characteristics:
- Time-series data with frequent sampling (every few minutes).
- Some missing values (NaN) in energy and flow-related columns.
- Combination of continuous, boolean, and timestamp variables.
