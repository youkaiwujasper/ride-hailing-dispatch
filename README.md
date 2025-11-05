This repository is for the article _On-Demand Mobility Services and Policy Impacts: A Case Study from Chengdu, China_, which is currently under review by Transporation Research Part A.
The review stage password is the manuscript number and we will make all codes open after being accepted.

**Data:**

**/data/2015-04-01-FULL.csv:** Example cruising taxi operations data from Chengdu, in 1/4/2015. All vehicle numbers are anonymized.
**/data/Chengdu-link.csv:** Road network data of Chengdu, China.

**Data Processing:**

**match_trip_to_node.py:** Match trips to the nodes of road network.
**gradient_descent_speed_estimation.py:** Estimate speeds of each road segment dynamically, using historical taxi operations data.
**MFD_Chengdu.py:** Fit and estimate the linear impact of fleet size on road speed, for Section 5.1.
**demand_management_spatial.py:** Construct dataset with spatial characteristics of trips, for Section 5.3.
**demand_management_temporal.py:** Construct dataset with temporal characteristics of trips, for Section 5.3.

**Simulation:**

**KM-batch-fleet.py:** Main simulation framework with control of fleet size, for Section 4, 5.1 in the article.
**KM-batch-geofencing.py:** Simulation with geofencing policy, for Section 5.2 in the article.
**KM-batch-demand.py:** Simulation with different demand management strategies, for Section 5.3 in the article.
