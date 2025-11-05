This repository is for the article _On-Demand Mobility Services and Policy Impacts: A Case Study from Chengdu, China_, which is currently under review by Transporation Research Part A.
The review stage password is the** manuscript number** and we will make all codes open after being accepted.

**Data:**

_/data/2015-04-01-FULL.csv:_ Example cruising taxi operations data from Chengdu, in 1/4/2015. All vehicle numbers are anonymized.

_/data/Chengdu-link.csv:_ Road network data of Chengdu, China.

**Data Processing:**

_match_trip_to_node.py:_ Match trips to the nodes of road network.

_gradient_descent_speed_estimation.py:_ Estimate speeds of each road segment dynamically, using historical taxi operations data.

_MFD_Chengdu.py:_ Fit and estimate the linear impact of fleet size on road speed, for Section 5.1.

_demand_management_spatial.py:_ Construct dataset with spatial characteristics of trips, for Section 5.3.

_demand_management_temporal.py:_ Construct dataset with temporal characteristics of trips, for Section 5.3.


**Simulation:**

_KM-batch-fleet.py:_ Main simulation framework with control of fleet size, for Section 4, 5.1 in the article.

_KM-batch-geofencing.py:_ Simulation with geofencing policy, for Section 5.2 in the article.

_KM-batch-demand.py:_ Simulation with different demand management strategies, for Section 5.3 in the article.
