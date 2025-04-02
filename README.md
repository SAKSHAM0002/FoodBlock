# FoodBlock: A Secure and Cost-Optimal Framework for Online Food Ordering

## Overview
This project simulates an online food ordering and delivery system, optimizing efficiency and fairness through auction mechanisms and intelligent rider selection. The simulation incorporates restaurant and rider behaviors, bidding strategies, and customer experiences.

## Functionality Overview
### Restaurant Generator
- Manages order arrivals and groups them based on time slots.
- Processes all orders from the same restaurant together.
- Ensures simulation timing aligns with real-world order placement patterns.

### Batch Auction Mechanism
- Selects a subset of riders for each batch of orders.
- Riders submit bids based on machine-predicted values, their rating, and order characteristics.
- The final rider selection is based on bid fairness, delivery feasibility, and auction constraints.

### Order Processing & Rider Actions
- Determines rider availability and assigns deliveries.
- Implements the nearest neighbor algorithm for optimal delivery sequencing.
- Tracks rider status, updating delivery progress and time spent on each order.

## Simulation Aspects
### Rider Behavior During Bidding
- Bids are influenced by rider ratings and order-specific factors.
- Riders should not artificially inflate bids to appear more competent.
- Riders are chosen based on an auction between the nearest competitors.

### Restaurant Owner Bidding Behavior
- Dependent on food order value and rider ratings.
- Ensures fair pricing for delivery charges.

### Customer Ratings
- Primarily influenced by wait times.
- Additional factors may include food freshness and rider politeness.

## Data Generation & Analysis
### Graphs to be Generated
- Average queue length over time.
- Distance traveled versus orders served.
- Total customer wait time versus orders served.
- Distribution of delivery distances.
- Distribution of customer wait times.

### Auction Mechanism Insights
- Change in rider ratings over time.
- Savings per order from the restaurant’s perspective, considering driving costs and rider tips.
- Earnings distribution among riders.
- Customer rating trends across different scenarios.

## Simulation Design Decisions
- Bidding depends on the rider’s total travel time and distance.
- Probabilistic approaches are used where applicable.
- Rider and customer ratings are updated dynamically.

## Key Considerations
- Base fare and per-kilometer charges influence bid values.
- The simulation operates within a maximum distance threshold.
- Auction mechanisms ensure fairness and efficiency in order allocation.

This documentation serves as a guide for understanding the simulation workflow and the various factors influencing rider and restaurant behaviors.

