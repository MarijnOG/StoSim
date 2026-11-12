# ASSIGNMENT 2

import pandas as pd
import numpy as np
import random 

# Objective 

# Model a security screening lane during the peak month (September) at the airport 

# 1. Validate discrete event simulator against theoretical predictions 
# 2. Evaluate two proposed interventions using hypothesis testing
# 3. Make a simulation-driven recommendation to the leadership team

# Calculate the arrival rate in passengers per minute
df = pd.read_csv('airport.csv')

print(df.columns)

values = df["Total Passengers"]


print(values.mean())

print(selected_values.mean())

# PART 1: Calculate the arrival rate in passengers per minute

mask = ((df['Year'].isin(range(2014, 2020))) & (df['Month'] == 'September'))
selected_values = df.loc[mask, 'Total Passengers']

# assuming 30 day months, 16 hour operational days, calculating the hourly arrival rate in passengers per minute
arrival_rate = selected_values.mean() / (30 * 16 * 60)
print(f"Arrival rate in passengers per minute: {arrival_rate:.4f}")

# System parameters for all simulations
# service time distribution: normal with mean 1 minutes
# standard deviation 0.25 minutes per passenger 
# if the service time sample is negative, resample until a positive value is obtained
# number of lanes: 1 (unless otherwise specified)

# PART 2: Hypothesis Testing & Simulation 
# I need to build a discrete event simulator
# Accept: arrival rate, service time parameters, number of servers, number of passengers to simulate
# Simulate: a queueing system until steady state
# Return: individual waiting times for each passenger (discard first 1000 passengers)

def simulate_q(arrival_rate, service_time_mean, service_time_std, num_servers, num_passengers):
    # initialize variables
    waiting_times = []
    queue = []
    current_time = 0
    next_arrival_time = current_time + np.random.exponential(1 / arrival_rate)
    service_times = np.random.normal(service_time_mean, service_time_std, num_passengers)

    # define events 
    # arrival: a passenger enters the system 
    # record the passenger's arrival time 
    # schedule the next arrival, unless the maximum arrivals has been reached
    # if a server is available, start service immediately and schedule a departure event
    # otherwise, place the passenger at the end of the queue 

    for i in range(num_passengers):
        # arrival
        if current_time >= next_arrival_time:
            # passenger arrives
            arrival_time = next_arrival_time
            if len(queue) < num_servers:
                # server available, start service immediately
                service_time = max(0, np.random.normal(service_time_mean, service_time_std))
                departure_time = arrival_time + service_time
                waiting_times.append(0)  # no waiting time
                current_time = departure_time
            else:
                # all servers busy, add to queue
                queue.append(arrival_time)
            # schedule next arrival
            next_arrival_time += np.random.exponential(1 / arrival_rate)
        else:
            # process departures if any servers are busy
            if queue:
                # a passenger is waiting in the queue
                waiting_passenger_arrival = queue.pop(0)
                service_time = max(0, np.random.normal(service_time_mean, service_time_std))
                departure_time = current_time + service_time
                waiting_times.append(current_time - waiting_passenger_arrival)  # calculate waiting time
                current_time = departure_time


    # departure: a passenger finishes service and leaves the system

    


