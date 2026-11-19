# Model passenger arrivals according to airport data

import numpy as np
import csv
from collections.abc import Callable
from typing import TypedDict
import matplotlib.pyplot as plt
from scipy import stats

        
class EventRecord(TypedDict):
    time: float
    event: str
    queue_length: int
    num_served: int
    wait_time: float | None


class Des:
    def __init__(self, lab: float, service_time: Callable[[], float]) -> None:
        self.lab = lab
        self.service_time = service_time
        self.queue = []
        self.time = 0.0
        self.next_arrival = np.random.exponential(1 / lab)
        self.next_departure = float("inf")
        self.num_served = 0
        self.total_wait_time = 0.0
        self.data: list[EventRecord] = []

    def run(
        self,
        max_passengers: int,
        num_servers: int = 1,
        store_data: bool = False,
        num_lanes: int = 1,
    ) -> None:
        """Run with ``num_servers`` servers in each of ``num_lanes`` lanes.

        ``lab`` is the arrival rate for the whole simulation. Arrivals join
        the lane with the fewest waiting or currently served passengers.
        """
        if max_passengers <= 0:
            return
        if num_servers <= 0 or num_lanes <= 0:
            raise ValueError("num_servers and num_lanes must be positive")

        queues: list[list[float]] = [[] for _ in range(num_lanes)]
        departures: list[list[float]] = [
            [float("inf")] * num_servers for _ in range(num_lanes)
        ]
        in_service: list[list[float | None]] = [
            [None] * num_servers for _ in range(num_lanes)
        ]

        if store_data:
            self.data.clear()
            self.data.append(self._record("initial"))

        while self.num_served < max_passengers:
            next_departure = min(
                departure for lane in departures for departure in lane
            )
            if self.next_arrival < next_departure:
                self.time = self.next_arrival
                lane = min(
                    range(num_lanes),
                    key=lambda index: len(queues[index])
                    + sum(
                        departure != float("inf")
                        for departure in departures[index]
                    ),
                )
                queues[lane].append(self.time)
                self.next_arrival = self.time + np.random.exponential(1 / self.lab)
                self._start_services(queues, departures, in_service)
                if store_data:
                    self.data.append(self._record("arrival"))
            else:
                self.time = next_departure
                lane, server = next(
                    (lane, server)
                    for lane, lane_departures in enumerate(departures)
                    for server, departure in enumerate(lane_departures)
                    if departure == next_departure
                )
                arrival_time = in_service[lane][server]
                if arrival_time is None:
                    raise RuntimeError("A departure occurred without an active passenger")
                wait_time = self.time - arrival_time
                self.total_wait_time += wait_time
                self.num_served += 1
                departures[lane][server] = float("inf")
                in_service[lane][server] = None
                self._start_services(queues, departures, in_service)
                if store_data:
                    self.data.append(self._record("departure", wait_time))

        self.queue.clear()
        self.next_departure = float("inf")

    def _start_services(
        self,
        queues: list[list[float]],
        departures: list[list[float]],
        in_service: list[list[float | None]],
    ) -> None:
        for lane, queue in enumerate(queues):
            for server, departure in enumerate(departures[lane]):
                if queue and departure == float("inf"):
                    in_service[lane][server] = queue.pop(0)
                    departures[lane][server] = self.time + self.service_time()

        self.queue = [
            arrival
            for lane, queue in enumerate(queues)
            for arrival in queue + [
                active for active in in_service[lane] if active is not None
            ]
        ]

    def _record(self, event: str, wait_time: float | None = None) -> EventRecord:
        return {
            "time": self.time,
            "event": event,
            "queue_length": len(self.queue),
            "num_served": self.num_served,
            "wait_time": wait_time,
        }

    def utilization(self, num_servers: int = 1, num_lanes: int = 1) -> float:
        """Estimate the utilization of the system based on the average service time and arrival rate.
        Explicitly does NOT use the actual service times from the simulation, but rather the expected 
        service time based on the distribution."""
        if num_servers <= 0 or num_lanes <= 0:
            raise ValueError("num_servers and num_lanes must be positive")
        total_servers = num_servers * num_lanes
        return float(
            self.lab
            * np.mean([self.service_time() for _ in range(1000)])
            / total_servers
        )

def service_time(mean: float = 1.0, std: float = 0.25) -> float:
    """Return a service time based on a normal distribution, redraw negative values."""
    while True:
        time = np.random.normal(mean, std)
        if time > 0:
            return time

def theoretical_wait_time(lab: float, mean_service_time: float, std_service_time: float) -> float:
    """Calculate the theoretical average wait time in an M/G/1 queue using the Pollaczek-Khinchine formula."""
    rho = lab * mean_service_time
    expected_service_time_squared = mean_service_time**2 + std_service_time**2
    return (lab * expected_service_time_squared) / (2 * (1 - rho))

if __name__ == "__main__":
    np.random.seed( 42)  # For reproducibility

    #==================================================================================
    # Part 1: Data
    # Calculate the average monthly passenger count across these Septembers
    file = "airport.csv"
    with open(file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        passenger_counts = [int(row[-1]) for row in reader if row[1] == "September" and int(row[0]) >= 2015 and int(row[0]) <= 2019]

    # Convert to hourly rate labda (30-day month, 16 operational hours)
    lab_hour_all_lanes = np.mean(passenger_counts) / 30 / 16 # 30 days, 16 hours
    lab = float(lab_hour_all_lanes / 50 / 60) # 50 lanes, 60 minutes
    print(f"labda = {lab_hour_all_lanes} with {np.mean(passenger_counts)} average passengers in September")
    print(f"Giving {lab} passengers per minute per lane")

    #==================================================================================
    # Part 2A: rho = 0.85 baseline

    lab_2a = 0.85 * (1 / 1.0) # 0.85 * (1 / mean_service_time)
    mean_service_time = 1.0
    std_service_time = 0.25
    theor_wait_time = theoretical_wait_time(lab_2a, mean_service_time, std_service_time)
    print(f"Theoretical average wait time: {theor_wait_time} minutes")

    # stable M/G/1 configuration (ran 40 times, rho = 0.8)
    wait_times = []
    for _ in range(40):
        des = Des(0.8, lambda: service_time(1.0, 0.25))
        des.run(max_passengers=1000)
        wait_times.append(des.total_wait_time / des.num_served)

    print(f"Average wait time over 40 runs: {np.mean(wait_times)} minutes")
    print(f"Standard deviation: {np.std(wait_times)} minutes")

    # Hypothesis test with alpha = 0.05
    t_stat, p_value = stats.ttest_1samp(wait_times, theor_wait_time)
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.2f}: ", end="")
    if p_value < 0.05:
        print("Reject the null hypothesis")
    else:
        print("Fail to reject the null hypothesis")

    #==================================================================================
    # Part 2B: lab from data, service time from Part 2A
    des_2b_base = Des(lab, lambda: service_time(1.0, 0.25))
    des_2b_a = Des(lab, lambda: service_time(1.0, 0.25))
    des_2b_b = Des(lab, lambda: service_time(1.0, 0.1))

    des_2b_base.run(max_passengers=1000, store_data=True)
    des_2b_a.run(max_passengers=1000, num_servers=2, store_data=True)
    des_2b_b.run(max_passengers=1000, store_data=True)

    print(f"Average wait time for base case: {des_2b_base.total_wait_time / des_2b_base.num_served} minutes")
    print(f"Average wait time for 2 servers: {des_2b_a.total_wait_time / des_2b_a.num_served} minutes")
    print(f"Average wait time for 0.5 service time: {des_2b_b.total_wait_time / des_2b_b.num_served} minutes")
    print(f"Utilization for base case: {des_2b_base.utilization(num_servers=1)}")
    print(f"Utilization for 2 servers: {des_2b_a.utilization(num_servers=2)}")
    print(f"Utilization for 0.5 service time: {des_2b_b.utilization(num_servers=1)}")
    print("\n")

    # plot resulting queue build up
    plt.figure(figsize=(10, 6))
    plt.plot([record["time"] for record in des_2b_base.data], [record["queue_length"] for record in des_2b_base.data], label="Base Case")
    plt.plot([record["time"] for record in des_2b_a.data], [record["queue_length"] for record in des_2b_a.data], label="2 Servers")
    plt.plot([record["time"] for record in des_2b_b.data], [record["queue_length"] for record in des_2b_b.data], label="0.5 Service Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Queue Length")
    plt.title("Queue Build Up")
    plt.legend()
    plt.show()