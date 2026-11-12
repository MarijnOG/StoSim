# Model passenger arrivals according to airport data

import numpy as np
import csv
from collections.abc import Callable
import matplotlib.pyplot as plt
        
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
        self.data: list[dict[str, float | int | str | None]] = []

    def run(self, max_passengers: int, store_data: bool = False) -> None:
        if store_data:
            self.data.clear()
            self.data.append(self._record("initial"))

        while self.num_served < max_passengers:
            if self.next_arrival < self.next_departure:
                self.time = self.next_arrival
                self.queue.append(self.time)
                self.next_arrival = self.time + np.random.exponential(1 / self.lab)
                if len(self.queue) == 1:
                    self.next_departure = self.time + self.service_time()
                if store_data:
                    self.data.append(self._record("arrival"))
            else:
                self.time = self.next_departure
                arrival_time = self.queue.pop(0)
                wait_time = self.time - arrival_time
                self.total_wait_time += wait_time
                self.num_served += 1
                if len(self.queue) > 0:
                    self.next_departure = self.time + self.service_time()
                else:
                    self.next_departure = float("inf")
                if store_data:
                    self.data.append(self._record("departure", wait_time))
        self.queue.clear()

    def _record(self, event: str, wait_time: float | None = None) -> dict[str, float | int | str | None]:
        return {
            "time": self.time,
            "event": event,
            "queue_length": len(self.queue),
            "num_served": self.num_served,
            "wait_time": wait_time,
        }

    def utilization(self) -> float:
        """Estimate the utilization of the system based on the average service time and arrival rate.
        Explicitly does NOT use the actual service times from the simulation, but rather the expected 
        service time based on the distribution."""
        return float(self.lab * np.mean([self.service_time() for _ in range(1000)]))

def service_time(mean: float = 1.0, std: float = 0.25) -> float:
    """Return a service time based on a normal distribution, redraw negative values."""
    while True:
        time = np.random.normal(mean, std)
        if time > 0:
            return time

if __name__ == "__main__":
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

    # Part 2: Simulation
    # stable M/G/1 configuration
    stable_des = Des(0.85, service_time) # test for rho = labda = 0.85
    stable_des.run(3000, store_data=True)
    print(f"Average wait time: {stable_des.total_wait_time / stable_des.num_served}")
    print(f"Total passengers served: {stable_des.num_served}")
    print(f"Utilization estimate: {stable_des.utilization()}")
    print(f"First 10 events: {stable_des.data[:10]}")

