import numpy as np

class Torus:
    def __init__(self, x, y, z, R, r):
        self.x = x
        self.y = y
        self.z = z
        self.R = R
        self.r = r

    def is_inside(self, point):
        x, y, z = point
        return (np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) - self.R) ** 2 + (z - self.z) ** 2 <= self.r ** 2

class Sphere:
    def __init__(self, x, y, z, k):
        self.x = x
        self.y = y
        self.z = z
        self.r = k

    def is_inside(self, point):
        x, y, z = point
        return (x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2 <= self.r ** 2

class IntersectShape:
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def is_inside(self, point):
        return self.shape1.is_inside(point) and self.shape2.is_inside(point)

class MonteCarloIntegrator:
    def __init__(
        self,
        shape,
        bounds,
        num_samples,
        rng=None,
        coordinate_generators=None,
    ):

        if coordinate_generators is not None:
            if len(coordinate_generators) != 3:
                raise ValueError("coordinate_generators must contain three generators")
            self.coordinate_generators = tuple(coordinate_generators)
        else:
            self.coordinate_generators = None

        self.shape = shape
        self.bounds = bounds
        self.num_samples = num_samples
        self.rng = rng if rng is not None else np.random.uniform

    def integrate(self):
        count_inside = 0
        lower = np.asarray(self.bounds[0], dtype=float)
        upper = np.asarray(self.bounds[1], dtype=float)
        span = upper - lower

        for _ in range(self.num_samples):
            if self.coordinate_generators is None:
                point = self.rng(lower, upper, size=3)
            else:
                unit_point = np.array(
                    [next(generator) for generator in self.coordinate_generators],
                    dtype=float,
                )
                point = lower + unit_point * span
            if self.shape.is_inside(point):
                count_inside += 1

        volume_bounds = np.prod(span)
        return (count_inside / self.num_samples) * volume_bounds

def logistic_map_generator(x, m):
    while True:
        x = m * x * (1 - x)
        yield x

if __name__ == "__main__":
    # Task 1: Shapes at origin, evaluate intersect for: 
    # Case a: k=1, R = 0.75 and r = 0.4
    # Case b: k= 1, R = 0.5 and r = 0.5

    np.random.seed(42)  # For reproducibility

    sphere1 = Sphere(0, 0, 0, 1)
    torus1 = Torus(0, 0, 0, 0.75, 0.4)
    bounds1 = [(-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)]
    intersect1 = IntersectShape(sphere1, torus1)
    integrator1 = MonteCarloIntegrator(intersect1, bounds1, 100000)
    volume1 = integrator1.integrate()
    print(f"Estimated volume of the intersect using numpy.random.uniform(Case a): {volume1}")

    sphere2 = Sphere(0, 0, 0, 1)
    torus2 = Torus(0, 0, 0, 0.5, 0.5)
    bounds2 = [(-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)]
    intersect2 = IntersectShape(sphere2, torus2)
    integrator2 = MonteCarloIntegrator(intersect2, bounds2, 100000)
    volume2 = integrator2.integrate()
    print(f"Estimated volume of the intersect using numpy.random.uniform(Case b): {volume2}")

    # Task 2: Use logistic map generator for coordinate generation
    x_gen = logistic_map_generator(0.9, 3.8)
    y_gen = logistic_map_generator(0.8, 3.8)
    z_gen = logistic_map_generator(0.7, 3.8)

    int1_logistic = MonteCarloIntegrator(
        intersect1,
        bounds1,
        100_000,
        coordinate_generators=(x_gen, y_gen, z_gen),
    )
    volume1_logistic = int1_logistic.integrate()
    print(f"Estimated volume of the sphere using logistic map generator (Case a): {volume1_logistic}")

    int2_logistic = MonteCarloIntegrator(
        intersect2,
        bounds2,
        100_000,
        coordinate_generators=(x_gen, y_gen, z_gen),
    )
    
    volume2_logistic = int2_logistic.integrate()
    print(f"Estimated volume of the sphere using logistic map generator (Case b): {volume2_logistic}")


    # Task 3: Estimate volume of a sphere and an off-center torus intersection
    # a: using numpy.random.uniform
    sphere3 = Sphere(0, 0, 0, 1)
    torus3 = Torus(0, 0, 0.1, 0.75, 0.4)
    bounds3 = [(-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)]
    intersect3 = IntersectShape(sphere3, torus3)
    integrator3 = MonteCarloIntegrator(intersect3, bounds3, 100000)
    volume3 = integrator3.integrate()
    print(f"Estimated volume of the intersect using numpy.random.uniform (off-center torus): {volume3}")

    # b: using importance sampling around (0, 0, 0.1)
    bounds3_1 = [(-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)]
    bounds3_2 = [(-0.5, -0.5, -0.55), (0.5, 0.5, 0.65)]
    integrator3_2 = MonteCarloIntegrator(intersect3, bounds3_2, 100000)
    volume3_2 = integrator3_2.integrate()
    print(f"Estimated volume of the intersect using importance sampling (off-center torus): {volume3_2}")