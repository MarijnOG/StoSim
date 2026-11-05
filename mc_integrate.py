import numpy as np
import matplotlib.pyplot as plt  # ADDED: for convergence / diagnostic plots

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
        self.running_counts = []  # ADDED: number of hits after each sample (for convergence plot)

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
            self.running_counts.append(count_inside)  # ADDED

        volume_bounds = np.prod(span)
        self.volume_bounds = volume_bounds  # ADDED: keep box volume for the error formula
        self.p_hat = count_inside / self.num_samples  # ADDED: fraction of points inside
        return (count_inside / self.num_samples) * volume_bounds

    # ADDED: standard error SE = V_box * sqrt(p(1-p)/N)
    # (hit/miss is a Bernoulli trial -> hit count is Binomial(N, p))
    # NOTE: only valid if samples are independent and uniform -> NOT true in Task 2
    def standard_error(self):  # ADDED
        return self.volume_bounds * np.sqrt(self.p_hat * (1 - self.p_hat) / self.num_samples)  # ADDED

    # ADDED: estimate after 1, 2, ..., N samples
    def running_estimate(self):  # ADDED
        n = np.arange(1, self.num_samples + 1)  # ADDED
        return self.volume_bounds * np.array(self.running_counts) / n  # ADDED

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
    print(f"  box used: {bounds1}, box volume = {integrator1.volume_bounds}")  # ADDED
    print(f"  standard error = {integrator1.standard_error():.4f}, "
          f"95% CI = [{volume1 - 1.96 * integrator1.standard_error():.4f}, "
          f"{volume1 + 1.96 * integrator1.standard_error():.4f}]")  # ADDED

    sphere2 = Sphere(0, 0, 0, 1)
    torus2 = Torus(0, 0, 0, 0.5, 0.5)
    bounds2 = [(-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)]
    intersect2 = IntersectShape(sphere2, torus2)
    integrator2 = MonteCarloIntegrator(intersect2, bounds2, 100000)
    volume2 = integrator2.integrate()
    print(f"Estimated volume of the intersect using numpy.random.uniform(Case b): {volume2}")
    print(f"  box used: {bounds2}, box volume = {integrator2.volume_bounds}")  # ADDED
    print(f"  standard error = {integrator2.standard_error():.4f}, "
          f"95% CI = [{volume2 - 1.96 * integrator2.standard_error():.4f}, "
          f"{volume2 + 1.96 * integrator2.standard_error():.4f}]")  # ADDED

    # ADDED: convergence plot for Task 1 (estimate vs number of samples)
    n_axis = np.arange(1, 100000 + 1)  # ADDED
    plt.figure(figsize=(8, 4.5))  # ADDED
    plt.semilogx(n_axis, integrator1.running_estimate(), label="Case a")  # ADDED
    plt.semilogx(n_axis, integrator2.running_estimate(), label="Case b")  # ADDED
    plt.xlabel("number of samples N")  # ADDED
    plt.ylabel("volume estimate")  # ADDED
    plt.title("Task 1: convergence, uniform sampling in [-1.5, 1.5]^3")  # ADDED
    plt.ylim(1.0, 3.5)  # ADDED
    plt.legend()  # ADDED
    plt.tight_layout()  # ADDED
    plt.savefig("task1_convergence.png", dpi=150)  # ADDED
    plt.close()  # ADDED

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

    # ADDED: restart the generators so case b uses the same sequence as case a (fair comparison)
    x_gen = logistic_map_generator(0.9, 3.8)  # ADDED
    y_gen = logistic_map_generator(0.8, 3.8)  # ADDED
    z_gen = logistic_map_generator(0.7, 3.8)  # ADDED

    int2_logistic = MonteCarloIntegrator(
        intersect2,
        bounds2,
        100_000,
        coordinate_generators=(x_gen, y_gen, z_gen),
    )
    
    volume2_logistic = int2_logistic.integrate()
    print(f"Estimated volume of the sphere using logistic map generator (Case b): {volume2_logistic}")

    # ADDED: compare logistic vs uniform estimate, measured in uniform standard errors
    for name, uni, logi in [("a", integrator1, volume1_logistic),
                            ("b", integrator2, volume2_logistic)]:  # ADDED
        uni_vol = uni.running_estimate()[-1]  # ADDED
        diff = logi - uni_vol  # ADDED
        print(f"  Case {name}: logistic - uniform = {diff:+.4f} "
              f"= {diff / uni.standard_error():+.1f} standard errors")  # ADDED

    # ADDED: diagnostics, why the logistic map is not a valid MC sampler
    diag_gen = logistic_map_generator(0.9, 3.8)  # ADDED: fresh x-sequence
    seq = np.array([next(diag_gen) for _ in range(100_000)])  # ADDED
    m = 3.8  # ADDED
    print(f"  logistic sequence range: [{seq.min():.4f}, {seq.max():.4f}]  "
          f"(theory: [{m**2 / 4 * (1 - m / 4):.4f}, {m / 4:.4f}]) -> part of the box is never sampled")  # ADDED
    print(f"  lag-1 autocorrelation: {np.corrcoef(seq[:-1], seq[1:])[0, 1]:.3f}  "
          f"(~0 for independent samples)")  # ADDED

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))  # ADDED
    axes[0].hist(np.random.uniform(0, 1, len(seq)), bins=100, density=True, alpha=0.6, label="np.random.uniform")  # ADDED
    axes[0].hist(seq, bins=100, density=True, alpha=0.6, label="logistic map")  # ADDED
    axes[0].set_title("Distribution of values in [0,1]")  # ADDED
    axes[0].legend()  # ADDED
    axes[1].scatter(seq[:-1][:3000], seq[1:][:3000], s=2)  # ADDED
    axes[1].set_xlabel("$X_n$")  # ADDED
    axes[1].set_ylabel("$X_{n+1}$")  # ADDED
    axes[1].set_title("$X_{n+1}$ vs $X_n$: fully dependent")  # ADDED
    axes[2].semilogx(n_axis, integrator1.running_estimate(), label="uniform, case a")  # ADDED
    axes[2].semilogx(n_axis, int1_logistic.running_estimate(), label="logistic, case a")  # ADDED
    axes[2].semilogx(n_axis, integrator2.running_estimate(), label="uniform, case b")  # ADDED
    axes[2].semilogx(n_axis, int2_logistic.running_estimate(), label="logistic, case b")  # ADDED
    axes[2].set_ylim(1.0, 3.5)  # ADDED
    axes[2].set_xlabel("number of samples N")  # ADDED
    axes[2].set_title("Convergence: logistic settles on the wrong value")  # ADDED
    axes[2].legend()  # ADDED
    plt.tight_layout()  # ADDED
    plt.savefig("task2_logistic_diagnostics.png", dpi=150)  # ADDED
    plt.close()  # ADDED


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