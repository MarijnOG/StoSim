import numpy as np


class Torus:
    def __init__(self, x, y, z, R, r):
        self.x = x
        self.y = y
        self.z = z
        self.R = R
        self.r = r

    def is_inside(self, point):
        x, y, z = np.asarray(point).T
        return (np.sqrt((x - self.x) ** 2 + (y - self.y) ** 2) - self.R) ** 2 + (z - self.z) ** 2 <= self.r ** 2

    def plot(self, ax, num_points=100, color="tab:blue", alpha=0.25):
        """Plot the torus in 3D using matplotlib."""
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, 2 * np.pi, num_points)
        u, v = np.meshgrid(u, v)

        x = (self.R + self.r * np.cos(v)) * np.cos(u) + self.x
        y = (self.R + self.r * np.cos(v)) * np.sin(u) + self.y
        z = self.r * np.sin(v) + self.z

        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

class Sphere:
    def __init__(self, x, y, z, k):
        self.x = x
        self.y = y
        self.z = z
        self.r = k

    def is_inside(self, point):
        x, y, z = np.asarray(point).T
        return (x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2 <= self.r ** 2

    def plot(self, ax, num_points=100, color="tab:red", alpha=0.2):
        """Plot the sphere in 3D using matplotlib."""
        u = np.linspace(0, 2 * np.pi, num_points)
        v = np.linspace(0, 2 * np.pi, num_points)
        u, v = np.meshgrid(u, v)

        x = self.r * np.sin(v) * np.cos(u) + self.x
        y = self.r * np.sin(v) * np.sin(u) + self.y
        z = self.r * np.cos(v) + self.z

        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

class IntersectShape:
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2

    def is_inside(self, point):
        return np.logical_and(
            self.shape1.is_inside(point),
            self.shape2.is_inside(point),
        )

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

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.shape = shape
        self.bounds = bounds
        self.num_samples = num_samples
        self.rng = rng if rng is not None else np.random.uniform

    def generate_and_label_points(self):
        """Generate sample points and label each point by whether it is inside."""
        lower = np.asarray(self.bounds[0], dtype=float)
        upper = np.asarray(self.bounds[1], dtype=float)
        span = upper - lower
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("bounds must contain two valid three-dimensional limits")

        if self.coordinate_generators is None:
            points = self.rng(lower, upper, size=(self.num_samples, 3))
        else:
            points = []
            for _ in range(self.num_samples):
                unit_point = np.array(
                    [next(generator) for generator in self.coordinate_generators],
                    dtype=float,
                )
                point = lower + unit_point * span
                points.append(point)

        self.points = np.asarray(points)
        self.labels = np.asarray(self.shape.is_inside(self.points))
        self.running_counts = np.cumsum(self.labels)
        self.volume_bounds = np.prod(span)
        return self.points, self.labels

    def estimate_integrand(self):
        """Return the Monte Carlo estimate from the generated point labels."""
        if not hasattr(self, "labels"):
            raise RuntimeError("generate_and_label_points must be called first")
        self.p_hat = np.mean(self.labels)
        return self.volume_bounds * self.p_hat

    def standard_error(self):
        """Return the Bernoulli standard error for uniform independent samples."""
        if not hasattr(self, "p_hat"):
            raise RuntimeError("estimate_integrand must be called first")
        return self.volume_bounds * np.sqrt(
            self.p_hat * (1 - self.p_hat) / self.num_samples
        )

    def running_estimate(self):
        """Return the estimate after each generated sample."""
        if not hasattr(self, "running_counts"):
            raise RuntimeError("generate_and_label_points must be called first")
        samples = np.arange(1, self.num_samples + 1)
        return self.volume_bounds * self.running_counts / samples

    def plot_points(self, filename=None, open=False, title="Monte Carlo Sample Points"):
        """Plot the generated points in 3D, colored by whether they are inside."""
        if not hasattr(self, "points") or not hasattr(self, "labels"):
            raise RuntimeError("generate_and_label_points must be called first")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.points[self.labels, 0],
            self.points[self.labels, 1],
            self.points[self.labels, 2],
            color="green",
            alpha=0.1,
            label="Inside",
            marker=".",
        )
        ax.scatter(
            self.points[~self.labels, 0],
            self.points[~self.labels, 1],
            self.points[~self.labels, 2],
            color="red",
            alpha=0.005,
            label="Outside",
            marker=".",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
        ax.set_ylim(self.bounds[0][1], self.bounds[1][1])
        ax.set_zlim(self.bounds[0][2], self.bounds[1][2])
        ax.set_box_aspect(np.asarray(self.bounds[1]) - np.asarray(self.bounds[0]))
        ax.legend()
        if filename:
            plt.savefig(filename)
        if open:
            plt.show()
        plt.close(fig)

    def plot_shapes(self, filename=None, open=False, num_points=80):
        """Plot both component shapes and highlight sampled intersection points."""
        if not hasattr(self, "points") or not hasattr(self, "labels"):
            raise RuntimeError("generate_and_label_points must be called first")
        if not isinstance(self.shape, IntersectShape):
            raise TypeError("plot_shapes requires an IntersectShape")

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        self.shape.shape1.plot(ax, num_points=num_points, color="tab:red", alpha=0.18)
        self.shape.shape2.plot(ax, num_points=num_points, color="tab:blue", alpha=0.22)

        intersection = self.points[self.labels]
        ax.scatter(
            intersection[:, 0],
            intersection[:, 1],
            intersection[:, 2],
            color="gold",
            s=4,
            alpha=0.8,
            label="Sampled intersection",
        )

        lower = np.asarray(self.bounds[0], dtype=float)
        upper = np.asarray(self.bounds[1], dtype=float)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(upper - lower)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Shapes and sampled intersection")
        ax.legend(handles=[
            Line2D([0], [0], color="tab:red", lw=8, alpha=0.5, label="Sphere"),
            Line2D([0], [0], color="tab:blue", lw=8, alpha=0.5, label="Torus"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
                   markersize=7, label="Sampled intersection"),
        ])
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches="tight")
        if open:
            plt.show()
        plt.close(fig)


class MixtureMonteCarloIntegrator(MonteCarloIntegrator):
    """Estimate an integral by sampling from a mixture of two uniform boxes."""

    def __init__(
        self,
        shape,
        bounds_big,
        bounds_small,
        probability_big,
        num_samples,
        rng=None,
    ):
        if not 0 < probability_big <= 1:
            raise ValueError("probability_big must be in the interval (0, 1]")
        super().__init__(shape, bounds_big, num_samples, rng=rng)
        self.bounds_big = bounds_big
        self.bounds_small = bounds_small
        self.probability_big = probability_big

    @staticmethod
    def _box_data(bounds):
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("bounds must contain two valid three-dimensional limits")
        return lower, upper, np.prod(upper - lower)

    def estimate_integrand(self):
        """Return an importance-sampling estimate over the full large box."""
        if not hasattr(self, "weights"):
            self.generate_and_label_points()
        return self.weights.mean()

    def naive_estimate(self):
        """Return the intentionally incorrect unweighted mixture estimate."""
        if not hasattr(self, "labels"):
            raise RuntimeError("generate_and_label_points must be called first")
        return self.volume_bounds * np.mean(self.labels)

    def generate_and_label_points(self):
        """Generate mixture samples, labels, proposal densities, and weights."""
        lower_big, upper_big, volume_big = self._box_data(self.bounds_big)
        lower_small, upper_small, volume_small = self._box_data(self.bounds_small)
        if np.any(lower_small < lower_big) or np.any(upper_small > upper_big):
            raise ValueError("small box must lie inside big box")

        self.from_big = self.rng(0, 1, size=self.num_samples) < self.probability_big
        big_points = self.rng(lower_big, upper_big, size=(self.num_samples, 3))
        small_points = self.rng(lower_small, upper_small, size=(self.num_samples, 3))
        self.points = np.where(self.from_big[:, None], big_points, small_points)
        in_small = np.all(
            (self.points >= lower_small) & (self.points <= upper_small),
            axis=1,
        )
        self.q = self.probability_big / volume_big
        self.q += (1 - self.probability_big) / volume_small * in_small
        self.labels = np.asarray(self.shape.is_inside(self.points))
        self.weights = self.labels / self.q
        self.volume_bounds = volume_big
        self.running_counts = np.cumsum(self.weights)
        self.naive_running_counts = np.cumsum(self.labels) * self.volume_bounds
        return self.points, self.labels

    def standard_error(self):
        """Return the estimated standard error of the weighted mean."""
        if not hasattr(self, "weights"):
            raise RuntimeError("generate_and_label_points must be called first")
        return self.weights.std(ddof=1) / np.sqrt(self.num_samples)

    def running_estimate(self):
        """Return the weighted estimate after each generated sample."""
        if not hasattr(self, "running_counts"):
            raise RuntimeError("generate_and_label_points must be called first")
        return self.running_counts / np.arange(1, self.num_samples + 1)

    def naive_running_estimate(self):
        """Return the intentionally incorrect unweighted running estimate."""
        if not hasattr(self, "naive_running_counts"):
            raise RuntimeError("generate_and_label_points must be called first")
        return self.naive_running_counts / np.arange(1, self.num_samples + 1)


def logistic_map_generator(x, m):
    while True:
        x = m * x * (1 - x)
        yield x


def reference_volume(k, R, r, zc=0.0, num_intervals=20000):
    """Numerically integrate the exact cylindrical cross-sectional volume."""
    rho = np.linspace(max(R - r, 0), min(R + r, k), num_intervals + 1)
    sphere_half_height = np.sqrt(np.maximum(k**2 - rho**2, 0))
    torus_radical = r**2 - (rho - R)**2
    torus_half_height = np.sqrt(np.maximum(torus_radical, 0))
    lower = np.maximum(-sphere_half_height, zc - torus_half_height)
    upper = np.minimum(sphere_half_height, zc + torus_half_height)
    integrand = 2 * np.pi * rho * np.maximum(upper - lower, 0)
    return np.sum((integrand[:-1] + integrand[1:]) * np.diff(rho) / 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Task 1: Shapes at origin, evaluate intersect for: 
    # Case a: k=1, R = 0.75 and r = 0.4
    # Case b: k= 1, R = 0.5 and r = 0.5

    np.random.seed(42)  # For reproducibility
    sphere1 = Sphere(0, 0, 0, 1)
    torus1 = Torus(0, 0, 0, 0.75, 0.4)
    intersect1 = IntersectShape(sphere1, torus1)
    bounds1 = np.array([(-1, -1, -0.4), (1, 1, 0.4)])
    integrator1 = MonteCarloIntegrator(intersect1, bounds1, 100000)
    integrator1.generate_and_label_points()
    volume1 = integrator1.estimate_integrand()
    integrator1.plot_points("task1_points.png", open=False)
    print(f"Estimated volume of the intersect using numpy.random.uniform(Case a): {volume1}")
    print(f"  box used: {bounds1}, box volume = {integrator1.volume_bounds}")
    print(f"  standard error = {integrator1.standard_error():.4f}, "
          f"95% CI = [{volume1 - 1.96 * integrator1.standard_error():.4f}, "
          f"{volume1 + 1.96 * integrator1.standard_error():.4f}]")

    sphere2 = Sphere(0, 0, 0, 1)
    torus2 = Torus(0, 0, 0, 0.5, 0.5)
    intersect2 = IntersectShape(sphere2, torus2)
    bounds2 = np.array([(-1, -1, -0.5), (1, 1, 0.5)])
    integrator2 = MonteCarloIntegrator(intersect2, bounds2, 100000)
    integrator2.generate_and_label_points()
    volume2 = integrator2.estimate_integrand()
    integrator2.plot_points("task1_points_b.png", open=False)
    print(f"Estimated volume of the intersect using numpy.random.uniform(Case b): {volume2}")
    print(f"  box used: {bounds2}, box volume = {integrator2.volume_bounds}")
    print(f"  standard error = {integrator2.standard_error():.4f}, "
          f"95% CI = [{volume2 - 1.96 * integrator2.standard_error():.4f}, "
          f"{volume2 + 1.96 * integrator2.standard_error():.4f}]")

    references = {
        "a": reference_volume(1, 0.75, 0.4),
        "b": reference_volume(1, 0.5, 0.5),
    }
    n_axis = np.arange(1, 100000 + 1)
    plt.figure(figsize=(8, 4.5))
    plt.semilogx(n_axis, integrator1.running_estimate(), label="Case a")
    plt.semilogx(n_axis, integrator2.running_estimate(), label="Case b")
    plt.axhline(references["a"], color="tab:blue", linestyle="--", alpha=0.7)
    plt.axhline(references["b"], color="tab:orange", linestyle="--", alpha=0.7)
    plt.xlabel("number of samples N")
    plt.ylabel("volume estimate")
    plt.title("Task 1: convergence with tight bounding boxes")
    plt.ylim(1.0, 3.5)
    plt.legend()
    plt.text(
        0.02,
        0.03,
        f"Final estimates: a={volume1:.4f}, b={volume2:.4f}\n"
        f"Reference: a={references['a']:.4f}, b={references['b']:.4f}",
        transform=plt.gca().transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    plt.tight_layout()
    plt.savefig("task1_convergence.png", dpi=150)
    plt.close()

    # Show both shapes and highlight the sampled intersection.
    # integrator1.plot_shapes("task0_shapes.png", open=True)

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
    int1_logistic.generate_and_label_points()
    volume1_logistic = int1_logistic.estimate_integrand()
    int1_logistic.plot_points("task2_points_a.png", open=False)
    print(f"Estimated volume of the sphere using logistic map generator (Case a): {volume1_logistic}")

    # Restart the generators so case b uses the same sequence as case a.
    x_gen = logistic_map_generator(0.9, 3.8)
    y_gen = logistic_map_generator(0.8, 3.8)
    z_gen = logistic_map_generator(0.7, 3.8)

    int2_logistic = MonteCarloIntegrator(
        intersect2,
        bounds2,
        100_000,
        coordinate_generators=(x_gen, y_gen, z_gen),
    )
    
    int2_logistic.generate_and_label_points()
    volume2_logistic = int2_logistic.estimate_integrand()
    int2_logistic.plot_points("task2_points_b.png", open=False)
    print(f"Estimated volume of the sphere using logistic map generator (Case b): {volume2_logistic}")

    # Compare logistic and uniform estimates in uniform standard errors.
    for name, uni, logi in [("a", integrator1, volume1_logistic),
                            ("b", integrator2, volume2_logistic)]:
        uni_vol = uni.running_estimate()[-1]
        diff = logi - uni_vol
        print(f"  Case {name}: logistic - uniform = {diff:+.4f} "
              f"= {diff / uni.standard_error():+.1f} standard errors")

    # Diagnose why the logistic map is not a valid MC sampler.
    diag_gen = logistic_map_generator(0.9, 3.8)
    seq = np.array([next(diag_gen) for _ in range(100_000)])
    m = 3.8
    print(f"  logistic sequence range: [{seq.min():.4f}, {seq.max():.4f}]  "
          f"(theory: [{m**2 / 4 * (1 - m / 4):.4f}, {m / 4:.4f}]) -> part of the box is never sampled")
    print(f"  lag-1 autocorrelation: {np.corrcoef(seq[:-1], seq[1:])[0, 1]:.3f}  "
          f"(~0 for independent samples)")

    experiment_rng = np.random.default_rng(0)

    def uniform_run(shape, sample_bounds, num_samples):
        mc = MonteCarloIntegrator(
            shape, sample_bounds, num_samples, rng=experiment_rng.uniform
        )
        mc.generate_and_label_points()
        return mc.estimate_integrand(), mc.standard_error()

    def logistic_run(shape, sample_bounds, num_samples):
        generators = [
            logistic_map_generator(x0, 3.8)
            for x0 in experiment_rng.uniform(0.05, 0.95, 3)
        ]
        mc = MonteCarloIntegrator(
            shape,
            sample_bounds,
            num_samples,
            coordinate_generators=generators,
        )
        mc.generate_and_label_points()
        return mc.estimate_integrand(), mc.standard_error()

    repeated_results = {}
    for name, shape, sample_bounds in [
        ("a", intersect1, bounds1),
        ("b", intersect2, bounds2),
    ]:
        for sampler_name, sampler in [
            ("uniform", uniform_run),
            ("logistic", logistic_run),
        ]:
            results_for_sampler = np.array([
                sampler(shape, sample_bounds, 100_000)
                for _ in range(30)
            ])
            repeated_results[name, sampler_name] = results_for_sampler

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(np.random.uniform(0, 1, len(seq)), bins=100, density=True, alpha=0.6, label="np.random.uniform")
    axes[0].hist(seq, bins=100, density=True, alpha=0.6, label="logistic map")
    axes[0].set_title("Distribution of values in [0,1]")
    axes[0].legend()
    axes[1].scatter(seq[:-1][:3000], seq[1:][:3000], s=2)
    axes[1].set_xlabel("$X_n$")
    axes[1].set_ylabel("$X_{n+1}$")
    axes[1].set_title("$X_{n+1}$ vs $X_n$: fully dependent")
    axes[2].semilogx(n_axis, integrator1.running_estimate(), label="uniform, case a")
    axes[2].semilogx(n_axis, int1_logistic.running_estimate(), label="logistic, case a")
    axes[2].semilogx(n_axis, integrator2.running_estimate(), label="uniform, case b")
    axes[2].semilogx(n_axis, int2_logistic.running_estimate(), label="logistic, case b")
    axes[2].set_ylim(1.0, 3.5)
    axes[2].set_xlabel("number of samples N")
    axes[2].set_title("Convergence: logistic settles on the wrong value")
    axes[2].legend()
    axes[2].text(
        0.02,
        0.03,
        "Logistic: precise but biased\n"
        f"a bias={volume1_logistic - references['a']:+.3f}, "
        f"b bias={volume2_logistic - references['b']:+.3f}\n"
        f"30-run SD: a={repeated_results['a', 'logistic'][:, 0].std(ddof=1):.3f}, "
        f"b={repeated_results['b', 'logistic'][:, 0].std(ddof=1):.3f}",
        transform=axes[2].transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    plt.tight_layout()
    plt.savefig("task2_logistic_diagnostics.png", dpi=150)
    plt.close()

    # Task 3: Estimate volume of a sphere and an off-center torus intersection
    # a: using numpy.random.uniform
    sphere3 = Sphere(0, 0, 0, 1)
    torus3 = Torus(0, 0, 0.1, 0.75, 0.4)
    intersect3 = IntersectShape(sphere3, torus3)
    bounds3_big = np.array([
        (-1.5, -1.5, -1.5),
        (1.5, 1.5, 1.5),
    ])
    integrator3 = MonteCarloIntegrator(intersect3, bounds3_big, 100000)
    integrator3.generate_and_label_points()
    volume3 = integrator3.estimate_integrand()
    print(f"Estimated volume of the intersect using numpy.random.uniform (off-center torus): {volume3}")
    reference3 = reference_volume(1, 0.75, 0.4, 0.1)
    se3 = integrator3.standard_error()
    print(f"  standard error = {se3:.4f}")
    print(
        f"  95% CI = [{volume3 - 1.96 * se3:.4f}, "
        f"{volume3 + 1.96 * se3:.4f}]"
    )
    print(f"  reference volume = {reference3:.4f}")

    # b: mixture sampling from the large box and a geometry-matched small box.
    bounds3_good = np.array([
        (-1, -1, -0.3),
        (1, 1, 0.5),
    ])
    mixture_integrator3 = MixtureMonteCarloIntegrator(
        intersect3,
        bounds3_big,
        bounds3_good,
        probability_big=0.2,
        num_samples=100000,
    )
    mixture_volume3 = mixture_integrator3.estimate_integrand()
    mixture_se3 = mixture_integrator3.standard_error()
    probabilities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}
    for probability in probabilities:
        weighted_estimates = []
        naive_estimates = []
        reported_errors = []
        for _ in range(30):
            mixture = MixtureMonteCarloIntegrator(
                intersect3,
                bounds3_big,
                bounds3_good,
                probability,
                100_000,
            )
            mixture.generate_and_label_points()
            weighted_estimates.append(mixture.estimate_integrand())
            naive_estimates.append(mixture.naive_estimate())
            reported_errors.append(mixture.standard_error())
        results[probability] = (
            np.mean(weighted_estimates),
            np.std(weighted_estimates, ddof=1),
            np.mean(reported_errors),
            np.mean(naive_estimates),
            np.std(naive_estimates, ddof=1),
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    weighted_means = [results[p][0] for p in probabilities]
    naive_means = [results[p][3] for p in probabilities]
    axes[0].plot(
        probabilities, weighted_means, "o-", color="tab:green",
        label="weighted estimate",
    )
    axes[0].plot(
        probabilities, naive_means, "s--", color="tab:red",
        label="unweighted estimate",
    )
    axes[0].axhline(reference3, color="black", linestyle=":", label="reference")
    axes[0].set_xlabel("probability of sampling from big box, p")
    axes[0].set_ylabel("estimated volume")
    axes[0].set_title("Bias: weighted vs. unweighted")
    axes[0].text(
        0.03,
        0.97,
        f"Reference volume = {reference3:.4f}\n"
        f"p=0.2 weighted = {results[0.2][0]:.4f}\n"
        f"p=0.2 unweighted = {results[0.2][3]:.4f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    weighted_spread = [results[p][1] for p in probabilities]
    reported = [results[p][2] for p in probabilities]
    naive_spread = [results[p][4] for p in probabilities]
    axes[1].plot(
        probabilities, weighted_spread, "o-", color="tab:green",
        label="weighted run spread",
    )
    axes[1].plot(
        probabilities, reported, "--", color="tab:green",
        label="weighted reported SE",
    )
    axes[1].plot(
        probabilities, naive_spread, "s-", color="tab:red",
        label="unweighted run spread",
    )
    axes[1].set_xlabel("probability of sampling from big box, p")
    axes[1].set_ylabel("run-to-run spread")
    axes[1].set_title("Variance comparison")
    axes[1].legend()
    fig.suptitle("Task 3b: the unweighted mixture is biased")
    fig.tight_layout()
    fig.savefig("task3_mixture_p_sweep.png", dpi=150)
    plt.close(fig)

    from matplotlib.patches import Rectangle

    good_histogram = MixtureMonteCarloIntegrator(
        intersect3, bounds3_big, bounds3_good, 0.1, 100_000
    )
    good_histogram.generate_and_label_points()
    good_histogram.estimate_integrand()
    good_weights = good_histogram.weights[good_histogram.weights > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(good_weights, bins=20, log=True, color="tab:green")
    ax.set_xlabel("non-zero importance weight")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Task 3b: S_good weights, p=0.1")
    ax.text(
        0.97,
        0.97,
        f"All non-zero weights = {good_weights[0]:.4f}\n"
        f"Estimate = {good_histogram.estimate_integrand():.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig("task3_weight_histogram.png", dpi=150)
    plt.close(fig)

    def plot_cross_section(ax, major_radius, minor_radius, z_center, boxes, title):
        x = np.linspace(-1.5, 1.5, 1000)
        sphere_z = np.sqrt(np.maximum(1 - x**2, 0))
        torus_z = np.sqrt(
            np.maximum(minor_radius**2 - (np.abs(x) - major_radius) ** 2, 0)
        )
        ax.plot(x, sphere_z, "r", label="sphere")
        ax.plot(x, -sphere_z, "r")
        ax.plot(x, z_center + torus_z, "b", label="torus")
        ax.plot(x, z_center - torus_z, "b")
        for box, color, label in boxes:
            (x0, _, z0), (x1, _, z1) = box
            ax.add_patch(
                Rectangle(
                    (x0, z0),
                    x1 - x0,
                    z1 - z0,
                    fill=False,
                    edgecolor=color,
                    linestyle="--",
                    label=label,
                )
            )
        ax.set_xlabel("x (with y=0)")
        ax.set_ylabel("z")
        ax.set_title(title)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.legend()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_cross_section(
        axes[0], 0.75, 0.4, 0.0,
        [(bounds1, "black", "B")],
        "Task 1a: x-z cross-section",
    )
    plot_cross_section(
        axes[1], 0.5, 0.5, 0.0,
        [(bounds2, "black", "B")],
        "Task 1b: x-z cross-section",
    )
    plot_cross_section(
        axes[2], 0.75, 0.4, 0.1,
        [(bounds3_big, "black", "B"), (bounds3_good, "tab:green", "S_good")],
        "Task 3b: x-z cross-section",
    )
    fig.suptitle(
        f"Task 3a uniform: {volume3:.4f} ± {1.96 * se3:.4f} (95% CI)   |   "
        f"Task 3b mixture: {mixture_volume3:.4f} ± {1.96 * mixture_se3:.4f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig("task3_cross_section.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    mixture = MixtureMonteCarloIntegrator(
        intersect3, bounds3_big, bounds3_good, 0.1, 100_000
    )
    mixture.generate_and_label_points()
    ax.semilogx(
        np.arange(1, mixture.num_samples + 1),
        mixture.running_estimate(),
        color="tab:green",
        label="weighted, p=0.1",
    )
    ax.semilogx(
        np.arange(1, mixture.num_samples + 1),
        mixture.naive_running_estimate(),
        color="tab:red",
        label="unweighted, p=0.1",
    )
    ax.axhline(reference3, color="black", linestyle="--", label="reference")
    ax.set_xlabel("number of samples")
    ax.set_ylabel("running estimate")
    ax.set_title("Task 3b: running mixture estimates")
    ax.set_ylim(0, 3.5)
    ax.text(
        0.03,
        0.97,
        f"Final estimate = {mixture.estimate_integrand():.4f}\n"
        f"Unweighted estimate = {mixture.naive_estimate():.4f}\n"
        f"Reference = {reference3:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8},
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("task3_running_estimates.png", dpi=150)
    plt.close(fig)