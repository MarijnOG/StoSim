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
        x, y, z = point
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

        points = []
        for _ in range(self.num_samples):
            if self.coordinate_generators is None:
                point = self.rng(lower, upper, size=3)
            else:
                unit_point = np.array(
                    [next(generator) for generator in self.coordinate_generators],
                    dtype=float,
                )
                point = lower + unit_point * span
            points.append(point)

        self.points = np.asarray(points)
        self.labels = np.array([self.shape.is_inside(point) for point in self.points])
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

    def plot_points(self, filename=None, open=False):
        """Plot the generated points in 3D, colored by whether they are inside."""
        if not hasattr(self, "points") or not hasattr(self, "labels"):
            raise RuntimeError("generate_and_label_points must be called first")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

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
        ax.set_title("Monte Carlo Sample Points")
        ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
        ax.set_ylim(self.bounds[0][1], self.bounds[1][1])
        ax.set_zlim(self.bounds[0][2], self.bounds[1][2])
        ax.legend()
        if filename:
            plt.savefig(filename)
        if open:
            plt.show()

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


class MixtureMonteCarloIntegrator:
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
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.shape = shape
        self.bounds_big = bounds_big
        self.bounds_small = bounds_small
        self.probability_big = probability_big
        self.num_samples = num_samples
        self.rng = rng if rng is not None else np.random.uniform

    @staticmethod
    def _box_data(bounds):
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("bounds must contain two valid three-dimensional limits")
        return lower, upper, np.prod(upper - lower)

    def estimate_integrand(self):
        """Return an importance-sampling estimate over the full large box."""
        lower_big, upper_big, volume_big = self._box_data(self.bounds_big)
        lower_small, upper_small, volume_small = self._box_data(self.bounds_small)
        weights = []

        for _ in range(self.num_samples):
            if self.rng(0, 1) < self.probability_big:
                point = self.rng(lower_big, upper_big, size=3)
            else:
                point = self.rng(lower_small, upper_small, size=3)

            in_small = np.all((point >= lower_small) & (point <= upper_small))
            density = self.probability_big / volume_big
            if in_small:
                density += (1 - self.probability_big) / volume_small
            weights.append(self.shape.is_inside(point) / density)

        weights = np.asarray(weights, dtype=float)
        self.standard_error = weights.std(ddof=1) / np.sqrt(self.num_samples)
        return weights.mean()


def logistic_map_generator(x, m):
    while True:
        x = m * x * (1 - x)
        yield x

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
    integrator1.plot_points("task1_points.png", open=True)
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
    integrator2.plot_points("task1_points_b.png", open=True)
    print(f"Estimated volume of the intersect using numpy.random.uniform(Case b): {volume2}")
    print(f"  box used: {bounds2}, box volume = {integrator2.volume_bounds}")
    print(f"  standard error = {integrator2.standard_error():.4f}, "
          f"95% CI = [{volume2 - 1.96 * integrator2.standard_error():.4f}, "
          f"{volume2 + 1.96 * integrator2.standard_error():.4f}]")

    n_axis = np.arange(1, 100000 + 1)
    plt.figure(figsize=(8, 4.5))
    plt.semilogx(n_axis, integrator1.running_estimate(), label="Case a")
    plt.semilogx(n_axis, integrator2.running_estimate(), label="Case b")
    plt.xlabel("number of samples N")
    plt.ylabel("volume estimate")
    plt.title("Task 1: convergence with tight bounding boxes")
    plt.ylim(1.0, 3.5)
    plt.legend()
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
    int1_logistic.plot_points("task2_points_a.png", open=True)
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
    int2_logistic.plot_points("task2_points_b.png", open=True)
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

    # b: using importance sampling around (0, 0, 0.1)
    bounds3_small = np.array([
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.7),  # centered at z = 0.1
    ])
    integrator3_2 = MonteCarloIntegrator(intersect3, bounds3_small, 100000)
    integrator3_2.generate_and_label_points()
    volume3_2 = integrator3_2.estimate_integrand()
    print(f"Estimated volume of the intersect using importance sampling (off-center torus): {volume3_2}")

    mixture_integrator3 = MixtureMonteCarloIntegrator(
        intersect3,
        bounds3_big,
        bounds3_small,
        probability_big=0.2,
        num_samples=100000,
    )
    mixture_volume3 = mixture_integrator3.estimate_integrand()
    print(
        "Estimated volume of the full off-center intersection using mixture "
        f"importance sampling: {mixture_volume3}"
    )
    print(f"  standard error = {mixture_integrator3.standard_error:.4f}")