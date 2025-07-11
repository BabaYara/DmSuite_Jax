import jax.numpy as jnp

def cheb(N: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Chebyshev points and differentiation matrix.

    Args:
        N: Order of the polynomial (number of points - 1).

    Returns:
        A tuple containing:
            - x: Chebyshev points (N+1).
            - D: Differentiation matrix (N+1, N+1).
    """
    if N == 0:
        return jnp.array([0.0]), jnp.array([[0.0]])

    x = jnp.cos(jnp.pi * jnp.arange(N + 1) / N)
    c = jnp.concatenate([jnp.array([2.0]), jnp.ones(N - 1), jnp.array([2.0])]) * (-1) ** jnp.arange(N + 1)

    X = jnp.tile(x, (N + 1, 1))
    dX = X - X.T

    D = (c[:, None] * (1.0 / c)[None, :]) / (dX + jnp.eye(N + 1))
    D = D - jnp.diag(jnp.sum(D, axis=1))  # Corrected sum over rows

    return x, D
