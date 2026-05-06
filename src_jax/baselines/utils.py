import jax.numpy as jnp

def check_gradient_overflow(grad_input):
    """Check and fix gradient overflow issues."""
    # Get finite values (not inf and not nan)
    finite_mask = jnp.isfinite(grad_input)
    
    max_finite = jnp.nanmax(jnp.where(finite_mask, grad_input, 0))
    min_finite = jnp.nanmin(jnp.where(finite_mask, grad_input, 0))
    
    # Replace +inf with max finite, -inf with min finite
    grad_input = jnp.where(grad_input == jnp.inf, max_finite, grad_input)
    grad_input = jnp.where(grad_input == -jnp.inf, min_finite, grad_input)
    
    # Replace NaN with 0
    grad_input = jnp.where(jnp.isnan(grad_input), 0.0, grad_input)
    
    return grad_input