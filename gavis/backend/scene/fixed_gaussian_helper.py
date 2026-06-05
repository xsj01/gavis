import numpy as np

def sample_volume_points(boundaries, N):
    """
    Uniformly samples N points from the volume defined by boundaries.
    
    Args:
        boundaries: A numpy array of shape (3, 2) where each row gives [low, high] for x, y, z.
        N: Number of points to sample.
    
    Returns:
        points: A numpy array of shape (N, 3) with points uniformly sampled from the volume.
    """
    boundaries = np.asarray(boundaries)
    low = boundaries[:, 0]    # shape (3,)
    high = boundaries[:, 1]   # shape (3,)
    diff = high - low         # shape (3,)

    # np.random.rand(N, 3) generates values in [0, 1) for each coordinate.
    points = np.random.rand(N, 3) * diff + low
    return points

def sample_shell_point(boundaries, D):
    # Extract low and high from boundaries
    low = boundaries[:, 0]
    high = boundaries[:, 1]
    L = high - low
    
    # Compute total and inner volumes
    V_total = np.prod(L)
    L_inner = L - 2*D
    V_inner = np.prod(L_inner)
    V_shell = V_total - V_inner
    
    # Compute disjoint volumes for Regions A, B, C:
    V_A = 2 * D * ((L[1] - 2*D) * (L[2] - 2*D) +
                   (L[0] - 2*D) * (L[2] - 2*D) +
                   (L[0] - 2*D) * (L[1] - 2*D))
    V_B = 4 * (D**2) * ((L[0] - 2*D) + (L[1] - 2*D) + (L[2] - 2*D))
    V_C = 8 * (D**3)
    
    # Determine region probabilities:
    p_A = V_A / V_shell
    p_B = V_B / V_shell
    # p_C = V_C / V_shell, but we can use else
    
    r = np.random.rand()
    point = np.zeros(3)
    
    if r < p_A:
        # Region A: exactly one coordinate is outer.
        outer_coord = np.random.choice(3)
        # Sample outer coordinate:
        if np.random.rand() < 0.5:
            point[outer_coord] = low[outer_coord] + np.random.rand() * D
        else:
            point[outer_coord] = high[outer_coord] - np.random.rand() * D
        # Other coordinates come from the inner intervals:
        for i in range(3):
            if i != outer_coord:
                point[i] = low[i] + D + np.random.rand() * (L[i] - 2*D)
    elif r < p_A + p_B:
        # Region B: exactly two coordinates are outer.
        inner_coord = np.random.choice(3)
        # Sample inner coordinate uniformly from the inner interval:
        point[inner_coord] = low[inner_coord] + D + np.random.rand() * (L[inner_coord] - 2*D)
        # For the other two coordinates, sample from outer intervals:
        for i in range(3):
            if i != inner_coord:
                if np.random.rand() < 0.5:
                    point[i] = low[i] + np.random.rand() * D
                else:
                    point[i] = high[i] - np.random.rand() * D
    else:
        # Region C: all three coordinates are outer.
        for i in range(3):
            if np.random.rand() < 0.5:
                point[i] = low[i] + np.random.rand() * D
            else:
                point[i] = high[i] - np.random.rand() * D
    return point

def sample_shell_points(boundaries, D, N):
    points = np.array([sample_shell_point(boundaries, D) for _ in range(N)])
    return points

# Example usage:
boundaries = np.array([[-3, 3], [-3, 3], [-3, 3]])
D = 0.5
N = 10000
shell_points = sample_shell_points(boundaries, D, N)
print("Shape of shell samples:", shell_points.shape)