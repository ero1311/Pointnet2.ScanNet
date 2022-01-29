import numpy as np

hsv_transpose_dict = {
    0: np.array([0, 1, 2]),
    1: np.array([1, 0, 2]),
    2: np.array([2, 0, 1]),
    3: np.array([2, 1, 0]),
    4: np.array([1, 2, 0]),
    5: np.array([0, 2, 1])
}
def hsv_to_rgb(h, s, v):
    c = s * v
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    rgb_prime = np.array([c, x, 0])
    rgb = (255 * (rgb_prime[hsv_transpose_dict[h // 60]] + m)).astype('uint8')
    
    return rgb

def generate_n_colors(N):
    colors = []
    for h in np.linspace(0, 360, num=N, endpoint=False):
        rgb = hsv_to_rgb(h, 1, 1)
        colors.append((rgb[0], rgb[1], rgb[2]))
    
    return colors
