import skimage
import numpy as np
import tifffile
import numpy as np
from find_device_edges import preprocess_raw_image_fast
import csv
from skimage.measure import regionprops
from skimage.segmentation import flood
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon
from sklearn.linear_model import RANSACRegressor

def _sweep_one_direction(binary_mask, seed_y, seed_x, direction, step, max_advance,
                        leak_frac=0.40):
    """
    Grow a strip from the seed row outward in one direction.
    Returns (advances, fractions, leak_idx) where leak_idx is the index at which
    leakage was first detected (filled pixels > leak_frac * total image), or None.
    """
    A, B = binary_mask.shape
    total_area = A * B
    advances, fractions = [], []
    leak_idx = None

    for i, adv in enumerate(range(step, max_advance + 1, step)):
        if direction == +1:
            y0, y1 = seed_y, seed_y + adv + 1
            seed_local = (0, seed_x)
        else:
            y0, y1 = seed_y - adv, seed_y + 1
            seed_local = (adv, seed_x)

        if y0 < 0 or y1 > A:
            break

        strip = binary_mask[y0:y1, :]
        filled = flood(strip, seed_local, connectivity=1)
        n_filled = filled.sum()

        # Leakage check: stop and record, but keep this point in the arrays
        # so the plot still shows the spike that triggered it.
        if n_filled > leak_frac * total_area:
            leak_idx = i
            fractions.append(n_filled / (B * adv))
            advances.append(adv)
            break

        fractions.append(n_filled / (B * adv))
        advances.append(adv)

    return np.array(advances), np.array(fractions), leak_idx

def _cusum_breakpoint(fractions, baseline_n=5, slack_frac=0.03, threshold_frac=0.005):
    """
    First index where the cumulative downward deviation from the plateau
    baseline exceeds threshold. Returns None if no change detected.
    """
    if len(fractions) <= baseline_n:
        return None

    baseline = fractions[:baseline_n].mean()
    slack = slack_frac * baseline
    threshold = threshold_frac * baseline

    cusum = 0.0
    for i in range(baseline_n, len(fractions)):
        cusum = min(0.0, cusum + (fractions[i] - baseline + slack))
        if cusum < -threshold:
            return i
    return None


def _step_breakpoint(fractions, baseline_n=5, jump_frac=0.05):
    if len(fractions) <= baseline_n:
        return None
    baseline = fractions[:baseline_n].mean()
    jump_threshold = jump_frac * baseline
    for i in range(baseline_n, len(fractions)):
        if abs(fractions[i] - fractions[i-1]) > jump_threshold:
            return i - 1
    return None

def analyze_growth(binary_mask, plot=True, write_csv = False, csv_filename='sweep_data.csv'):
    A, B = binary_mask.shape
    seed_y, seed_x = A // 2, B // 2

    half = max(1, int(0.05 * A))
    strip0 = binary_mask[seed_y - half : seed_y + half + 1, :]
    fill0 = flood(strip0, (half, seed_x), connectivity=1)
    fill0 = binary_fill_holes(fill0)
    if fill0.sum() > 0.38 * A * B:
        return None, None

    step = max(1, int(0.01 * A))
    max_advance = int(0.40 * A)

    adv_up,   frac_up,   leak_up   = _sweep_one_direction(binary_mask, seed_y, seed_x, -1, step, max_advance)
    adv_down, frac_down, leak_down = _sweep_one_direction(binary_mask, seed_y, seed_x, +1, step, max_advance)

    k_up_cusum = _cusum_breakpoint(frac_up)
    k_down_cusum = _cusum_breakpoint(frac_down)
    k_up_step = _step_breakpoint(frac_up)
    k_down_step = _step_breakpoint(frac_down)

    candidates_up = [k for k in [k_up_cusum, k_up_step] if k is not None]
    k_up = min(candidates_up) if candidates_up else len(adv_up) - 1

    candidates_down = [k for k in [k_down_cusum, k_down_step] if k is not None]
    k_down = min(candidates_down) if candidates_down else len(adv_down) - 1

    if k_up   is None: k_up   = len(adv_up)   - 1
    if k_down is None: k_down = len(adv_down) - 1

    cut_up   = int(adv_up[k_up])
    cut_down = int(adv_down[k_down])

    y0, y1 = seed_y - cut_up, seed_y + cut_down + 1
    strip = binary_mask[y0:y1, :]
    seed_local = (cut_up, seed_x)
    filled_strip = flood(strip, seed_local, connectivity=1)
    filled_strip = binary_fill_holes(filled_strip)

    out_mask = np.zeros_like(binary_mask, dtype=bool)
    out_mask[y0:y1, :] = filled_strip

    if write_csv:

        # --- CSV Export Section ---
        # We calculate gradients ahead of time to export them alongside the raw data
        grad_up = np.gradient(frac_up, adv_up)
        grad_down = np.gradient(frac_down, adv_down)

        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Header row
            writer.writerow(['direction', 'advance_pixels', 'advance_pct', 'fraction', 'gradient', 'is_breakpoint'])
            
            # Write upward data
            for i in range(len(adv_up)):
                is_bk = 1 if i == k_up else 0
                writer.writerow(['upward', adv_up[i], (adv_up[i] / A * 100), frac_up[i], grad_up[i], is_bk])
                
            # Write downward data
            for i in range(len(adv_down)):
                is_bk = 1 if i == k_down else 0
                writer.writerow(['downward', adv_down[i], (adv_down[i] / A * 100), frac_down[i], grad_down[i], is_bk])
        # --------------------------

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')
        
        for col, (adv, frac, grad, k, label) in enumerate([
            (adv_up,   frac_up,   grad_up,   k_up,   'upward'),
            (adv_down, frac_down, grad_down, k_down, 'downward'),
        ]):
            x_pct = adv / A * 100
            ax_top = axes[0, col]
            ax_top.plot(x_pct, frac, 'o-', markersize=8, linewidth=7.5, label='data')
            ax_top.axvline(adv[k] / A * 100, color='k', linestyle='--', alpha=0.5,
                        linewidth=1.0,
                        label=f'breakpoint = {adv[k]/A*100:.1f}%')
            ax_top.set_ylabel('filled / ROI area')
            ax_top.set_title(f'{label} sweep')
            ax_top.grid(True, alpha=0.3)
            ax_top.legend(fontsize=8)
            
            ax_bot = axes[1, col]
            ax_bot.plot(x_pct, grad, 'o-', markersize=8, linewidth=7.5, color='C1')
            ax_bot.axhline(0, color='k', linewidth=1.0)
            ax_bot.axvline(adv[k] / A * 100, color='k', linestyle='--', alpha=0.5,
                        linewidth=1.0)
            ax_bot.set_xlabel('advance from seed (% of A)')
            ax_bot.set_ylabel('d(fraction) / d(advance)')
            ax_bot.grid(True, alpha=0.3)
            
        plt.tight_layout()
        fig.savefig('sweep.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)

    return out_mask  # Added explicit return assumin


def create_rectangular_mask(processed_binary):
    output_img = np.zeros_like(processed_binary)
    max_slope = np.tan(np.deg2rad(15))
    ransac_image = np.zeros_like(processed_binary)
    support_mask = processed_binary 
    walk_back = 50 

    A, B = processed_binary.shape

    outer_m, outer_c = [], []
    channel_wall_lines = []

    # For each row, find leftmost and rightmost mask pixel
    rows = np.arange(A)
    left_pts, right_pts = [], []
    for y in rows:
        xs = np.where(processed_binary[y, :] == 1)[0]
        if xs.size:
            left_pts.append((y, xs[0]))
            right_pts.append((y, xs[-1]))
    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)

    for side_idx, (is_right, points) in enumerate([(False, left_pts), (True, right_pts)]):
        if len(points) > 10:
            ransac = RANSACRegressor(residual_threshold=4).fit(
                points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1))
            m = np.clip(ransac.estimator_.coef_[0][0], -max_slope, max_slope)
            c = ransac.estimator_.intercept_[0]
            c = c - 80 if is_right else c + 80   # outward offset, same as before
            outer_m.append(m)
            outer_c.append(c)

            inlier_points = points[ransac.inlier_mask_]
            y_min_channel, y_max_channel = 0, A - 1
            channel_wall_lines.append((m, c, y_min_channel, y_max_channel, side_idx))

            y_channel = np.arange(int(y_min_channel), int(y_max_channel)).reshape(-1, 1)
            x_fitted = (m * y_channel + c).astype(int)
            v = (x_fitted >= 0) & (x_fitted < B)
            output_img[y_channel[v].flatten(), x_fitted[v].flatten()] = 255
            ransac_image[y_channel[v].flatten(), x_fitted[v].flatten()] = 255

            for pt in inlier_points:
                y_idx, x_idx = pt[0], pt[1]
                if 0 <= y_idx < A and 0 <= x_idx < B:
                    processed_binary[y_idx, x_idx] = 0

    def support_range(m, c):
        y = np.arange(A)
        x = np.round(m * y + c).astype(int)
        valid = (x >= 0) & (x < B)
        supported = np.zeros(A, dtype=bool)
        supported[valid] = support_mask[y[valid], x[valid]] == 1
        ys = np.where(supported)[0]
        if ys.size == 0:
            return 0, A - 1
        return int(ys.min()), int(ys.max())

    m_l, c_l = outer_m[0], outer_c[0]
    m_r, c_r = outer_m[1], outer_c[1]

    y_l_min, y_l_max = support_range(m_l, c_l)
    y_r_min, y_r_max = support_range(m_r, c_r)

    # Walk 50 inward on each end of each line
    y_l_top = min(y_l_min + walk_back, A - 1)
    y_l_bot = max(y_l_max - walk_back, 0)
    y_r_top = min(y_r_min + walk_back, A - 1)
    y_r_bot = max(y_r_max - walk_back, 0)

    # Four corners, ordered TL -> TR -> BR -> BL (clockwise in image coords)
    corner_y = np.array([y_l_top, y_r_top, y_r_bot, y_l_bot])
    corner_x = np.array([
        m_l * y_l_top + c_l,
        m_r * y_r_top + c_r,
        m_r * y_r_bot + c_r,
        m_l * y_l_bot + c_l,
    ]).round().astype(int)

    # Fill
    rr, cc = polygon(corner_y, corner_x, shape=(A, B))
    filled = np.zeros((A, B), dtype=np.uint8)
    filled[rr, cc] = 1

    # If you want it on output_img:
    output_img[rr, cc] = 255
    output_img[ransac_image > 0] = 0

    final_mask = output_img

    # Corner x-coordinates of the rectangle (from previous step)
    x_l_top = m_l * y_l_top + c_l
    x_r_top = m_r * y_r_top + c_r
    x_l_bot = m_l * y_l_bot + c_l
    x_r_bot = m_r * y_r_bot + c_r

    y_grid, x_grid = np.mgrid[0:A, 0:B].astype(np.float32)

    # ---- EDT to walls: perpendicular distance to nearer RANSAC line ----
    # Signed perp distance to x = m*y + c. Positive on the side we want inside.
    denom_l = np.sqrt(1 + m_l**2)
    denom_r = np.sqrt(1 + m_r**2)
    dist_left  = (x_grid - m_l * y_grid - c_l) / denom_l    # >0 right of left wall
    dist_right = (m_r * y_grid + c_r - x_grid) / denom_r    # >0 left  of right wall

    edt_to_walls = np.minimum(dist_left, dist_right)        # dist to nearer side wall
    edt_to_walls = np.clip(edt_to_walls, 0, None)
    edt_to_walls *= (final_mask > 0)
    edt_to_walls /= (edt_to_walls.max() + 1e-8)

    # ---- distance_to_end: projection onto tilted channel axis ----
    # Axis runs from midpoint of top end-line to midpoint of bottom end-line.
    top_mid_y = 0.5 * (y_l_top + y_r_top)
    top_mid_x = 0.5 * (x_l_top + x_r_top)
    bot_mid_y = 0.5 * (y_l_bot + y_r_bot)
    bot_mid_x = 0.5 * (x_l_bot + x_r_bot)

    axis_dy = bot_mid_y - top_mid_y
    axis_dx = bot_mid_x - top_mid_x
    axis_len = np.hypot(axis_dy, axis_dx)
    uy = axis_dy / axis_len
    ux = axis_dx / axis_len

    # Signed projection from top midpoint, normalized by axis length: 0 at top, 1 at bottom.
    along = ((y_grid - top_mid_y) * uy + (x_grid - top_mid_x) * ux) / axis_len
    distance_to_end = np.clip(along, 0.0, 1.0) * (final_mask > 0)

    return final_mask, distance_to_end, edt_to_walls
    
if __name__ == "__main__":
    metadata = skimage.io.imread("Pedi-LC-2330_2.tif")
    edge_mask = preprocess_raw_image_fast(metadata)
    tifffile.imwrite("edge_mask.tif", edge_mask)
    first_mask = analyze_growth(edge_mask, True)
    rect_mask, edt_end, edt_wall = create_rectangular_mask(first_mask)
    tifffile.imwrite("rectangular_mask.tif", rect_mask)
    tifffile.imwrite("edt_end.tif", edt_end)
    tifffile.imwrite("edt_wall.tif", edt_wall)