# Wall Follow Node

**Fastest Lap Time: 14.9s**

https://youtu.be/lVyHqh_UCHI









Reactive LiDAR controller for the AutoDRIVE RoboRacer (`wall_follow_node.py`). It keeps the car centered between the track walls, detects corners with a forward cone, and plans speed with ReLU terms tracked by a PI speed loop.

## Features

### 1. Two-beam wall measurement (`get_wall_distance`)
Each side fires two beams: one at 90° (straight at the wall) and one at 90° − 50°. These give the wall's angle relative to the car (`alpha`), and the distance is projected forward by the `lookahead` (0.5 m) so the controller reacts to where the car will be. If the beams read beyond `wall_max_range` (2.5 m), that wall is reported as missing (an opening). This runs on both the left and right walls.

### 2. Track centering
- **Both walls visible:** error = (left − right) / 2, the offset from the center line. This also updates a smoothed track-width estimate (90% old, 10% new).
- **One wall visible:** hold half the track width from the visible wall.
- **No walls visible:** error = 0; gap and corner steering take over.

### 3. PID steering
- **P** (`kp`): steers back toward center in proportion to the error.
- **I** (`ki_steer`): removes a persistent offset. It is capped at ±`i_max`, only accumulates on straights with both walls visible and unsaturated steering, and decays 10% per scan otherwise so it doesn't fight turns.
- **D** (`kd`): damps oscillation. It is skipped on the scan where the set of visible walls changes, since the error jumps there.

### 4. Corner detection with a cone
The closest reading in a forward cone (±`cone_angle_deg`) gives `d_front`. When `d_front < d_corner`, a corner is detected. Two diagonal cones at ±`side_cone_center_deg` (±`side_cone_width_deg` wide) are compared, and the car turns toward the more open side with strength `k_corner × relu(d_corner − d_front)`, so it turns harder as the wall gets closer.

### 5. Gap steering (`get_gap_angle`)
Within ±`gap_fov_deg` ahead, ranges are smoothed with a 9-beam moving average, and the widest run of beams within 90% of the deepest reading is found. `k_gap` × the angle to its center is added to the steering, a constant gentle pull toward open space.

**Final steering** = P + I + D + gap + corner, clamped to ±1.

### 6. ReLU speed target
```
v_target = v_max − k_front · relu(d_brake − d_front) − k_steer · relu(|steering| − steer_deadband)
```
The car runs at `v_max` on clear straights, slows down as the wall ahead comes inside `d_brake` or as steering exceeds the deadband, and is clamped to `[v_min, v_max]`.

### 7. PI speed controller
Measured speed comes from odometry. Throttle = `kv` · (target − actual) + `ki` · ∫error, clamped to `[throttle_min, throttle_max]`. The integral only accumulates while the throttle is not saturated (anti-windup).

## Parameters (defaults)

| Parameter | Default | Description |
|---|---|---|
| `desired_distance` | 0.8 | Distance to a lone wall before the track width is measured [m] |
| `kp` | 1.0 | Steering proportional gain |
| `ki_steer` | 0.05 | Steering integral gain |
| `kd` | 0.1 | Steering derivative gain |
| `i_max` | 1.0 | Steering integral clamp [m·s] |
| `lookahead` | 0.5 | Wall-distance lookahead [m] |
| `wall_max_range` | 2.5 | Range beyond which a wall is treated as missing [m] |
| `d_corner` | **1.5** | Forward clearance that triggers corner steering [m] (tuned) |
| `k_corner` | **0.7** | Corner steering per meter inside `d_corner` (tuned) |
| `side_cone_center_deg` | 45 | Diagonal cone direction [deg] |
| `side_cone_width_deg` | 15 | Diagonal cone half-width [deg] |
| `k_gap` | 0.3 | Steering per radian of gap angle |
| `gap_fov_deg` | 70 | Gap search half-width [deg] |
| `cone_angle_deg` | 8 | Forward cone half-width [deg] |
| `v_max` | 2.5 | Target speed on straights [m/s] |
| `v_min` | 1.0 | Minimum target speed [m/s] |
| `d_brake` | 3.0 | Forward clearance where slowing starts [m] |
| `k_front` | 0.6 | Speed reduction per meter inside `d_brake` [(m/s)/m] |
| `k_steer` | 2.0 | Speed reduction per unit steering beyond the deadband [m/s] |
| `steer_deadband` | 0.3 | Steering magnitude ignored by the speed planner |
| `kv` | 0.5 | Speed controller proportional gain |
| `ki` | 0.2 | Speed controller integral gain |
| `throttle_min` | 0.0 | Lowest throttle command |
| `throttle_max` | 1.0 | Highest throttle command |
