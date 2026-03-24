import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt
import qrcode

#========Constants========


# Speed of light in metres per second — used to convert clock bias (seconds) into metres
C = 299_792_458.0

# Earth's gravitational parameter (GM): used to compute satellite orbital speed
MU = 3.986004418e14

# Earth's rotation rate in radians per second — used to rotate from ECI to ECEF frame
OMEGA_E = 7.2921159e-5

# OSGB36 ellipsoid semi-major axis in metres (roughly Earth's equatorial radius)
OSGB36_A = 6377565.398              

# OSGB36 flattening factor: describes how much the Earth is squashed at the poles
OSGB36_F = 1 / 299.3249646

# Used in geodetic-to-ECEF conversions to account for Earth's non-spherical shape
OSGB36_E2 = OSGB36_F * (2 - OSGB36_F)   #WGS84_E2 = WGS84_F * (2 - WGS84_F)

# Altitude of GPS satellite orbits above Earth's surface (m)
GPS_ALTITUDE = 20_200_000.0

# Total orbital radius
A_ORBIT = OSGB36_A + GPS_ALTITUDE

# Orbital inclination of satellites (degrees to radians)
INC = np.deg2rad(61.0)

# Mean speed in rad per second, derived from keplers third law
N_MEAN = np.sqrt(MU / A_ORBIT**3)

# Relativistic clock drift rate for satellites (general - special relativity effects (45-7=38 per day)) (s/s)
REL_BASE = 38e-6 / 86400.0


#========Utility Functions========


# Returns a 3x3 rotation matrix about the x-axis, ie only y and z coordinates change
def rot1(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s], 
        [0, s,  c]
    ])

# Returns a 3x3 rotation matrix about the z-axis, ie x and y change
def rot3(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, -s, 0], 
        [ s,  c, 0],
        [ 0,  0, 1]  
    ])

# Converts geodetic coordinates (latitude, longitude, height) to (ECEF) cartesian coordinates
def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # N is the radius of curvature — accounts for Earth's ellipsoidal shape
    N = OSGB36_A / np.sqrt(1 - OSGB36_E2 * sin_lat**2)

    # Compute ECEF cartisean coordinates
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1 - OSGB36_E2) + h_m) * sin_lat

    return np.array([x, y, z])

# Rotation matrix to convert ECEF into East-North-Up frame
def ecef_to_enu_matrix(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    return np.array([
        [-sin_lon,            cos_lon,           0        ],  # East direction
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat      ],  # North direction
        [ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat      ]   # Up direction
    ])

# Position error vector at at the receiver's location (lat, lon) by converting ECEF to ENU
def ecef_error_to_enu(err_ecef, lat_deg, lon_deg):
    R = ecef_to_enu_matrix(lat_deg, lon_deg)
    return R @ err_ecef

 # Computes the elevation angle of satellite from receiver
def elevation_angle_deg(sat_ecef, recv_ecef, lat_deg, lon_deg):

    # Line-of-sight vector from receiver to satellite in ECEF
    los = sat_ecef - recv_ecef

    # los vector ecef to enu
    enu = ecef_error_to_enu(los, lat_deg, lon_deg)

    # Unpack components
    east, north, up = enu

    # Horizontal distance from receiver to satellite
    horiz = np.sqrt(east**2 + north**2)

    # Elevation angle = arctan of vertical over horizontal distance
    return np.rad2deg(np.arctan2(up, horiz))


#========Satellite Constellation========


def build_constellation():
    """
    20 satellites, 5 planes, 4 sats per plane.

    """

    # Right Ascension of Ascending Node (RAAN) for each of the 5 orbital planes 
    raans_deg = [0, 72, 144, 216, 288]

    # Offset for each plane to stagger satellites between planes so they line up in orbit
    plane_offsets_deg = [0, 18, 36, 54, 72]

    # List to collect satellite dictionaries
    sats = []

    # Running satellite ID counter
    sat_id = 0

    # Creates small satellite variations in relativistic drift rate with range: ±0.2% of rel_base
    differential = np.linspace(-0.002, 0.002, 20)

    # Loop over each orbital plane and its starting offset
    for p, (raan_deg, offset_deg) in enumerate(zip(raans_deg, plane_offsets_deg)):
        # Place 4 satellites evenly spaced (90° apart) within the plane
        for k in range(4):
            # Initial argument of latitude for this satellite (degrees), wrapped to 360°
            u0_deg = offset_deg + 90 * k

            # Satellite-specific relativistic drift rate, the differential means receiver clock bias does not absorb relativistic errors
            alpha_rel = REL_BASE * (1.0 + differential[sat_id])

            # Store all orbital and clock parameters for this satellite
            sats.append({
                "id": sat_id,                          
                "raan": np.deg2rad(raan_deg),          
                "u0": np.deg2rad(u0_deg % 360),        
                "inc": INC,                           
                "a": A_ORBIT,                       
                "alpha_rel": alpha_rel,               
                "alpha_basic": 0.0                  
            })
            sat_id += 1

    # Return the list of 12 satellite dictionaries
    return sats


#========Orbit Propagation========


def satellite_position_eci(sat, t):
    """
    Circular orbit in ECI.
    u(t) = u0 + n t
    """
    # Current latitude: starting angle plus how far it has moved in time t
    u = sat["u0"] + N_MEAN * t

    # Position in the orbital plane (2D)
    r_orb = np.array([
        sat["a"] * np.cos(u),  # x component in orbital plane
        sat["a"] * np.sin(u),  # y component in orbital plane
        0.0                    # z = 0
    ])

    # Rotate into ECI frame, first inclination (rot1), then rotate to the correct orbital plane (rot3 with RAAN)
    return rot3(sat["raan"]) @ rot1(sat["inc"]) @ r_orb

# Converts a position from the ECI to ECEF
def eci_to_ecef(r_eci, t):
    # Undo Earth rotation
    theta = OMEGA_E * t         
    return rot3(-theta) @ r_eci  

def satellite_position_ecef(sat, t):
    return eci_to_ecef(satellite_position_eci(sat, t), t)


#========Clock========


def satellite_clock_relative_correction(sat, t):

    # Relativistic clock drift component — accumulates over time
    b_rel = sat["alpha_rel"] * t

    # Total clock bias is the sum of both components
    return b_rel


#========Measurement========

def raw_pseudorange(sat_ecef, recv_true_ecef, sat_clock_bias_sec, noise_std=0.0):
    """
    Raw pseudorange:
      P_raw = geometric_range - c * satellite_clock_bias + noise
    Receiver true clock is set to zero here.
    """
    # Geometric (true) range between satellite and receiver in metres
    rho = np.linalg.norm(sat_ecef - recv_true_ecef)

    # Optional Gaussian measurement noise (simulates atmospheric/multipath errors)
    noise = np.random.normal(0.0, noise_std)

    # Pseudorange = true range minus the clock-bias-induced range error, plus noise
    # Satellite clock bias is subtracted because a fast satellite clock makes signals
    # appear to arrive earlier, shortening the apparent range
    return rho - C * sat_clock_bias_sec + noise

def visible_satellites(sats, t, recv_ecef, lat_deg, lon_deg, elev_mask_deg=10.0):
    # Returns a list of satellites that are above the elevation mask angle
    # (i.e., high enough above the horizon to be usable)
    visible = []
    for sat in sats:
        # Compute satellite ECEF position at time t
        sat_ecef = satellite_position_ecef(sat, t)
        # Compute elevation angle from receiver to satellite
        elev = elevation_angle_deg(sat_ecef, recv_ecef, lat_deg, lon_deg)
        # Only include satellites above the mask angle (default 10°)
        if elev > elev_mask_deg:
            visible.append((sat, sat_ecef, elev))  # Store satellite, its position, and elevation
    return visible

#========Solve Position========

def solve_receiver_position(sat_positions_ecef, pseudoranges, x0, max_iter=10, tol=1e-4):
    """
    Solve for [x, y, z, b] where b is receiver clock bias in meters.
    """
    # Start with the initial guess [x, y, z, clock_bias_in_metres]
    x = x0.copy()

    # Iterative least-squares solver (linearised Newton-style)
    for _ in range(max_iter):
        H = []          # Design matrix (partial derivatives of pseudorange w.r.t. unknowns)
        residuals = []  # Difference between measured and predicted pseudoranges

        # Current estimated position (first 3 elements)
        recv_est = x[:3]
        # Current estimated receiver clock bias in metres
        b_est = x[3]

        for sat_pos, P in zip(sat_positions_ecef, pseudoranges):
            # Vector from estimated receiver position to satellite
            diff = sat_pos - recv_est
            # Estimated geometric range
            r = np.linalg.norm(diff)

            # Predicted pseudorange including clock bias
            P_hat = r + b_est

            # Residual: how far off our prediction is from the measurement
            v = P - P_hat
            residuals.append(v)

            # Unit vector from receiver to satellite (line-of-sight direction)
            los = diff / r

            # Row of the design matrix H:
            # Partial derivatives of P_hat w.r.t. [x, y, z, b]
            # Negative sign on position partials because increasing position moves away from sat
            H.append([
                -los[0],  # d(P_hat)/dx
                -los[1],  # d(P_hat)/dy
                -los[2],  # d(P_hat)/dz
                1.0       # d(P_hat)/db = 1
            ])

        # Convert lists to numpy arrays
        H = np.array(H)
        residuals = np.array(residuals)

        # Solve H * dx = residuals in least-squares sense to find the correction step
        dx, *_ = np.linalg.lstsq(H, residuals, rcond=None)

        # Apply the correction to current estimate
        x += dx

        # Check convergence: stop if position and clock corrections are both tiny
        if np.linalg.norm(dx[:3]) < tol and abs(dx[3]) < tol:
            break

    # Return the solved [x, y, z, clock_bias_in_metres] vector
    return x


#========RUN========


def run_simulation():
    # Fix the random seed so noise is reproducible across runs
    np.random.seed(4)

    # True receiver location: Manchester, UK
    lat_deg = 53.4670   # Latitude in degrees North
    lon_deg = -2.2305   # Longitude in degrees (negative = West)
    h_m = 70.0          # Approximate elevation above ellipsoid in metres

    # Convert true receiver geodetic position to ECEF coordinates
    recv_true = geodetic_to_ecef(lat_deg, lon_deg, h_m)

    # Build the 12-satellite constellation
    sats = build_constellation()

    # Define the simulation time grid
    dt = 60.0                   # Time step: 1 minute in seconds
    t_end = 100 * 3600.0        # Total simulation duration: 100 hours in seconds
    times = np.arange(0.0, t_end + dt, dt)  # Array of time steps from 0 to t_end

    # Initial guess for the least-squares solver:
    # Perturb the true position slightly to simulate an imperfect starting estimate
    x0_corr = np.array([recv_true[0] + 100.0, recv_true[1] - 80.0, recv_true[2] + 50.0, 0.0])
    # Uncorrected solver uses the same starting guess
    x0_unc  = x0_corr.copy()

    # Storage lists for results at each time step
    enu_corr_list = []       # ENU position errors with full relativistic correction
    enu_unc_list = []        # ENU position errors without relativistic correction
    clock_corr_list = []     # Estimated receiver clock bias (corrected), in seconds
    clock_unc_list = []      # Estimated receiver clock bias (uncorrected), in seconds
    num_visible_list = []    # Number of visible satellites at each time step

    # Main simulation loop — iterate over every time step
    for t in times:
        # Get list of satellites above the elevation mask at this time
        vis = visible_satellites(sats, t, recv_true, lat_deg, lon_deg, elev_mask_deg=10.0)

        # Need at least 4 satellites to solve for 4 unknowns (x, y, z, clock)
        if len(vis) < 4:
            # Append NaN placeholders and skip this time step
            enu_corr_list.append([np.nan, np.nan, np.nan])
            enu_unc_list.append([np.nan, np.nan, np.nan])
            clock_corr_list.append(np.nan)
            clock_unc_list.append(np.nan)
            num_visible_list.append(len(vis))
            continue

        # Collect satellite positions and pseudoranges for visible satellites
        sat_positions = []  # ECEF positions of visible satellites
        pr_corr = []        # Pseudoranges corrected for full satellite clock bias (incl. relativity)
        pr_unc = []         # Pseudoranges corrected only for basic clock bias (no relativity)

        for sat, sat_ecef, elev in vis:
            # Full satellite clock bias including relativistic drift
            b_full = satellite_clock_relative_correction(sat, t)

            # Generate the raw pseudorange measurement (simulates what the satellite transmits)
            # Uses the full clock bias as the "truth" and adds optional noise
            P_raw = raw_pseudorange(
                sat_ecef=sat_ecef,
                recv_true_ecef=recv_true,
                sat_clock_bias_sec=b_full,
                noise_std=0.5  # 0.5 m standard deviation noise
            )

            # Corrected pseudorange: receiver applies the full clock bias to remove it
            P_used_corr = P_raw + C * b_full

            # Uncorrected pseudorange: receiver only removes the basic (zero) bias,
            # leaving the relativistic component uncorrected — this causes position drift
            P_used_unc = P_raw

            # Accumulate satellite data
            sat_positions.append(sat_ecef)
            pr_corr.append(P_used_corr)
            pr_unc.append(P_used_unc)

        # Convert lists to numpy arrays for matrix operations
        sat_positions = np.array(sat_positions)
        pr_corr = np.array(pr_corr)
        pr_unc = np.array(pr_unc)

        # Solve for receiver position using corrected pseudoranges
        # Use the previous solution as the warm-start initial guess
        x_corr = solve_receiver_position(sat_positions, pr_corr, x0_corr)
        x0_corr = x_corr  # Update warm-start for next iteration

        # Solve for receiver position using uncorrected pseudoranges
        x_unc = solve_receiver_position(sat_positions, pr_unc, x0_unc)
        x0_unc = x_unc  # Update warm-start for next iteration

        # Compute ECEF position errors (estimated minus true position)
        err_corr_ecef = x_corr[:3] - recv_true
        err_unc_ecef = x_unc[:3] - recv_true

        # Rotate errors from ECEF into local ENU frame for easier interpretation
        err_corr_enu = ecef_error_to_enu(err_corr_ecef, lat_deg, lon_deg)
        err_unc_enu = ecef_error_to_enu(err_unc_ecef, lat_deg, lon_deg)

        # Store ENU errors
        enu_corr_list.append(err_corr_enu)
        enu_unc_list.append(err_unc_enu)

        # Store estimated receiver clock bias, converted from metres to seconds
        clock_corr_list.append(x_corr[3] / C)
        clock_unc_list.append(x_unc[3] / C)

        # Store the count of visible satellites at this time step
        num_visible_list.append(len(vis))

    # Return all results as a dictionary of numpy arrays
    return {
        "times":       times,
        "enu_corr":    np.array(enu_corr_list),   # (N, 3): [East, North, Up] errors, corrected
        "enu_unc":     np.array(enu_unc_list),    # (N, 3): [East, North, Up] errors, uncorrected
        "clock_corr":  np.array(clock_corr_list), # (N,): receiver clock bias, corrected [s]
        "clock_unc":   np.array(clock_unc_list),  # (N,): receiver clock bias, uncorrected [s]
        "num_visible": np.array(num_visible_list),# (N,): number of visible satellites
    }


#========Plots========


def plot_results(results):
    # Convert simulation times from seconds to hours
    t_hours = results["times"] / 3600.0

    # Unpack ENU error arrays and clock bias arrays from results
    enu_corr = results["enu_corr"]
    enu_unc = results["enu_unc"]
    clock_corr = results["clock_corr"]
    clock_unc = results["clock_unc"]

    # East vs North, corrected
    plt.figure(figsize=(7, 6))
    plt.plot(enu_corr[:, 0], enu_corr[:, 1], label="Corrected")
    plt.xlabel("East error [m]")
    plt.ylabel("North error [m]")
    plt.title("Estimated receiver track in local EN plane")
    plt.legend()
    plt.axis("equal")  
    plt.grid(True)

    # --- Plot 4: Estimated receiver clock bias over time ---
    plt.figure(figsize=(8, 5))
    plt.plot(t_hours, clock_corr * 1e6, label="Corrected")  
    plt.plot(t_hours, clock_unc  * 1e6, label="Uncorrected")
    plt.xlabel("Time [hours]")
    plt.ylabel("Receiver clock bias [microseconds]")
    plt.title("Estimated receiver clock bias vs time")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(7, 6))
    plt.plot(enu_unc[:, 0],  enu_unc[:, 1],  label="Uncorrected")
    plt.xlabel("East error [m]")
    plt.ylabel("North error [m]")
    plt.title("Estimated receiver track in local EN plane")
    plt.legend()
    plt.axis("equal")  
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_on_map_animated(results, lat_deg=53.4670, lon_deg=-2.2305):
    
    enu_corr = results["enu_corr"]
    enu_unc  = results["enu_unc"]
    times    = results["times"]
    
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_deg))

    def enu_to_latlon(enu_errors):
        delta_lat = enu_errors[:, 1] / metres_per_deg_lat
        delta_lon = enu_errors[:, 0] / metres_per_deg_lon
        return lat_deg + delta_lat, lon_deg + delta_lon

    lats_corr, lons_corr = enu_to_latlon(enu_corr)
    lats_unc,  lons_unc  = enu_to_latlon(enu_unc)

    valid = ~np.isnan(lats_corr) & ~np.isnan(lats_unc)

    # Print to verify full 100 hours is covered
    valid_times = times[valid]
    print(f"First timestep: {valid_times[0]/3600:.2f} hours")
    print(f"Last timestep:  {valid_times[-1]/3600:.2f} hours")
    print(f"Total valid points: {valid.sum()}")

    lats_corr = lats_corr[valid]
    lons_corr = lons_corr[valid]
    lats_unc  = lats_unc[valid]
    lons_unc  = lons_unc[valid]

    # Use ALL points — no downsampling
    # Animation speed auto-adjusts to fit 10 seconds
    coords_corr_js = [[float(lat), float(lon)] for lat, lon in zip(lats_corr, lons_corr)]
    coords_unc_js  = [[float(lat), float(lon)] for lat, lon in zip(lats_unc,  lons_unc)]

    print(f"Total frames in animation: {len(coords_corr_js)}")

    m = folium.Map(
        location=[lat_deg, lon_deg],
        zoom_start=19,
        tiles="CartoDB positron"
    )

    animation_js = f"""
    <script>
    var coordsCorr  = {coords_corr_js};
    var coordsUnc   = {coords_unc_js};
    var totalFrames  = coordsCorr.length;
    var duration     = 10000;        // 10 seconds total
    var frameInterval = duration / totalFrames;
    var currentFrame = 0;
    var animationTimer = null;
    var polylineCorr = null;
    var polylineUnc  = null;
    var drawnCorr = [];
    var drawnUnc  = [];
    var isPlaying = false;

    document.addEventListener('DOMContentLoaded', function() {{
        var mapObj = Object.values(window).find(v => v instanceof L.Map);
        if (!mapObj) return;

        polylineUnc = L.polyline([], {{
            color: '#D85A30',
            weight: 1,
            opacity: 0.5
        }}).addTo(mapObj);
        
        polylineCorr = L.polyline([], {{
            color: '#1D9E75',
            weight: 1,
            opacity: 1.0
        }}).addTo(mapObj);

        // Legend
        var legend = L.control({{position: 'topright'}});
        legend.onAdd = function() {{
            var div = L.DomUtil.create('div');
            div.innerHTML =
                '<div style="background:white;padding:8px;border-radius:4px;font-size:12px;line-height:1.8;">' +
                '<span style="color:#1D9E75;font-weight:bold;">— </span>Corrected<br>' +
                '<span style="color:#D85A30;font-weight:bold;">— </span>Uncorrected' +
                '</div>';
            return div;
        }};
        legend.addTo(mapObj);

        // Play button
        var btn = L.control({{position: 'bottomright'}});
        btn.onAdd = function() {{
            var div = L.DomUtil.create('div');
            div.innerHTML = '<button id="playBtn" style="' +
                'padding:8px 16px;font-size:14px;background:#333;' +
                'color:white;border:none;border-radius:4px;cursor:pointer;">' +
                '▶ Play</button>';
            return div;
        }};
        btn.addTo(mapObj);

        document.getElementById('playBtn').addEventListener('click', function() {{
            if (isPlaying) return;

            // Reset
            currentFrame = 0;
            drawnCorr = [];
            drawnUnc  = [];
            polylineCorr.setLatLngs([]);
            polylineUnc.setLatLngs([]);
            isPlaying = true;
            this.textContent = '⏸ Playing...';
            var btnRef = this;

            animationTimer = setInterval(function() {{
                if (currentFrame >= totalFrames) {{
                    clearInterval(animationTimer);
                    isPlaying = false;
                    btnRef.textContent = '▶ Play';
                    return;
                }}

                // Add points in batches to keep 10 second duration
                // regardless of total frame count
                drawnCorr.push(coordsCorr[currentFrame]);
                drawnUnc.push(coordsUnc[currentFrame]);
                polylineCorr.setLatLngs(drawnCorr);
                polylineUnc.setLatLngs(drawnUnc);
                currentFrame++;
            }}, frameInterval);
        }});
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(animation_js))
    
    # Exact conversion factors for Manchester's latitude
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_deg))

    # Convert desired metre offsets to exact degree offsets
    m_offsets = [-500, -450, -400, -350, -300, -250, -200, -150, -100, -75, -50, -25, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    lat_offsets = [d / metres_per_deg_lat for d in m_offsets]  # North-South
    lon_offsets = [d / metres_per_deg_lon for d in m_offsets]  # East-West

    labels = ['-500m', '-450m', '-400m', '-350m', '-300m', '-250m', '-200m', '-150m', '-100m', '-75m', '-50m' ,'-25m', 
              '25m', '50m' ,'75m', '+100m', '150m', '+200m', '250m',  '+300m', '350m',  '+400m', '450m', '+500m']

    # Tighter label offset (closer to axis line)
    label_nudge_lat = 5 / metres_per_deg_lat   # 50m nudge south of EW axis
    label_nudge_lon = 5 / metres_per_deg_lon   # 80m nudge west of NS axis

    crosshair_js = f"""
    <script>
    function addAxes(map) {{
        L.polyline([
            [{lat_deg}, {lon_deg - 600/metres_per_deg_lon}],
            [{lat_deg}, {lon_deg + 600/metres_per_deg_lon}]
        ], {{ color: 'black', weight: 1.5, dashArray: '5,5', opacity: 0.7 
        }}).addTo(map).bindTooltip('East-West axis');

        L.polyline([
            [{lat_deg - 600/metres_per_deg_lat}, {lon_deg}],
            [{lat_deg + 600/metres_per_deg_lat}, {lon_deg}]
        ], {{ color: 'black', weight: 1.5, dashArray: '5,5', opacity: 0.7 
        }}).addTo(map).bindTooltip('North-South axis');

        var lon_offsets = {lon_offsets};
        var lat_offsets = {lat_offsets};
        var labels     = {labels};

        lon_offsets.forEach(function(offset, i) {{
            // East-West labels (sit just below the EW axis line)
            L.marker([{lat_deg} - {label_nudge_lat}, {lon_deg} + offset], {{
                icon: L.divIcon({{
                    html: '<div style="font-size:9px;color:black;white-space:nowrap;">' + labels[i] + '</div>',
                    className: ''
                }})
            }}).addTo(map);
        }});

        lat_offsets.forEach(function(offset, i) {{
            // North-South labels (sit just left of the NS axis line)
            L.marker([{lat_deg} + offset, {lon_deg} - {label_nudge_lon}], {{
                icon: L.divIcon({{
                    html: '<div style="font-size:9px;color:black;white-space:nowrap;">' + labels[i] + '</div>',
                    className: ''
                }})
            }}).addTo(map);
        }});
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        var mapObj = Object.values(window).find(v => v instanceof L.Map);
        if (mapObj) addAxes(mapObj);
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(crosshair_js))

    output_path = "gps_simulation_map_animated.html"
    m.save(output_path)
    print(f"Animated map saved to {output_path}")
    return m


# Your live GitHub Pages URL
url = "https://joejackmoore-star.github.io/GPS_MAP/gps_simulation_map.html"

# Generate QR code
qr = qrcode.make(url)
qr.save("map_qr.png")
print("QR code saved as map_qr.png")

if __name__ == "__main__":
    results = run_simulation()
    plot_results(results)
    plot_on_map_animated(results)
