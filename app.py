import time
import json
import csv
import io
import urllib.request
from app import app as application
# import googlemap
from collections import defaultdict
import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# --- CONFIGURATION ---
MAX_ALLOWED_FAILS = 4
LOCKOUT_DURATION = 240  # Default block duration in seconds (2 minutes)

# --- IN-MEMORY DATA STORES ---
request_logs = defaultdict(list)
blocked_ips = {}  # ip -> {unblock_time, reason, block_type, timestamp, country, lat, lon, rep_score}
ip_history_logs = []  # List of all block events for history and CSV export

# --- ML MODEL INITIALIZATION ---
ml_model = IsolationForest(contamination=0.1, random_state=42)

def generate_synthetic_data():
    """Generates baseline traffic data for ML training."""
    np.random.seed(42)
    normal_traffic = np.column_stack([
        np.random.uniform(1.0, 5.0, 200),
        np.random.uniform(0.7, 1.0, 200)
    ])
    brute_force_traffic = np.column_stack([
        np.random.uniform(10.0, 50.0, 50),
        np.random.uniform(0.0, 0.1, 50)
    ])
    return np.vstack([normal_traffic, brute_force_traffic])

ml_model.fit(generate_synthetic_data())

# --- IP UNMASKING & GEOLOCATION ---
def get_real_ip(req):
    """Unmasks the true client IP address by inspecting HTTP proxy headers."""
    if req.headers.get('CF-Connecting-IP'):
        return req.headers.get('CF-Connecting-IP').strip()
    if req.headers.get('X-Real-IP'):
        return req.headers.get('X-Real-IP').strip()
    if req.headers.get('X-Forwarded-For'):
        return req.headers.get('X-Forwarded-For').split(',')[0].strip()
    return req.remote_addr

def get_ip_metadata(ip):
    """Resolves IP Geolocation coordinates and threat reputation score."""
    # Handle local loopback / private IPs with mock test locations
    if ip in ['127.0.0.1', 'localhost', '::1'] or ip.startswith(('10.', '192.168.', '172.')):
        locations = [
            {"country": "United States (Local)", "lat": 40.7128, "lon": -74.0060},
            {"country": "United Kingdom (Dev)", "lat": 51.5074, "lon": -0.1278},
            {"country": "Germany (Test)", "lat": 52.5200, "lon": 13.4050},
            {"country": "Japan (Proxy)", "lat": 35.6762, "lon": 139.6503}
        ]
        # Pick location deterministically based on IP hash
        loc = locations[sum(ord(c) for c in ip) % len(locations)]
        return {
            "country": loc["country"],
            "lat": loc["lat"],
            "lon": loc["lon"],
            "rep_score": 88  # High threat score for local flagged testing
        }

    try:
        url = f"http://ip-api.com/json/{ip}?fields=status,country,lat,lon"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode())
            if data.get('status') == 'success':
                return {
                    "country": data.get('country', 'Unknown'),
                    "lat": data.get('lat', 0.0),
                    "lon": data.get('lon', 0.0),
                    "rep_score": int(np.random.uniform(65, 99))  # AbuseIPDB Threat Confidence score simulation
                }
    except Exception:
        pass

    return {"country": "Unknown Origin", "lat": 20.0, "lon": 0.0, "rep_score": 75}

# --- HELPER FUNCTIONS ---
def get_ip_block_status(ip_address):
    """Checks if an IP is blocked and handles temporary block expiration."""
    now = time.time()
    if ip_address in blocked_ips:
        record = blocked_ips[ip_address]
        unblock_time = record['unblock_time']
        if now < unblock_time:
            return True, int(unblock_time - now), record
        else:
            # Expiration reached: Auto-unblock IP
            del blocked_ips[ip_address]
            request_logs[ip_address] = []
            
    return False, 0, None

def extract_features(ip_address):
    logs = request_logs[ip_address]
    now = time.time()
    recent_logs = [entry for entry in logs if now - entry['time'] <= 60]
    request_logs[ip_address] = recent_logs
    
    if len(recent_logs) < 3:
        return None

    time_span = max(now - recent_logs[0]['time'], 1.0)
    req_per_sec = len(recent_logs) / time_span
    success_rate = sum(1 for e in recent_logs if e['success']) / len(recent_logs)
    
    return [req_per_sec, success_rate]

def analyze_ip_behavior(ip_address):
    features = extract_features(ip_address)
    if not features:
        return False
    prediction = ml_model.predict([features])
    return prediction[0] == -1

def record_block_event(ip, reason, block_type, duration_sec=LOCKOUT_DURATION):
    now = time.time()
    meta = get_ip_metadata(ip)
    unblock_time = now + duration_sec
    
    block_data = {
        "ip": ip,
        "unblock_time": unblock_time,
        "reason": reason,
        "block_type": block_type,
        "timestamp": now,
        "time_str": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now)),
        "country": meta["country"],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "rep_score": meta["rep_score"]
    }
    
    blocked_ips[ip] = block_data
    ip_history_logs.insert(0, block_data)
    return block_data

# --- HTML & DASHBOARD TEMPLATE ---
FULL_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brute-force Detection System</title>
    
    <!-- Leaflet CSS & JS for Interactive Map -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <style>
        :root {
            --bg-color: #090d16;
            --card-bg: rgba(21, 30, 48, 0.75);
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
            --border: rgba(255, 255, 255, 0.08);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); padding: 1.5rem; min-height: 100vh; }

        .dashboard-header {
            display: flex; justify-content: space-between; align-items: center;
            padding-bottom: 1rem; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border);
        }
        .sys-status { display: flex; align-items: center; gap: 0.75rem; }
        .status-badge {
            display: inline-flex; align-items: center; gap: 0.5rem;
            padding: 0.35rem 0.85rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem;
        }
        .status-ok { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
        .status-alert { background: rgba(239, 68, 68, 0.15); color: var(--danger); border: 1px solid var(--danger); }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }

        .grid-layout { display: grid; grid-template-columns: 340px 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }

        .card {
            background: var(--card-bg); backdrop-filter: blur(12px);
            border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem;
        }

        /* Form Controls */
        .form-group { margin-bottom: 0.85rem; }
        label { display: block; font-size: 0.8rem; color: var(--text-sub); margin-bottom: 0.3rem; }
        input[type="text"], input[type="password"], select {
            width: 100%; padding: 0.65rem 0.85rem; border-radius: 8px;
            border: 1px solid var(--border); background: rgba(10, 15, 26, 0.7);
            color: #fff; font-size: 0.9rem; outline: none;
        }
        button {
            padding: 0.65rem 1rem; border-radius: 8px; border: none; font-weight: 600;
            cursor: pointer; transition: 0.2s; font-size: 0.85rem;
        }
        .btn-primary { background: var(--accent); color: white; width: 100%; }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-danger { background: var(--danger); color: white; }
        .btn-success { background: var(--success); color: white; }

        /* Map Styling */
        #map { height: 320px; width: 100%; border-radius: 10px; border: 1px solid var(--border); }

        /* Toolbar & Controls */
        .table-toolbar {
            display: flex; justify-content: space-between; align-items: center;
            gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;
        }
        .filters { display: flex; gap: 0.75rem; flex: 1; max-width: 600px; }
        .bulk-actions { display: flex; gap: 0.5rem; }

        /* Data Tables */
        .table-container { overflow-x: auto; margin-top: 0.5rem; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left; }
        th, td { padding: 0.85rem 1rem; border-bottom: 1px solid var(--border); }
        th { color: var(--text-sub); font-weight: 600; background: rgba(255, 255, 255, 0.02); }

        .tag {
            padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; display: inline-block;
        }
        .tag-hardware { background: rgba(99, 102, 241, 0.2); color: #818cf8; }
        .tag-software { background: rgba(245, 158, 11, 0.2); color: var(--warning); }
        .tag-ml { background: rgba(236, 72, 153, 0.2); color: #f472b6; }

        .rep-score { font-weight: 700; border-radius: 4px; padding: 2px 6px; }
        .rep-high { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .rep-mid { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }

        #alert-box { display: none; padding: 0.65rem; border-radius: 8px; margin-top: 0.75rem; font-size: 0.85rem; text-align: center; }

        @media (max-width: 900px) {
            .grid-layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <!-- SYSTEM HEADER & HEALTH -->
    <div class="dashboard-header">
        <div>
            <h2>Brute-force Detection System</h2>
            <p style="font-size: 0.85rem; color: var(--text-sub);">Real-time Threat Monitoring & Dynamic Blocklist</p>
        </div>
        <div class="sys-status">
            <span>System Health:</span>
            <div id="healthBadge" class="status-badge status-ok">
                <span class="dot"></span> <span id="healthText">NORMAL</span>
            </div>
        </div>
    </div>

    <!-- MAIN DASHBOARD GRID -->
    <div class="grid-layout">
        
        <!-- LEFT PANEL: AUTH TESTER & QUICK IP LOCKOUT -->
        <div class="card">
            <h3 style="font-size: 1rem; margin-bottom: 1rem;">Authentication Portal</h3>
            <form id="loginForm">
                <div class="form-group">
                    <label>Username</label>
                    <input type="text" id="username" name="username" placeholder="admin" required autocomplete="off">
                </div>
                <div class="form-group">
                    <label>Password</label>
                    <input type="password" id="password" name="password" placeholder="••••••••" required>
                </div>
                <button type="submit" class="btn-primary">Sign In</button>
            </form>
            <div id="alert-box"></div>

            <hr style="border: 0; border-top: 1px solid var(--border); margin: 1.25rem 0;">

            <h4 style="font-size: 0.9rem; margin-bottom: 0.75rem;">Instant Manual IP Override</h4>
            <div class="form-group">
                <input type="text" id="manualIpInput" placeholder="Target IP Address">
            </div>
            <div class="form-group">
                <select id="manualReasonSelect">
                    <option value="Manual Admin Block">Manual Admin Block</option>
                    <option value="DoS Suspect">DoS Suspect</option>
                    <option value="Known Malicious Proxy">Known Malicious Proxy</option>
                </select>
            </div>
            <div style="display: flex; gap: 0.5rem;">
                <button onclick="instantBlock()" class="btn-danger" style="flex: 1;">Block IP</button>
                <button onclick="instantUnblock()" class="btn-success" style="flex: 1;">Unblock IP</button>
            </div>
        </div>

        <!-- RIGHT PANEL: GEOGRAPHIC ATTACK MAP -->
        <div class="card">
            <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">Interactive Attack Origin Map</h3>
            <div id="map"></div>
        </div>
    </div>

    <!-- LOWER SECTION: HISTORICAL & LIVE THREAT LIST -->
    <div class="card">
        <div class="table-toolbar">
            <h3>Live & Historical Threat Log</h3>
            
            <div class="filters">
                <input type="text" id="ipSearch" placeholder="Search IP Address..." onkeyup="filterTable()">
                <select id="reasonFilter" onchange="filterTable()">
                    <option value="">All Block Reasons</option>
                    <option value="Brute Force">Brute Force</option>
                    <option value="ML Threat Model Anomaly">ML Anomaly</option>
                    <option value="Manual Admin Block">Manual Block</option>
                </select>
            </div>

            <div class="bulk-actions">
                <button onclick="exportCSV()" style="background: #334155; color: #fff;">Export CSV</button>
                <button onclick="document.getElementById('csvInput').click()" style="background: #334155; color: #fff;">Import CSV</button>
                <input type="file" id="csvInput" style="display: none;" accept=".csv" onchange="importCSV(this)">
            </div>
        </div>

        <div class="table-container">
            <table id="threatTable">
                <thead>
                    <tr>
                        <th>IP Address</th>
                        <th>Country / Region</th>
                        <th>Timestamp</th>
                        <th>Reason</th>
                        <th>Block Type</th>
                        <th>Reputation Score</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="threatTableBody">
                    <!-- Populated via JS -->
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let map, mapMarkers = [];

        // Initialize Leaflet Map
        function initMap() {
            map = L.map('map').setView([20, 0], 2);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap & CartoDB',
                maxZoom: 18
            }).addTo(map);
        }

        async function fetchDashboardData() {
            try {
                const res = await fetch('/api/history');
                const data = await res.json();
                
                updateTable(data.history);
                updateMapMarkers(data.history);
                updateHealthStatus(data.active_blocks_count);
            } catch (err) {
                console.error("Failed to load dashboard telemetry:", err);
            }
        }

        function updateHealthStatus(activeBlocks) {
            const badge = document.getElementById('healthBadge');
            const text = document.getElementById('healthText');
            if (activeBlocks > 0) {
                badge.className = 'status-badge status-alert';
                text.innerText = `ELEVATED THREAT (${activeBlocks} BLOCKED)`;
            } else {
                badge.className = 'status-badge status-ok';
                text.innerText = 'NORMAL';
            }
        }

        function updateMapMarkers(records) {
            // Clear existing markers
            mapMarkers.forEach(m => map.removeLayer(m));
            mapMarkers = [];

            records.forEach(item => {
                if (item.lat && item.lon) {
                    const marker = L.circleMarker([item.lat, item.lon], {
                        color: item.is_active ? '#ef4444' : '#64748b',
                        radius: 7,
                        fillOpacity: 0.8
                    }).addTo(map);

                    marker.bindPopup(`
                        <strong>IP:</strong> ${item.ip}<br>
                        <strong>Country:</strong> ${item.country}<br>
                        <strong>Reason:</strong> ${item.reason}<br>
                        <strong>Rep Score:</strong> ${item.rep_score}%
                    `);
                    mapMarkers.push(marker);
                }
            });
        }

        function updateTable(records) {
            const tbody = document.getElementById('threatTableBody');
            tbody.innerHTML = '';

            records.forEach(row => {
                const tr = document.createElement('tr');
                
                const repClass = row.rep_score > 80 ? 'rep-high' : 'rep-mid';
                const tagClass = row.block_type.includes('Hardware') ? 'tag-hardware' : 
                                (row.block_type.includes('ML') ? 'tag-ml' : 'tag-software');

                tr.innerHTML = `
                    <td><strong>${row.ip}</strong></td>
                    <td>${row.country}</td>
                    <td>${row.time_str}</td>
                    <td>${row.reason}</td>
                    <td><span class="tag ${tagClass}">${row.block_type}</span></td>
                    <td><span class="rep-score ${repClass}">${row.rep_score}% Confidence</span></td>
                    <td>${row.is_active ? '<span style="color:var(--danger); font-weight:600;">Blocked</span>' : '<span style="color:var(--text-sub);">Expired</span>'}</td>
                    <td>
                        ${row.is_active ? 
                          `<button class="btn-success" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="quickUnblock('${row.ip}')">Unblock</button>` : 
                          `<button class="btn-danger" style="padding:0.25rem 0.6rem; font-size:0.75rem;" onclick="quickBlock('${row.ip}')">Block</button>`}
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function instantBlock() {
            const ip = document.getElementById('manualIpInput').value.trim();
            const reason = document.getElementById('manualReasonSelect').value;
            if (!ip) return alert('Please enter a valid IP address');
            
            await fetch('/api/block', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ ip, reason, duration: 300, block_type: 'Software Firewall' })
            });
            document.getElementById('manualIpInput').value = '';
            fetchDashboardData();
        }

        async function instantUnblock() {
            const ip = document.getElementById('manualIpInput').value.trim();
            if (!ip) return alert('Please enter a valid IP address');
            
            await fetch('/api/unblock', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ ip })
            });
            document.getElementById('manualIpInput').value = '';
            fetchDashboardData();
        }

        function quickBlock(ip) {
            document.getElementById('manualIpInput').value = ip;
            instantBlock();
        }

        function quickUnblock(ip) {
            document.getElementById('manualIpInput').value = ip;
            instantUnblock();
        }

        function filterTable() {
            const search = document.getElementById('ipSearch').value.toLowerCase();
            const reason = document.getElementById('reasonFilter').value.toLowerCase();
            const rows = document.querySelectorAll('#threatTableBody tr');

            rows.forEach(row => {
                const ipText = row.children[0].innerText.toLowerCase();
                const reasonText = row.children[3].innerText.toLowerCase();

                const matchesSearch = ipText.includes(search);
                const matchesReason = !reason || reasonText.includes(reason);

                row.style.display = (matchesSearch && matchesReason) ? '' : 'none';
            });
        }

        function exportCSV() {
            window.location.href = '/api/export-csv';
        }

        async function importCSV(input) {
            const file = input.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            await fetch('/api/import-csv', { method: 'POST', body: formData });
            input.value = '';
            fetchDashboardData();
        }

        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const alertBox = document.getElementById('alert-box');
            const formData = new FormData(e.target);

            const res = await fetch('/login', { method: 'POST', body: formData });
            const result = await res.json();

            alertBox.style.display = 'block';
            alertBox.innerText = result.message;
            alertBox.style.background = res.status === 200 ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)';
            alertBox.style.color = res.status === 200 ? 'var(--success)' : 'var(--danger)';

            fetchDashboardData();
        });

        window.onload = () => {
            initMap();
            fetchDashboardData();
            setInterval(fetchDashboardData, 10000);
        };
    </script>
</body>
</html>
"""

# --- FLASK ENDPOINTS ---

@app.route('/')
def home():
    return render_template_string(FULL_DASHBOARD_TEMPLATE)

@app.route('/login', methods=['POST'])
def login():
    client_ip = get_real_ip(request)
    now = time.time()

    is_blocked, time_remaining, record = get_ip_block_status(client_ip)
    if is_blocked:
        mins, secs = divmod(time_remaining, 60)
        return jsonify({
            "status": "Blocked",
            "message": f"Access blocked. Try again in {mins}m {secs}s."
        }), 403

    username = request.form.get('username')
    password = request.form.get('password')

    if username == "admin" and password == "pass123":
        request_logs[client_ip] = []
        return jsonify({"status": "Success", "message": f"Welcome back, {username}!"}), 200

    request_logs[client_ip].append({'time': now, 'success': False})
    failed_attempts = sum(1 for entry in request_logs[client_ip] if not entry['success'])

    is_ml_anomaly = analyze_ip_behavior(client_ip)

    if failed_attempts > MAX_ALLOWED_FAILS:
        record_block_event(client_ip, "Brute Force Attack Exceeded Threshold", "Software Dynamic Rule")
        return jsonify({
            "status": "Blocked",
            "message": "Security Alert: Exceeded failed attempts threshold. IP blocked."
        }), 429
    elif is_ml_anomaly:
        record_block_event(client_ip, "ML Threat Model Anomaly Detected", "ML Dynamic Rule")
        return jsonify({
            "status": "Blocked",
            "message": "Security Alert: ML engine detected anomalous traffic behavior. IP blocked."
        }), 429

    attempts_left = MAX_ALLOWED_FAILS - failed_attempts + 1
    return jsonify({
        "status": "Failed",
        "message": f"Invalid credentials. {attempts_left} attempt(s) remaining."
    }), 401

@app.route('/api/history', methods=['GET'])
def get_history():
    now = time.time()
    formatted_history = []
    
    for item in ip_history_logs:
        active = (item['ip'] in blocked_ips) and (now < blocked_ips[item['ip']]['unblock_time'])
        formatted_history.append({
            **item,
            "is_active": active
        })

    active_blocks_count = sum(1 for ip in blocked_ips if now < blocked_ips[ip]['unblock_time'])

    return jsonify({
        "history": formatted_history,
        "active_blocks_count": active_blocks_count
    })

@app.route('/api/block', methods=['POST'])
def manual_block():
    data = request.get_json() or {}
    ip = data.get('ip')
    reason = data.get('reason', 'Manual Admin Block')
    duration = int(data.get('duration', LOCKOUT_DURATION))
    block_type = data.get('block_type', 'Hardware Firewall Rule')

    if ip:
        record_block_event(ip, reason, block_type, duration)
        return jsonify({"status": "Success", "message": f"IP {ip} blocked successfully."})
    return jsonify({"status": "Error", "message": "Missing IP parameter."}), 400

@app.route('/api/unblock', methods=['POST'])
def manual_unblock():
    data = request.get_json() or {}
    ip = data.get('ip')

    if ip in blocked_ips:
        del blocked_ips[ip]
        request_logs[ip] = []
        return jsonify({"status": "Success", "message": f"IP {ip} unblocked."})
    return jsonify({"status": "Error", "message": "IP not actively blocked."}), 404

@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['IP Address', 'Country', 'Timestamp', 'Reason', 'Block Type', 'Reputation Score'])

    for record in ip_history_logs:
        writer.writerow([
            record['ip'], record['country'], record['time_str'],
            record['reason'], record['block_type'], record['rep_score']
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=threat_blocklist.csv"}
    )

@app.route('/api/import-csv', methods=['POST'])
def import_csv():
    if 'file' not in request.files:
        return jsonify({"status": "Error", "message": "No file uploaded"}), 400
    
    file = request.files['file']
    stream = io.StringIO(file.stream.read().decode("UTF-8"), newline=None)
    csv_input = csv.reader(stream)
    
    header = next(csv_input, None)  # Skip CSV header
    imported_count = 0

    for row in csv_input:
        if row and len(row) >= 1:
            ip = row[0].strip()
            reason = row[3] if len(row) > 3 else "Imported CSV Rule"
            block_type = row[4] if len(row) > 4 else "Hardware Firewall Rule"
            record_block_event(ip, reason, block_type)
            imported_count += 1

    return jsonify({"status": "Success", "message": f"Imported {imported_count} IPs."})

if __name__ == '__app__':
    app.run(debug=True, port=5000)
