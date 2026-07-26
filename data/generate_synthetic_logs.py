"""
Synthetic Access-Log Generator for Behavioral Anomaly Detection
=================================================================

Generates realistic per-entity access/connection logs with injected attack
patterns for training + evaluating a behavioral anomaly detection system.

Design assumptions (documented per hackathon deliverable #1):
- 3 entity types: user, service_account, edge_device, each with distinct
  "normal" behavioral signatures (login-hour distribution, geo-stability,
  resource diversity, session-duration profile).
- Each user/device is assigned a home geo-location (city) and a small set
  of "known" devices/fingerprints. Rare legitimate travel is modeled
  separately from "impossible travel" attacks (legitimate travel respects
  plausible flight speeds; attacks do not).
- Attacks are injected as additional session records on top of the normal
  baseline, at independently configurable rates per attack type (spec asks
  for 0.5-3% of total sessions -> default 1.8% here, tunable below).
- Ground truth label is retained in `label` column for training/eval, and
  a labels-stripped version is also produced to simulate real inference
  conditions (label hidden at inference).
"""

import os
import numpy as np
import pandas as pd
from faker import Faker
import random
import uuid
from datetime import datetime, timedelta

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

SIM_START = datetime(2026, 5, 1)
SIM_DAYS = 60  # 2 months of activity

N_USERS = 120
N_SERVICE_ACCOUNTS = 25
N_EDGE_DEVICES = 35

# Target anomaly injection rate as a fraction of TOTAL normal sessions generated
ATTACK_RATE = 0.018  # 1.8%, within the 0.5-3% spec range

ATTACK_MIX = {
    # relative weights, must sum to 1.0
    "brute_force": 0.20,
    "impossible_travel": 0.15,
    "credential_stuffing": 0.15,
    "lateral_movement": 0.15,
    "device_spoofing": 0.15,
    "low_and_slow_exfil": 0.10,
    "insider_drift": 0.10,  # edge case, ambiguous label
}

# World cities used for geo-location simulation: (city, country, lat, lon)
CITIES = [
    ("Mumbai", "IN", 19.0760, 72.8777),
    ("Bengaluru", "IN", 12.9716, 77.5946),
    ("Delhi", "IN", 28.7041, 77.1025),
    ("Pune", "IN", 18.5204, 73.8567),
    ("Chennai", "IN", 13.0827, 80.2707),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("London", "GB", 51.5074, -0.1278),
    ("Frankfurt", "DE", 50.1109, 8.6821),
    ("New York", "US", 40.7128, -74.0060),
    ("San Francisco", "US", 37.7749, -122.4194),
    ("Sydney", "AU", -33.8688, 151.2093),
    ("Tokyo", "JP", 35.6762, 139.6503),
    ("Moscow", "RU", 55.7558, 37.6173),
    ("Lagos", "NG", 6.5244, 3.3792),
]
# Every city can be a legitimate home base. Reserving certain cities purely
# as attack origins would make those city names a giveaway for the label,
# which is an artifact of the simulator rather than a real detection signal.
HOME_CITIES = CITIES
ATTACK_ORIGIN_CITIES = CITIES  # attacks can originate anywhere

RESOURCE_POOL = [
    "file_share_hr", "file_share_finance", "file_share_eng",
    "db_prod_customers", "db_prod_orders", "db_analytics",
    "admin_panel_iam", "admin_panel_billing",
    "api_gateway_payments", "api_gateway_orders",
    "vpn_gateway_1", "vpn_gateway_2",
    "email_server", "source_repo_backend", "source_repo_frontend",
    "financial_system_erp", "print_server", "iot_hub_controller",
    "crm_system", "ticketing_system",
    # ports (spec: "file, endpoint, port, or device function")
    "port_22_ssh", "port_3389_rdp", "port_445_smb", "port_1883_mqtt",
    # device functions
    "device_fn_firmware_update", "device_fn_telemetry_push", "device_fn_reboot",
]
PRIVILEGED_RESOURCES = {
    "db_prod_customers", "db_prod_orders", "admin_panel_iam",
    "admin_panel_billing", "financial_system_erp", "iot_hub_controller",
}
PRIVILEGED_ACTIONS = ["read_record", "export_data", "modify_record",
                       "delete_record", "grant_access", "run_query"]

AUTH_METHODS = ["password", "token", "certificate", "biometric", "mfa_push"]
OS_FINGERPRINTS = ["Windows11-x64", "macOS-14", "Ubuntu-22.04",
                   "iOS-18", "Android-15", "RTOS-edge-fw2.3"]
PROTOCOLS = ["TLSv1.3", "TLSv1.2", "SSHv2", "MQTTv5", "HTTPS"]

# ----------------------------------------------------------------------------
# BENIGN CONFUSABILITY
# ----------------------------------------------------------------------------
# An earlier revision of this generator made every attack trivially separable,
# because the benign baseline never produced the signals the attacks were
# defined by. Three oracles existed, each measured at 100% label purity:
#
#   auth_result == "failure"              -> 100% anomaly (62% of all anomalies)
#   distinct_entities_per_ip_1h >= 5      -> 100% anomaly (19% of all anomalies)
#   geo_city != home_city                 -> anomaly by construction
#
# A two-line rule therefore filled the entire top-1% analyst alert budget with
# guaranteed true positives, and the reported "0 false positives" measured the
# simulator rather than the detector. The constants below inject the benign
# versions of each attack signal, so the detector must learn a THRESHOLD
# (how many failures, how much IP fan-out, how fast the implied travel)
# instead of a boolean.
#
# Each behaviour below is drawn from a real-world equivalent, not tuned to
# make the numbers move.

# 1. Password typos / expired certs. Real orgs run 2-8% benign auth failure.
#    Benign failures arrive in small bursts from a KNOWN device, which is what
#    separates them from a 15-60 attempt burst from an unknown one.
BENIGN_FAIL_RATE = {"user": (0.02, 0.06), "service_account": (0.002, 0.010),
                    "edge_device": (0.005, 0.020)}
BENIGN_FAIL_BURST = (1, 3)          # failures immediately before a success

# 2. Shared office egress IPs. Previously every benign session drew a fresh
#    random public IP, so ANY IP fan-out meant credential stuffing. Real
#    corporate traffic NATs hundreds of users behind one egress address.
OFFICE_IP_SHARE = 0.65              # share of on-site sessions using office IP

# 3. Second devices (personal phone, loaner laptop) and mid-timeline hardware
#    refresh outside the drift cohort, so "new device" is not proof of spoofing.
OCCASIONAL_DEVICE_RATE = 0.05
P_DEVICE_REFRESH = 0.15             # share of baseline entities issued new kit

# 4. Legitimate scope change: covering for a colleague, project rotation.
OCCASIONAL_RESOURCE_RATE = 0.03
P_RESOURCE_BECOMES_PERMANENT = 0.30

# 5. Genuine business travel, respecting commercial flight speed. Outbound and
#    return days carry no sessions (the entity is in transit), which guarantees
#    a >=24h gap and therefore an implied velocity below 900 km/h for every
#    city pair in CITIES. Impossible-travel attacks remain the only sub-flight
#    -time relocations.
P_TRAVELLER = 0.18                  # share of users who travel at all
TRIP_DAYS = (2, 5)


def _office_ips():
    return {city[0]: fake.ipv4_public() for city in CITIES}


OFFICE_IPS = _office_ips()


def random_mac():
    return ":".join(f"{random.randint(0, 255):02X}" for _ in range(6))


def make_fingerprint(os_name=None, mac=None, protocol=None):
    """device_fingerprint per spec: 'OS/firmware version, MAC address, protocol used'.

    Encoded as a single pipe-delimited string so it stays one schema column
    while carrying all three components the spec names.
    """
    os_name = os_name or random.choice(OS_FINGERPRINTS)
    mac = mac or random_mac()
    protocol = protocol or random.choice(PROTOCOLS)
    return f"{os_name}|MAC:{mac}|{protocol}"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------
# ENTITY PROFILE GENERATION
# ----------------------------------------------------------------------------

class EntityProfile:
    """Holds the 'normal' behavioral signature for one entity."""

    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.home_city = random.choice(HOME_CITIES)
        # cohort metadata (evaluation only, never a model feature):
        #   join_day     -> first day this entity exists (cold-start if late)
        #   drift_day    -> day legitimate behaviour permanently evolves
        self.cohort = "baseline"
        self.join_day = 0
        self.drift_day = None
        self.drift_hour_shift = 0.0
        self.drift_new_device = None

        # --- benign confusability state (see BENIGN CONFUSABILITY above) ---
        lo, hi = BENIGN_FAIL_RATE[entity_type]
        self.auth_failure_rate = np.random.uniform(lo, hi)
        self.office_ip = OFFICE_IPS[self.home_city[0]]
        self.residential_ip = fake.ipv4_public()   # stable home broadband / DC NIC
        self.occasional_devices = [make_fingerprint()
                                   for _ in range(random.randint(0, 2))]
        self.device_refresh_day = None             # legitimate hardware swap
        self.device_refresh_fp = None
        self.travel_trips = []                     # (start_day, duration, city)

        if entity_type == "user":
            self.login_hour_mean = np.random.normal(13, 3) % 24  # roughly work hours (UTC-ish)
            self.login_hour_std = np.random.uniform(1.5, 3.0)
            self.active_days = sorted(random.sample(range(7), k=random.randint(4, 6)))  # weekday-ish
            self.resources = set(random.sample(RESOURCE_POOL, k=random.randint(4, 8)))
            self.auth_method = random.choices(AUTH_METHODS, weights=[0.45, 0.25, 0.05, 0.20, 0.05])[0]
            self.known_devices = [make_fingerprint() for _ in range(random.randint(1, 2))]
            self.session_duration_mean = np.random.uniform(15, 60)  # minutes
            self.sessions_per_day = np.random.uniform(1.0, 4.0)

        elif entity_type == "service_account":
            self.login_hour_mean = np.random.choice([2, 3, 4, 14, 15])  # batch job hours, tight
            self.login_hour_std = 0.5
            self.active_days = list(range(7))  # runs every day
            self.resources = set(random.sample(RESOURCE_POOL, k=random.randint(1, 3)))
            self.auth_method = "certificate"
            self.known_devices = [make_fingerprint(random.choice(OS_FINGERPRINTS[:3]))]
            self.session_duration_mean = np.random.uniform(2, 10)
            self.sessions_per_day = np.random.uniform(2.0, 8.0)

        else:  # edge_device
            self.login_hour_mean = None  # devices connect around the clock
            self.login_hour_std = None
            self.active_days = list(range(7))
            self.resources = {"iot_hub_controller"}
            self.auth_method = "token"
            self.known_devices = [make_fingerprint(random.choice(OS_FINGERPRINTS[-2:]))]
            self.session_duration_mean = np.random.uniform(1, 5)
            self.sessions_per_day = np.random.uniform(10, 40)

        # Resources this entity may legitimately touch once in a while
        # (covering a colleague, project rotation) but that are not part of its
        # habitual set. Without these, "resource never seen before" is proof of
        # lateral movement rather than evidence for it.
        outside = [r for r in RESOURCE_POOL if r not in self.resources]
        self.occasional_resources = set(
            random.sample(outside, k=min(random.randint(2, 5), len(outside))))


LATE_JOIN_DAY = 47        # inside the final 25% of the timeline (test window)
N_LATE_JOIN_USERS = 12    # brand-new employees  -> cold-start
N_LATE_JOIN_DEVICES = 8   # newly provisioned IoT -> cold-start
N_DRIFT_USERS = 18        # legitimate behaviour change mid-timeline


def build_entities():
    entities = []
    for i in range(N_USERS):
        entities.append(EntityProfile(f"user_{i:04d}", "user"))
    for i in range(N_SERVICE_ACCOUNTS):
        entities.append(EntityProfile(f"svc_{i:03d}", "service_account"))
    for i in range(N_EDGE_DEVICES):
        entities.append(EntityProfile(f"edge_{i:03d}", "edge_device"))

    # --- COLD-START cohort: entities whose FIRST EVER session is in the test window ---
    for i in range(N_LATE_JOIN_USERS):
        p = EntityProfile(f"newuser_{i:03d}", "user")
        p.cohort, p.join_day = "late_join", LATE_JOIN_DAY
        entities.append(p)
    for i in range(N_LATE_JOIN_DEVICES):
        p = EntityProfile(f"newedge_{i:03d}", "edge_device")
        p.cohort, p.join_day = "late_join", LATE_JOIN_DAY
        entities.append(p)

    # --- CONCEPT DRIFT cohort: legitimate, gradual behaviour evolution ---
    # (new shift pattern + newly issued laptop). These stay labeled "normal":
    # a good system must NOT flag them permanently.
    drift_candidates = [e for e in entities if e.entity_type == "user"
                        and e.cohort == "baseline"]
    for p in random.sample(drift_candidates, k=min(N_DRIFT_USERS, len(drift_candidates))):
        p.cohort = "drifted"
        p.drift_day = random.randint(30, 44)
        p.drift_hour_shift = random.choice([-5, -4, -3, 3, 4, 5])
        p.drift_new_device = make_fingerprint()  # newly issued laptop

    # --- BENIGN TRAVEL: real trips that respect commercial flight speed ---
    # Outbound and return days are left empty (entity in transit), so the gap
    # either side of a trip is always >= 24h. At 900 km/h that covers 21,600 km
    # -- further than any pair of cities in CITIES -- so no legitimate trip can
    # ever imply impossible travel. Attacks remain the only sub-flight-time hops.
    for p in entities:
        if p.entity_type != "user" or random.random() > P_TRAVELLER:
            continue
        n_trips = random.randint(1, 2)
        for _ in range(n_trips):
            dur = random.randint(*TRIP_DAYS)
            latest = SIM_DAYS - dur - 3
            if latest <= p.join_day + 2:
                continue
            start = random.randint(p.join_day + 2, latest)
            city = random.choice([c for c in CITIES if c[0] != p.home_city[0]])
            p.travel_trips.append((start, dur, city))

    # --- BENIGN HARDWARE REFRESH outside the drift cohort ---
    for p in entities:
        if p.cohort == "baseline" and random.random() < P_DEVICE_REFRESH:
            p.device_refresh_day = random.randint(10, SIM_DAYS - 5)
            p.device_refresh_fp = make_fingerprint()

    return entities


def travel_state(profile, day):
    """('home', None) | ('transit', None) | ('away', city) for a given day."""
    for start, dur, city in profile.travel_trips:
        if day == start or day == start + dur + 1:
            return "transit", None
        if start < day <= start + dur:
            return "away", city
    return "home", None


def pick_source_ip(profile, on_site):
    """Shared office egress IP vs. stable residential/DC address.

    Benign IP fan-out is the whole point: many entities behind one office IP
    means 'many accounts from one source' cannot by itself mean credential
    stuffing.
    """
    if on_site and random.random() < OFFICE_IP_SHARE:
        return profile.office_ip
    return profile.residential_ip


def pick_device(profile, day):
    """Habitual device, with legitimate refreshes and occasional second devices."""
    drifted = profile.drift_day is not None and day >= profile.drift_day
    if drifted and profile.drift_new_device and random.random() < 0.7:
        return profile.drift_new_device
    if (profile.device_refresh_day is not None
            and day >= profile.device_refresh_day
            and random.random() < 0.8):
        return profile.device_refresh_fp
    if profile.occasional_devices and random.random() < OCCASIONAL_DEVICE_RATE:
        return random.choice(profile.occasional_devices)
    return random.choice(profile.known_devices)


def pick_resource(profile):
    """Habitual resource, with occasional legitimate scope expansion."""
    if profile.occasional_resources and random.random() < OCCASIONAL_RESOURCE_RATE:
        r = random.choice(sorted(profile.occasional_resources))
        if random.random() < P_RESOURCE_BECOMES_PERMANENT:
            profile.resources.add(r)          # permanent, legitimate expansion
            profile.occasional_resources.discard(r)
        return r
    return random.choice(sorted(profile.resources))


# ----------------------------------------------------------------------------
# NORMAL SESSION GENERATION
# ----------------------------------------------------------------------------

def sample_normal_timestamp(profile, day_offset):
    base_day = SIM_START + timedelta(days=day_offset)
    if profile.entity_type == "edge_device":
        # roughly uniform across the day
        seconds = random.randint(0, 86399)
        return base_day + timedelta(seconds=seconds)
    hour = np.clip(np.random.normal(profile.login_hour_mean, profile.login_hour_std), 0, 23.99)
    minute = random.randint(0, 59)
    return base_day + timedelta(hours=hour, minutes=minute)


def generate_normal_sessions(profile):
    rows = []
    for day in range(profile.join_day, SIM_DAYS):
        weekday = (SIM_START + timedelta(days=day)).weekday()
        # --- benign CONCEPT DRIFT: legitimate behaviour permanently evolves ---
        drifted = profile.drift_day is not None and day >= profile.drift_day

        # --- benign TRAVEL: in transit means no sessions at all that day ---
        state, trip_city = travel_state(profile, day)
        if state == "transit":
            continue
        on_site = state == "home"
        city = profile.home_city if on_site else trip_city

        if profile.entity_type == "user" and weekday not in profile.active_days:
            # travelling users work through the weekend more often
            skip_p = 0.05 if on_site else 0.40
            if random.random() > skip_p:
                continue
        n_sessions_today = np.random.poisson(profile.sessions_per_day)
        for _ in range(max(0, n_sessions_today)):
            ts = sample_normal_timestamp(profile, day)
            if drifted and profile.drift_hour_shift:
                ts = ts + timedelta(hours=float(profile.drift_hour_shift))
            resource = pick_resource(profile)
            device = pick_device(profile, day)
            src_ip = pick_source_ip(profile, on_site)
            cmd_seq = []
            if resource in PRIVILEGED_RESOURCES:
                cmd_seq = random.sample(PRIVILEGED_ACTIONS, k=random.randint(1, 2))

            def _row(timestamp, auth_result, duration, session_cmds):
                return {
                    "session_id": str(uuid.uuid4())[:12],
                    "entity_id": profile.entity_id,
                    "entity_type": profile.entity_type,
                    "timestamp": timestamp,
                    "source_ip": src_ip,
                    "geo_city": city[0],
                    "geo_lat": city[2],
                    "geo_lon": city[3],
                    "resource_accessed": resource,
                    "auth_method": profile.auth_method,
                    "auth_result": auth_result,
                    "session_duration": duration,
                    "command_sequence": ";".join(session_cmds),
                    "device_fingerprint": device,
                    "label": "normal",
                    "cohort": profile.cohort,
                }

            # --- benign AUTH FAILURES: password typo / expired cert, from the
            # entity's OWN device and IP, in a burst of 1-3 before succeeding.
            # Attackers produce 15-60 from an unknown device, so the model must
            # learn a threshold on failure COUNT rather than on failure itself.
            if random.random() < profile.auth_failure_rate:
                n_fail = random.randint(*BENIGN_FAIL_BURST)
                for k in range(n_fail, 0, -1):
                    rows.append(_row(
                        ts - timedelta(seconds=k * random.randint(4, 25)),
                        "failure", round(random.uniform(0.05, 0.3), 3), []))

            rows.append(_row(
                ts, "success",
                max(0.5, np.random.normal(profile.session_duration_mean,
                                          profile.session_duration_mean * 0.3)),
                cmd_seq))
    return rows


# ----------------------------------------------------------------------------
# ATTACK INJECTORS
# ----------------------------------------------------------------------------

def inject_brute_force(profile, day_lo=0):
    """Rapid repeated failed-auth attempts from one source in a short window."""
    day = random.randint(day_lo, SIM_DAYS - 1)
    start = SIM_START + timedelta(days=day, hours=random.randint(0, 23))
    attacker_ip = fake.ipv4_public()
    # A real attacker connects from a real place on a real machine. Using a
    # sentinel like "Unknown" here would hand the model a free giveaway
    # (100% label purity), so the attacker gets a genuine but UNFAMILIAR
    # city and a genuine but UNRECOGNISED device fingerprint.
    a_city = random.choice([c for c in ATTACK_ORIGIN_CITIES
                            if c[0] != profile.home_city[0]])
    attacker_fp = make_fingerprint()
    n_attempts = random.randint(15, 60)
    rows = []
    for i in range(n_attempts):
        ts = start + timedelta(seconds=i * random.randint(2, 8))
        rows.append({
            "session_id": str(uuid.uuid4())[:12],
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": attacker_ip,
            "geo_city": a_city[0], "geo_lat": a_city[2], "geo_lon": a_city[3],
            "resource_accessed": random.choice(list(profile.resources)),
            "auth_method": profile.auth_method,
            "auth_result": "failure" if i < n_attempts - 1 else random.choice(["failure", "success"]),
            "session_duration": round(random.uniform(0.05, 0.4), 3),
            "command_sequence": "",
            "device_fingerprint": attacker_fp,
            "label": "anomaly_brute_force",
        })
    return rows


def inject_impossible_travel(profile, day_lo=0):
    """Same entity logging in from geographically distant locations within an
    implausible time gap (faster than ~900 km/h commercial flight speed)."""
    day = random.randint(day_lo, SIM_DAYS - 1)
    t1 = SIM_START + timedelta(days=day, hours=random.randint(0, 20))
    city1 = profile.home_city
    city2 = random.choice([c for c in ATTACK_ORIGIN_CITIES if c[0] != city1[0]])
    dist = haversine_km(city1[2], city1[3], city2[2], city2[3])
    min_hours = dist / 900.0
    gap_hours = max(0.1, min_hours * random.uniform(0.1, 0.5))  # impossibly fast
    t2 = t1 + timedelta(hours=gap_hours)
    rows = []
    for ts, city, lab in [(t1, city1, "normal"), (t2, city2, "anomaly_impossible_travel")]:
        rows.append({
            "session_id": str(uuid.uuid4())[:12],
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": fake.ipv4_public(),
            "geo_city": city[0], "geo_lat": city[2], "geo_lon": city[3],
            "resource_accessed": random.choice(list(profile.resources)),
            "auth_method": profile.auth_method,
            "auth_result": "success",
            "session_duration": np.random.uniform(5, 30),
            "command_sequence": "",
            "device_fingerprint": random.choice(profile.known_devices),
            "label": lab,
        })
    return rows


def inject_credential_stuffing(all_profiles, day_lo=0):
    """Many entity_ids, few source_ips, high failure rate."""
    day = random.randint(day_lo, SIM_DAYS - 1)
    start = SIM_START + timedelta(days=day, hours=random.randint(0, 23))
    attacker_ips = [fake.ipv4_public() for _ in range(random.randint(2, 4))]
    # each attacking host has one consistent origin city + device fingerprint
    ip_origin = {ip: random.choice(ATTACK_ORIGIN_CITIES) for ip in attacker_ips}
    ip_device = {ip: make_fingerprint() for ip in attacker_ips}
    targets = random.sample(all_profiles, k=random.randint(20, 50))
    rows = []
    for i, profile in enumerate(targets):
        ts = start + timedelta(seconds=i * random.randint(1, 5))
        src_ip = random.choice(attacker_ips)
        o_city = ip_origin[src_ip]
        rows.append({
            "session_id": str(uuid.uuid4())[:12],
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": src_ip,
            "geo_city": o_city[0], "geo_lat": o_city[2], "geo_lon": o_city[3],
            "resource_accessed": random.choice(RESOURCE_POOL),
            "auth_method": profile.auth_method,
            "auth_result": "failure" if random.random() < 0.9 else "success",
            "session_duration": round(random.uniform(0.05, 0.4), 3),
            "command_sequence": "",
            "device_fingerprint": ip_device[src_ip],
            "label": "anomaly_credential_stuffing",
        })
    return rows


def inject_lateral_movement(profile, day_lo=0):
    """Compromised entity accessing an unusual breadth/sequence of resources
    it has never touched before."""
    day = random.randint(day_lo, SIM_DAYS - 1)
    start = SIM_START + timedelta(days=day, hours=random.randint(0, 23))
    unseen_resources = [r for r in RESOURCE_POOL if r not in profile.resources]
    n_hops = random.randint(5, 10)
    hops = random.sample(unseen_resources, k=min(n_hops, len(unseen_resources)))
    rows = []
    for i, resource in enumerate(hops):
        ts = start + timedelta(minutes=i * random.randint(2, 6))
        cmd_seq = random.sample(PRIVILEGED_ACTIONS, k=random.randint(1, 3)) if resource in PRIVILEGED_RESOURCES else []
        rows.append({
            "session_id": str(uuid.uuid4())[:12],
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": fake.ipv4_public(),
            "geo_city": profile.home_city[0], "geo_lat": profile.home_city[2], "geo_lon": profile.home_city[3],
            "resource_accessed": resource,
            "auth_method": profile.auth_method,
            "auth_result": "success",
            "session_duration": np.random.uniform(1, 8),
            "command_sequence": ";".join(cmd_seq),
            "device_fingerprint": random.choice(profile.known_devices),
            "label": "anomaly_lateral_movement",
        })
    return rows


def inject_device_spoofing(profile, day_lo=0):
    """Device/entity reappearing with a mismatched fingerprint."""
    day = random.randint(day_lo, SIM_DAYS - 1)
    ts = SIM_START + timedelta(days=day, hours=random.randint(0, 23))
    # Spec: "a device_id reappearing with a mismatched fingerprint
    # (different OS/MAC than history)". Half the incidents change only the MAC
    # (subtle hardware spoof), half change the OS too (blunt spoof).
    known_os = [fp.split("|")[0] for fp in profile.known_devices]
    if random.random() < 0.5:
        spoofed_fp = make_fingerprint(os_name=random.choice(known_os))   # new MAC only
    else:
        alt_os = [o for o in OS_FINGERPRINTS if o not in known_os] or OS_FINGERPRINTS
        spoofed_fp = make_fingerprint(os_name=random.choice(alt_os))
    return [{
        "session_id": str(uuid.uuid4())[:12],
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type,
        "timestamp": ts,
        "source_ip": fake.ipv4_public(),
        "geo_city": profile.home_city[0], "geo_lat": profile.home_city[2], "geo_lon": profile.home_city[3],
        "resource_accessed": random.choice(list(profile.resources) or RESOURCE_POOL),
        "auth_method": profile.auth_method,
        "auth_result": "success",
        "session_duration": np.random.uniform(1, 15),
        "command_sequence": "",
        "device_fingerprint": spoofed_fp,
        "label": "anomaly_device_spoofing",
    }]


def inject_low_and_slow_exfil(profile, day_lo=0):
    """Gradual, SMALL, off-hours resource access building up over days/weeks.

    The defining property of low-and-slow exfiltration is that every
    individual access stays *below* the size that would trip a per-event
    threshold; the volume accumulates through FREQUENCY, not through larger
    sessions. Earlier versions grew the session duration, which contradicted
    the spec's word "small" and made the pattern trivially detectable as an
    outlier on duration alone.
    """
    start_day = random.randint(min(day_lo, SIM_DAYS - 15), SIM_DAYS - 15)
    n_days = random.randint(7, 14)
    rows = []
    resources_pool = list(profile.resources) + random.sample(
        [r for r in RESOURCE_POOL if r not in profile.resources], k=2)
    for d in range(n_days):
        day = start_day + d
        # buildup: accesses per night ramps 1 -> ~4 across the campaign
        n_tonight = 1 + int(d * 3 / max(n_days - 1, 1))
        for _ in range(n_tonight):
            off_hour = random.choice([1, 2, 3, 22, 23])
            ts = SIM_START + timedelta(days=day, hours=off_hour,
                                       minutes=random.randint(0, 59))
            rows.append({
                "session_id": str(uuid.uuid4())[:12],
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": ts,
                "source_ip": fake.ipv4_public(),
                "geo_city": profile.home_city[0], "geo_lat": profile.home_city[2],
                "geo_lon": profile.home_city[3],
                "resource_accessed": random.choice(resources_pool),
                "auth_method": profile.auth_method,
                "auth_result": "success",
                # deliberately SMALL: shorter than this entity's typical session
                "session_duration": round(np.random.uniform(0.5, 2.5), 2),
                "command_sequence": "export_data",
                "device_fingerprint": random.choice(profile.known_devices),
                "label": "anomaly_low_and_slow_exfil",
            })
    return rows


def inject_insider_drift(profile):
    """Legitimate entity slowly expanding privilege/resource footprint.
    Labeled as an EDGE CASE (not 'anomaly') -- used to tune false positives."""
    start_day = random.randint(0, SIM_DAYS - 20)
    n_days = random.randint(10, 18)
    rows = []
    expanding_pool = [r for r in RESOURCE_POOL if r not in profile.resources]
    for d in range(n_days):
        if random.random() > 0.4:
            continue
        day = start_day + d
        ts = sample_normal_timestamp(profile, day)
        # Spec: "slowly expanding PRIVILEGE or resource footprint". Footprint
        # widens over time AND the share of privileged resources rises, so the
        # pattern is genuinely ambiguous rather than merely novel.
        priv_prob = min(0.15 + 0.05 * d, 0.75)
        priv_pool = [r for r in PRIVILEGED_RESOURCES if r not in profile.resources]
        if priv_pool and random.random() < priv_prob:
            resource = random.choice(priv_pool)
        else:
            resource = random.choice(expanding_pool[: max(1, d // 2)] or expanding_pool)
        rows.append({
            "session_id": str(uuid.uuid4())[:12],
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": fake.ipv4_public(),
            "geo_city": profile.home_city[0], "geo_lat": profile.home_city[2], "geo_lon": profile.home_city[3],
            "resource_accessed": resource,
            "auth_method": profile.auth_method,
            "auth_result": "success",
            "session_duration": np.random.uniform(5, 25),
            "command_sequence": (";".join(random.sample(PRIVILEGED_ACTIONS,
                                                        k=random.randint(1, 2)))
                                 if resource in PRIVILEGED_RESOURCES else ""),
            "device_fingerprint": random.choice(profile.known_devices),
            "label": "edge_case_insider_drift",
        })
    return rows


# ----------------------------------------------------------------------------
# MAIN GENERATION PIPELINE
# ----------------------------------------------------------------------------

def main():
    print("Building entity profiles...")
    entities = build_entities()
    established = [e for e in entities if e.cohort != "late_join"]
    user_and_svc = [e for e in established
                    if e.entity_type in ("user", "service_account")]
    established_users = [e for e in established if e.entity_type == "user"]

    print("Generating normal baseline sessions...")
    normal_rows = []
    for profile in entities:
        normal_rows.extend(generate_normal_sessions(profile))
    n_normal = len(normal_rows)
    print(f"  -> {n_normal} normal sessions generated")

    n_attack_sessions_target = int(n_normal * ATTACK_RATE)
    print(f"Injecting attacks (~{n_attack_sessions_target} attack-related rows target)...")

    attack_rows = []
    # Roughly allocate "incidents" (each incident produces several rows) per type
    budget = n_attack_sessions_target
    while sum(len(r) for r in [attack_rows]) < n_attack_sessions_target:
        attack_type = random.choices(list(ATTACK_MIX.keys()), weights=list(ATTACK_MIX.values()))[0]
        if attack_type == "brute_force":
            profile = random.choice(user_and_svc)
            attack_rows.extend(inject_brute_force(profile))
        elif attack_type == "impossible_travel":
            profile = random.choice(established_users)
            attack_rows.extend(inject_impossible_travel(profile))
        elif attack_type == "credential_stuffing":
            attack_rows.extend(inject_credential_stuffing(user_and_svc))
        elif attack_type == "lateral_movement":
            profile = random.choice(user_and_svc)
            attack_rows.extend(inject_lateral_movement(profile))
        elif attack_type == "device_spoofing":
            profile = random.choice(established)
            attack_rows.extend(inject_device_spoofing(profile))
        elif attack_type == "low_and_slow_exfil":
            profile = random.choice(established_users)
            attack_rows.extend(inject_low_and_slow_exfil(profile))
        elif attack_type == "insider_drift":
            profile = random.choice(established_users)
            attack_rows.extend(inject_insider_drift(profile))


    # --- Evaluation-coverage pass: guarantee every attack type appears in the
    # final 25% of the timeline (the held-out test window) ---
    LATE = 46
    for _ in range(12):
        attack_rows.extend(inject_impossible_travel(random.choice(established_users), day_lo=LATE))
        attack_rows.extend(inject_device_spoofing(random.choice(established), day_lo=LATE))
    for _ in range(12):
        attack_rows.extend(inject_impossible_travel(random.choice(established_users), day_lo=0))
        attack_rows.extend(inject_device_spoofing(random.choice(established), day_lo=0))
    for _ in range(2):
        attack_rows.extend(inject_brute_force(random.choice(user_and_svc), day_lo=LATE))
        attack_rows.extend(inject_impossible_travel(random.choice(established_users), day_lo=LATE))
        attack_rows.extend(inject_credential_stuffing(user_and_svc, day_lo=LATE))
        attack_rows.extend(inject_lateral_movement(random.choice(user_and_svc), day_lo=LATE))
        attack_rows.extend(inject_device_spoofing(random.choice(established), day_lo=LATE))
        attack_rows.extend(inject_low_and_slow_exfil(random.choice(established_users), day_lo=LATE))

    print(f"  -> {len(attack_rows)} attack rows generated across {len(ATTACK_MIX)} patterns")

    all_rows = normal_rows + attack_rows
    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "cohort" not in df.columns:
        df["cohort"] = "baseline"
    df["cohort"] = df["cohort"].fillna("attack_injected")

    print(f"Total dataset size: {len(df)} rows")
    print(df["label"].value_counts())
    print("\nCohorts (evaluation slices):")
    print(df["cohort"].value_counts())

    # Save full labeled version (for training/eval)
    df.to_csv(f"{_PROJECT_ROOT}/data/access_logs_labeled.csv", index=False)

    # Save an inference-style version with labels stripped (simulates real deployment)
    df_unlabeled = df.drop(columns=["label", "cohort"])
    df_unlabeled.to_csv(f"{_PROJECT_ROOT}/data/access_logs_unlabeled.csv", index=False)

    # Save ground truth separately, keyed by session_id
    df[["session_id", "label"]].to_csv(f"{_PROJECT_ROOT}/data/ground_truth_labels.csv", index=False)

    print("Saved: access_logs_labeled.csv, access_logs_unlabeled.csv, ground_truth_labels.csv")
    return df


if __name__ == "__main__":
    main()
