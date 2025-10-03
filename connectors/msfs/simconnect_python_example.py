import time, json, urllib.request

BASE = "http://localhost:8000"
FLIGHT_ID = "<paste your flight_id>"

def post(pt):
    req = urllib.request.Request(f"{BASE}/telemetry", data=json.dumps(pt).encode("utf-8"), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req) as resp:
        resp.read()

def main():
    k=0
    lat, lon, alt = 47.4502, -122.3088, 400
    while True:
        pt = {"flight_id": FLIGHT_ID, "k": k, "lat": lat, "lon": lon, "alt": alt, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            post(pt)
            print("posted", k)
        except Exception as e:
            print("post failed", e)
        k += 1
        lon += 0.01
        time.sleep(0.3)

if __name__ == "__main__":
    main()
