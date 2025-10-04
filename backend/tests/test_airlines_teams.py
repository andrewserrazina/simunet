import importlib
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

try:
    from fastapi.testclient import TestClient  # type: ignore

    HAS_TEST_CLIENT = True
except Exception:  # pragma: no cover - optional dependency guard
    TestClient = None  # type: ignore
    HAS_TEST_CLIENT = False


@unittest.skipUnless(HAS_TEST_CLIENT, "httpx is required for TestClient")
class AirlineTeamJobFlowTest(unittest.TestCase):
    """Validate virtual airline and team mission workflows."""

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        state_path = Path(self._tempdir.name) / "state.json"
        os.environ["SIMUNET_STATE_PATH"] = str(state_path)

        for name in list(sys.modules):
            if name == "backend" or name.startswith("backend."):
                sys.modules.pop(name)

        import backend.app  # noqa: F401  # force fresh import with new state path

        self.app_module = importlib.reload(sys.modules["backend.app"])
        if not HAS_TEST_CLIENT or TestClient is None:  # pragma: no cover - safety check
            self.skipTest("httpx is required for TestClient")
        self.client = TestClient(self.app_module.app)

    def tearDown(self) -> None:
        if HAS_TEST_CLIENT and TestClient is not None:
            self.client.close()
        self._tempdir.cleanup()
        os.environ.pop("SIMUNET_STATE_PATH", None)

    def test_team_airline_job_claim_flow(self) -> None:
        """Simulate creating a job for a team and claiming it as a member."""

        register_admin = self.client.post(
            "/auth/register",
            json={"email": "ops@example.com", "password": "password123"},
        )
        self.assertEqual(register_admin.status_code, 201)

        airline_resp = self.client.post(
            "/airlines",
            json={
                "name": "Aurora Air",
                "description": "Northern cargo operations",
                "created_by": "ops@example.com",
            },
        )
        self.assertEqual(airline_resp.status_code, 201)
        airline_data = airline_resp.json()
        airline_id = airline_data["airline"]["id"]

        team_resp = self.client.post(
            "/teams",
            json={
                "name": "Northern Dispatch",
                "description": "Coordinated mountain hops",
                "created_by": "ops@example.com",
                "airline_id": airline_id,
            },
        )
        self.assertEqual(team_resp.status_code, 201)
        team_data = team_resp.json()
        team_id = team_data["team"]["id"]

        register_pilot = self.client.post(
            "/auth/register",
            json={"email": "pilot@example.com", "password": "password123"},
        )
        self.assertEqual(register_pilot.status_code, 201)

        deadline = datetime.utcnow() + timedelta(hours=6)
        job_payload = {
            "title": "Glacier Supply Chain",
            "platform": "Microsoft Flight Simulator",
            "payload": "Field equipment",
            "weight_lbs": 5400,
            "departure_airport": "PANC",
            "arrival_airport": "PAFG",
            "deadline": deadline.isoformat() + "Z",
            "notes": "Coordinate with Fairbanks ground crew for unloading.",
            "created_by": "ops@example.com",
            "team_id": team_id,
            "airline_id": airline_id,
            "legs": [
                {"seq": 1, "mode": "flight", "origin_airport": "PANC", "destination_airport": "PABE"},
                {"seq": 2, "mode": "flight", "origin_airport": "PABE", "destination_airport": "PAFG"},
            ],
        }

        job_resp = self.client.post("/jobs", json=job_payload)
        self.assertEqual(job_resp.status_code, 201)
        job_data = job_resp.json()["job"]
        job_id = job_data["job_id"]

        listing_before = self.client.get("/jobs", params={"email": "pilot@example.com"})
        self.assertEqual(listing_before.status_code, 200)
        payload_before = listing_before.json()
        self.assertEqual(payload_before.get("available"), [])

        join_resp = self.client.post(
            f"/teams/{team_id}/join",
            json={"email": "pilot@example.com"},
        )
        self.assertEqual(join_resp.status_code, 200)

        listing_after = self.client.get("/jobs", params={"email": "pilot@example.com"})
        self.assertEqual(listing_after.status_code, 200)
        payload_after = listing_after.json()
        self.assertEqual(len(payload_after["available"]), 1)
        available_job = payload_after["available"][0]
        self.assertEqual(available_job["job_id"], job_id)
        self.assertEqual(available_job["team_id"], team_id)
        self.assertEqual(available_job["airline_id"], airline_id)
        self.assertEqual(len(available_job["legs"]), 2)

        claim_resp = self.client.post(
            f"/jobs/{job_id}/claim",
            json={"email": "pilot@example.com"},
        )
        self.assertEqual(claim_resp.status_code, 200)

        listing_final = self.client.get("/jobs", params={"email": "pilot@example.com"})
        self.assertEqual(listing_final.status_code, 200)
        payload_final = listing_final.json()
        self.assertEqual(payload_final["available"], [])
        self.assertEqual(len(payload_final["mine"]), 1)
        self.assertEqual(payload_final["mine"][0]["job_id"], job_id)
        self.assertEqual(payload_final["mine"][0]["assigned_to"], "pilot@example.com")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
