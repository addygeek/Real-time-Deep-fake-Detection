import requests
import sys
import os

BASE_URL = "http://localhost:4000"

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health Check Passed")
            print(response.json())
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health Check Error: {e}")

def test_upload_flow():
    # Create a dummy video file
    dummy_filename = "test_video.mp4"
    with open(dummy_filename, "wb") as f:
        f.write(os.urandom(1024 * 1024)) # 1MB dummy file

    try:
        print("Testing Upload...")
        files = {'video': (dummy_filename, open(dummy_filename, 'rb'), 'video/mp4')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 201:
            data = response.json()
            analysis_id = data.get('analysisId')
            print(f"✅ Upload Passed. Analysis ID: {analysis_id}")
            
            # Check Status
            status_url = f"{BASE_URL}/analysis/status/{analysis_id}"
            status_res = requests.get(status_url)
            print(f"ℹ️ Initial Status: {status_res.json().get('status')}")
            
        else:
            print(f"❌ Upload Failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Upload Test Error: {e}")
    finally:
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)

if __name__ == "__main__":
    print("Running Integration Tests...")
    test_health()
    test_upload_flow()
