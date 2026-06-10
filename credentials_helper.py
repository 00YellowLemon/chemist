import os
import json
import tempfile

def setup_google_credentials():
    """
    Initializes and sets up the Google Cloud credentials.
    1. If GOOGLE_SERVICE_ACCOUNT_JSON is present and contains JSON, write it to a temp file
       and point GOOGLE_APPLICATION_CREDENTIALS to it.
    2. If GOOGLE_APPLICATION_CREDENTIALS contains inline JSON, write it to a temp file
       and point GOOGLE_APPLICATION_CREDENTIALS to it.
    3. If GOOGLE_APPLICATION_CREDENTIALS is not set, try to fall back to the local developer key.
    4. If GOOGLE_APPLICATION_CREDENTIALS points to a local file that does not exist,
       clear the environment variable so Google SDKs fall back to standard Application Default Credentials (ADC)
       (which is the default behavior on GCP Cloud Run using the service's runtime service account).
    """
    # 1. Check for GOOGLE_SERVICE_ACCOUNT_JSON
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    
    # 2. Check for inline JSON in GOOGLE_APPLICATION_CREDENTIALS
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    json_str = None
    if sa_json and sa_json.strip().startswith("{"):
        json_str = sa_json
    elif creds_path and creds_path.strip().startswith("{"):
        json_str = creds_path
        
    if json_str:
        try:
            # Parse to validate JSON
            creds_data = json.loads(json_str)
            # Create a temporary file in the temp directory (e.g. /tmp/gcp_service_account.json)
            temp_dir = tempfile.gettempdir()
            temp_cred_path = os.path.join(temp_dir, "gcp_service_account.json")
            with open(temp_cred_path, "w", encoding="utf-8") as f:
                json.dump(creds_data, f)
            
            # Point GOOGLE_APPLICATION_CREDENTIALS to the temp file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_path
            print(f"[INFO] Inline credentials written to temporary file at: {temp_cred_path}")
            return temp_cred_path
        except Exception as e:
            print(f"[ERROR] Failed to parse/write inline credentials to temporary file: {e}")
            # Fall back to checking path

    # 3. Local fallback for development if credentials aren't set
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        ricit_key = os.path.join("..", "ricit", "restaurant-c1836-firebase-adminsdk-fbsvc-3d3c323a70.json")
        if os.path.exists(ricit_key):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ricit_key
            print(f"[INFO] Local developer key file found and loaded: {ricit_key}")
            return ricit_key

    # 4. If it is a file path, check if it exists
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        # If it doesn't exist, clear it to prevent library crashes and use default ADC
        if not os.path.exists(creds_path):
            print(f"[WARNING] Google credentials file not found at: {creds_path}. Clearing GOOGLE_APPLICATION_CREDENTIALS to fall back to Application Default Credentials.")
            if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        else:
            print(f"[INFO] Google credentials found at: {creds_path}")
            return creds_path
    else:
        print("[INFO] GOOGLE_APPLICATION_CREDENTIALS is not set. Defaulting to Application Default Credentials.")
    
    return None
