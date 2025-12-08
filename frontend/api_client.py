"""
API Client for connecting Streamlit frontend to FastAPI backend.
Handles all HTTP requests to the backend API endpoints.
"""

import requests
from typing import Optional, Dict, List, Any
import os
from io import BytesIO


class APIClient:
    # Client for interacting with the FastAPI backend
    
    def __init__(self, base_url: Optional[str] = None):
        # Initialize API client with optional base URL
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        # Strip trailing slashes to avoid double slashes in endpoint URLs
        self.base_url = self.base_url.rstrip("/")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        # Make HTTP request to backend API
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(
                    url, 
                    files=files,
                    data=data,
                    json=json,
                    params=params,
                    timeout=60  # Longer timeout for image processing
                )
            elif method.upper() == "DELETE":
                response = requests.delete(url, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to backend at {self.base_url}. Is the backend running?")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out")
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {response.status_code} error"
            try:
                error_detail = response.json().get("detail", str(e))
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {str(e)}"
            raise requests.exceptions.HTTPError(error_msg)
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    def upload_wardrobe_item(
        self, 
        image_file: BytesIO, 
        filename: str,
        category: str
    ) -> Dict[str, Any]:
        # Upload a wardrobe item image to the backend
        # Reset file pointer to beginning
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
        
        files = {"file": (filename, image_file, "image/jpeg")}
        params = {"category": category}
        
        return self._make_request("POST", "/wardrobe/upload", files=files, params=params)
    
    def predict_image(self, image_file: BytesIO, filename: str) -> Dict[str, Any]:
        # Predict/classify an uploaded image
        if hasattr(image_file, 'seek'):
            image_file.seek(0)
        
        files = {"file": (filename, image_file, "image/jpeg")}
        
        return self._make_request("POST", "/predict", files=files)
    
    def generate_outfits(self, occasion: str, season: str) -> Dict[str, Any]:
        # Generate outfit recommendations based on occasion and season
        json_data = {
            "occasion": occasion,
            "season": season
        }
        
        return self._make_request("POST", "/outfits/generate", json=json_data)
    
    def save_outfit(self, item_ids: List[int], occasion: str, season: str, name: str = "") -> Dict[str, Any]:
        # Save an outfit to the database
        json_data = {
            "items": item_ids,
            "occasion": occasion,
            "season": season,
            "name": name
        }
        
        return self._make_request("POST", "/outfits/save", json=json_data)
    
    def get_saved_outfits(self) -> Dict[str, Any]:
        # Retrieve all saved outfits
        return self._make_request("GET", "/outfits/saved")
    
    def get_wardrobe_items(self) -> Dict[str, Any]:
        # Retrieve all wardrobe items from database
        return self._make_request("GET", "/wardrobe/items")
    
    def health_check(self) -> Dict[str, Any]:
        # Check if backend is running
        return self._make_request("GET", "/")
    
    def delete_wardrobe_item(self, item_id: int) -> Dict[str, Any]:
        # Delete a single wardrobe item from database
        return self._make_request("DELETE", f"/wardrobe/item/{item_id}")
    
    def delete_outfit(self, outfit_id: str) -> Dict[str, Any]:
        # Delete a saved outfit from database
        return self._make_request("DELETE", f"/outfits/{outfit_id}")
    
    def clear_all_wardrobe_items(self) -> Dict[str, Any]:
        # Clear all wardrobe items from database (for testing)
        return self._make_request("DELETE", "/wardrobe/clear-all")
