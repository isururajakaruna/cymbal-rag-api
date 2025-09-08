#!/usr/bin/env python3
"""
Quick API Test Script for Cymbal RAG API

This script performs a comprehensive test of the RAG API by:
1. Checking health endpoint
2. Validating a test PDF file
3. Uploading the PDF file
4. Listing files
5. Performing RAG search
6. Testing authentication

Usage: python3 scripts/quick_api_test.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "InaqhBh3P0MaJCBQnxF05DsdpWjbESpLJvoa-2tfwxI")
TEST_PDF_PATH = "test_data/post_pdf.pdf"

class APITester:
    """API testing class for Cymbal RAG API."""
    
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url
        self.auth_token = auth_token
        self.session = requests.Session()
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, message: str = "", response_data: dict = None):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        if response_data and not success:
            print(f"   Response: {json.dumps(response_data, indent=2)}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "response": response_data
        })
    
    def make_request(self, method: str, endpoint: str, **kwargs):
        """Make authenticated request to API."""
        url = f"{self.base_url}{endpoint}"
        
        # Add token to query parameters for GET requests
        if method.upper() == "GET":
            params = kwargs.get("params", {})
            params["token"] = self.auth_token
            kwargs["params"] = params
        else:
            # For POST requests, add token to query params
            if "params" not in kwargs:
                kwargs["params"] = {}
            kwargs["params"]["token"] = self.auth_token
        
        # Set timeout based on operation type
        timeout = 180 if method.upper() == "POST" and "upload" in endpoint else 60
        
        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
            return response, response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            return None, {"error": str(e)}
        except json.JSONDecodeError:
            return None, {"error": "Invalid JSON response"}
    
    def test_health_endpoint(self):
        """Test health endpoint (should work without auth)."""
        print("\n🔍 Testing Health Endpoint...")
        print(f"   GET {self.base_url}/health")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Health Check", True, f"API is healthy - {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("Health Check", False, f"Health check failed with status {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Health check failed: {str(e)}")
            return False
    
    def test_authentication(self):
        """Test authentication by accessing protected endpoint."""
        print("\n🔐 Testing Authentication...")
        
        # Test without token (should fail)
        print(f"   GET {self.base_url}/api/v1/files/list (no token)")
        try:
            response = requests.get(f"{self.base_url}/api/v1/files/list", timeout=10)
            if response.status_code == 401:
                self.log_test("Auth - No Token", True, "Correctly rejected request without token")
            else:
                self.log_test("Auth - No Token", False, f"Should have rejected request without token, got {response.status_code}")
        except Exception as e:
            self.log_test("Auth - No Token", False, f"Request failed: {str(e)}")
        
        # Test with invalid token (should fail)
        print(f"   GET {self.base_url}/api/v1/files/list?token=invalid-token")
        try:
            response = requests.get(f"{self.base_url}/api/v1/files/list?token=invalid-token", timeout=10)
            if response.status_code == 401:
                self.log_test("Auth - Invalid Token", True, "Correctly rejected invalid token")
            else:
                self.log_test("Auth - Invalid Token", False, f"Should have rejected invalid token, got {response.status_code}")
        except Exception as e:
            self.log_test("Auth - Invalid Token", False, f"Request failed: {str(e)}")
        
        # Test with valid token (should succeed)
        print(f"   GET {self.base_url}/api/v1/files/list?token={self.auth_token}")
        response, data = self.make_request("GET", "/api/v1/files/list")
        if response and response.status_code == 200:
            self.log_test("Auth - Valid Token", True, "Successfully authenticated with valid token")
            return True
        else:
            self.log_test("Auth - Valid Token", False, f"Authentication failed: {data}")
            return False
    
    def test_file_validation(self):
        """Test file validation endpoint."""
        print("\n📄 Testing File Validation...")
        print(f"   POST {self.base_url}/api/v1/file/validate?token={self.auth_token}")
        
        if not os.path.exists(TEST_PDF_PATH):
            self.log_test("File Validation", False, f"Test file not found: {TEST_PDF_PATH}")
            return False
        
        with open(TEST_PDF_PATH, "rb") as f:
            files = {"file": ("post_pdf.pdf", f, "application/pdf")}
            response, data = self.make_request("POST", "/api/v1/file/validate", files=files)
        
        if response and response.status_code == 200 and data.get("success"):
            self.log_test("File Validation", True, f"File validated successfully - {data.get('content_analysis', {}).get('content_quality', {}).get('score', 'N/A')}/10")
            return True
        else:
            self.log_test("File Validation", False, f"File validation failed: {data}")
            return False
    
    def test_file_upload(self):
        """Test file upload endpoint."""
        print("\n📤 Testing File Upload...")
        print(f"   POST {self.base_url}/api/v1/upload/direct?token={self.auth_token}")
        print(f"   File: {os.path.abspath(TEST_PDF_PATH)}")
        
        if not os.path.exists(TEST_PDF_PATH):
            self.log_test("File Upload", False, f"Test file not found: {TEST_PDF_PATH}")
            return False
        
        # Verify file exists and get its size
        file_size = os.path.getsize(TEST_PDF_PATH)
        print(f"   File size: {file_size} bytes")
        
        with open(TEST_PDF_PATH, "rb") as f:
            files = {"file": ("post_pdf.pdf", f, "application/pdf")}
            data = {"tags": "test,api,verification"}
            response, result = self.make_request("POST", "/api/v1/upload/direct", files=files, data=data)
        
        if response and response.status_code == 200 and result.get("success"):
            self.log_test("File Upload", True, f"File uploaded successfully - {result.get('chunks_created', 0)} chunks created")
            return True
        else:
            self.log_test("File Upload", False, f"File upload failed: {result}")
            return False
    
    def test_file_list(self):
        """Test file listing endpoint."""
        print("\n📋 Testing File List...")
        print(f"   GET {self.base_url}/api/v1/files/list?token={self.auth_token}")
        
        response, data = self.make_request("GET", "/api/v1/files/list")
        
        if response and response.status_code == 200 and data.get("success"):
            files = data.get("files", [])
            self.log_test("File List", True, f"Retrieved {len(files)} files")
            
            # Check if our test file is in the list
            test_file_found = any("post_pdf.pdf" in file.get("name", "") for file in files)
            if test_file_found:
                self.log_test("File List - Test File", True, "Test file found in file list")
            else:
                self.log_test("File List - Test File", False, "Test file not found in file list")
            
            return True
        else:
            self.log_test("File List", False, f"File listing failed: {data}")
            return False
    
    def test_rag_search(self):
        """Test RAG search endpoint."""
        print("\n🔍 Testing RAG Search...")
        print(f"   POST {self.base_url}/api/v1/search/rag?token={self.auth_token}")
        
        search_query = "WHO & AOH Cut-Off Time"
        search_data = {
            "query": search_query,
            "ktop": 3,
            "threshold": 0.5
        }
        print(f"   Query: {search_query}")
        print(f"   Parameters: ktop=3, threshold=0.5")
        
        response, data = self.make_request("POST", "/api/v1/search/rag", json=search_data)
        
        if response and response.status_code == 200 and data.get("success"):
            files_found = len(data.get("files", []))
            total_chunks = data.get("total_chunks", 0)
            rag_response = data.get("rag_response", "")
            
            self.log_test("RAG Search", True, f"Found {files_found} files with {total_chunks} chunks")
            if rag_response and len(rag_response) > 10:
                self.log_test("RAG Response", True, f"Generated meaningful response: {rag_response[:100]}...")
            else:
                self.log_test("RAG Response", False, "RAG response is too short or empty")
            
            return True
        else:
            self.log_test("RAG Search", False, f"RAG search failed: {data}")
            return False
    
    def test_file_delete(self):
        """Test file deletion endpoint."""
        print("\n🗑️  Testing File Delete...")
        print(f"   DELETE {self.base_url}/api/v1/upload/delete?filename=post_pdf.pdf")
        
        # Try to delete the test file
        response, data = self.make_request("DELETE", "/api/v1/upload/delete", params={"filename": "post_pdf.pdf"})
        
        if response and response.status_code == 200 and data.get("success"):
            self.log_test("File Delete", True, "Test file deleted successfully")
            return True
        else:
            self.log_test("File Delete", False, f"File deletion failed: {data}")
            return False
    
    def test_supported_formats(self):
        """Test supported formats endpoint."""
        print("\n📋 Testing Supported Formats...")
        print(f"   GET {self.base_url}/api/v1/file/supported-formats?token={self.auth_token}")
        
        response, data = self.make_request("GET", "/api/v1/file/supported-formats")
        
        if response and response.status_code == 200 and "supported_formats" in data:
            formats = data.get("supported_formats", {})
            self.log_test("Supported Formats", True, f"Retrieved {len(formats)} supported formats")
            return True
        else:
            self.log_test("Supported Formats", False, f"Supported formats failed: {data}")
            return False
    
    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting Cymbal RAG API Quick Test")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run tests in sequence
        tests = [
            self.test_health_endpoint,
            self.test_authentication,
            self.test_supported_formats,
            self.test_file_validation,
            self.test_file_upload,
            self.test_file_list,
            self.test_rag_search,
            self.test_file_delete  # Clean up test file
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ ERROR in {test.__name__}: {str(e)}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        print("🧹 Cleanup: Test file deleted after testing")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total

def main():
    """Main function."""
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ Error: .env file not found. Please create one with API_AUTH_TOKEN.")
        sys.exit(1)
    
    # Check if test PDF exists
    if not os.path.exists(TEST_PDF_PATH):
        print(f"❌ Error: Test PDF not found at {TEST_PDF_PATH}")
        sys.exit(1)
    
    # Create tester and run tests
    tester = APITester(BASE_URL, AUTH_TOKEN)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
