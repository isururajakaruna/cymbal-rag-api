#!/usr/bin/env python3
"""
Test script to verify concurrent request handling in FastAPI.

This script sends multiple concurrent requests to test if the server
can handle them without blocking.
"""

import asyncio
import aiohttp
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8000"
AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "InaqhBh3P0MaJCBQnxF05DsdpWjbESpLJvoa-2tfwxI")

async def make_request(session, endpoint, method="GET", data=None):
    """Make a single request."""
    url = f"{BASE_URL}{endpoint}?token={AUTH_TOKEN}"
    
    start_time = time.time()
    try:
        if method == "GET":
            async with session.get(url) as response:
                result = await response.json()
                status = response.status
        else:
            async with session.post(url, json=data) as response:
                result = await response.json()
                status = response.status
        
        duration = time.time() - start_time
        return {
            "endpoint": endpoint,
            "status": status,
            "duration": duration,
            "success": status == 200
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "endpoint": endpoint,
            "status": "error",
            "duration": duration,
            "success": False,
            "error": str(e)
        }

async def test_concurrent_requests():
    """Test concurrent request handling."""
    print("🚀 Testing Concurrent Request Handling")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        "/health",
        "/api/v1/file/supported-formats",
        "/api/v1/files/list",
        "/api/v1/search/rag"
    ]
    
    # RAG search data
    rag_data = {
        "query": "test query",
        "ktop": 2,
        "threshold": 0.5
    }
    
    # Create session
    async with aiohttp.ClientSession() as session:
        # Test 1: Multiple health checks (should be fast)
        print("\n🔍 Test 1: Multiple Health Checks (5 concurrent)")
        health_tasks = [make_request(session, "/health") for _ in range(5)]
        health_results = await asyncio.gather(*health_tasks)
        
        for result in health_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['endpoint']}: {result['duration']:.2f}s")
        
        # Test 2: Mixed endpoint requests
        print("\n🔍 Test 2: Mixed Endpoint Requests (10 concurrent)")
        mixed_tasks = []
        for i in range(10):
            if i % 4 == 0:
                mixed_tasks.append(make_request(session, "/health"))
            elif i % 4 == 1:
                mixed_tasks.append(make_request(session, "/api/v1/file/supported-formats"))
            elif i % 4 == 2:
                mixed_tasks.append(make_request(session, "/api/v1/files/list"))
            else:
                mixed_tasks.append(make_request(session, "/api/v1/search/rag", "POST", rag_data))
        
        mixed_results = await asyncio.gather(*mixed_tasks)
        
        success_count = 0
        total_duration = 0
        for result in mixed_results:
            status = "✅" if result["success"] else "❌"
            print(f"   {status} {result['endpoint']}: {result['duration']:.2f}s")
            if result["success"]:
                success_count += 1
            total_duration += result["duration"]
        
        # Test 3: Rapid sequential requests
        print("\n🔍 Test 3: Rapid Sequential Requests (5 requests)")
        sequential_results = []
        for i in range(5):
            result = await make_request(session, "/api/v1/files/list")
            sequential_results.append(result)
            status = "✅" if result["success"] else "❌"
            print(f"   {status} Request {i+1}: {result['duration']:.2f}s")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 CONCURRENT REQUEST TEST SUMMARY")
        print("=" * 50)
        
        all_results = health_results + mixed_results + sequential_results
        total_requests = len(all_results)
        successful_requests = sum(1 for r in all_results if r["success"])
        avg_duration = sum(r["duration"] for r in all_results) / total_requests
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {successful_requests}")
        print(f"Success Rate: {(successful_requests/total_requests)*100:.1f}%")
        print(f"Average Duration: {avg_duration:.2f}s")
        
        if successful_requests == total_requests:
            print("🎉 All requests succeeded! Server handles concurrency well.")
        else:
            print("⚠️  Some requests failed. Server may have concurrency issues.")
        
        return successful_requests == total_requests

def main():
    """Main function."""
    print("Testing FastAPI concurrent request handling...")
    print("Make sure the server is running with multiple workers!")
    print("Usage: ./start_server.sh 8000 4  # 4 workers")
    print()
    
    try:
        success = asyncio.run(test_concurrent_requests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
