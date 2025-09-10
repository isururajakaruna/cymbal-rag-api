#!/usr/bin/env python3
"""
Vector Search Evaluation Script

This script tests the RAG API with both related and unrelated queries
to evaluate the performance of the vector search system and suggest
an optimal threshold value.
"""

import os
import requests
import json
import time
from typing import List, Dict, Any
import statistics
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
TEST_FILE_PATH = "test_data/product_catalog.csv"
TEST_QUERIES = {
    "related": [
        "What is the wireless mouse P001?",
        "Tell me about the mechanical keyboard P002",
        "What electronics products are available?",
        "Show me products from TechCorp supplier",
        "What is the price of the office chair P003?",
        "List all furniture products",
        "What products are in stock?",
        "Tell me about the USB-C hub P005",
        "What is the webcam P007?",
        "Show me products under $50"
    ],
    "unrelated": [
        "What is a banana?",
        "How to cook pasta?",
        "What is the weather today?",
        "Tell me about quantum physics",
        "What is machine learning?",
        "How to play guitar?",
        "What is the capital of France?",
        "Tell me about cooking recipes",
        "What is artificial intelligence?",
        "How to learn Spanish?"
    ]
}

def upload_test_file() -> bool:
    """Upload the test file to the system."""
    print(f"📤 Uploading test file: {TEST_FILE_PATH}")
    
    if not os.path.exists(TEST_FILE_PATH):
        print(f"❌ Test file not found: {TEST_FILE_PATH}")
        return False
    
    url = f"{API_BASE_URL}/upload/direct?token={API_AUTH_TOKEN}"
    
    try:
        with open(TEST_FILE_PATH, 'rb') as f:
            files = {'file': f}
            data = {'replace_existing': 'true', 'tags': 'test,product,catalog'}
            response = requests.post(url, files=files, data=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                print(f"✅ File uploaded successfully: {result.get('filename')}")
                print(f"   Chunks: {result.get('total_chunks')}, Embeddings: {result.get('embeddings_stored')}")
                return True
            else:
                print(f"❌ Upload failed: {result.get('message', 'Unknown error')}")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Error uploading file: {e}")
        return False

def cleanup_test_file() -> bool:
    """Delete the test file from the system."""
    print(f"🗑️  Cleaning up test file...")
    
    filename = os.path.basename(TEST_FILE_PATH)
    url = f"{API_BASE_URL}/upload/delete?token={API_AUTH_TOKEN}"
    params = {'filename': filename}
    
    try:
        response = requests.delete(url, params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result.get("success"):
            print(f"✅ File deleted successfully: {filename}")
            return True
        else:
            print(f"❌ Delete failed: {result.get('message', 'Unknown error')}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error deleting file: {e}")
        return False

def make_rag_request(query: str, ktop: int = 10, threshold: float = 0.8) -> Dict[str, Any]:
    """Make a RAG search request to the API."""
    url = f"{API_BASE_URL}/search/rag?token={API_AUTH_TOKEN}"
    payload = {
        "query": query,
        "ktop": ktop,
        "threshold": threshold
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request for query '{query}': {e}")
        return None

def analyze_search_results(results: Dict[str, Any], query: str, query_type: str) -> Dict[str, Any]:
    """Analyze search results and extract metrics."""
    if not results or not results.get("success"):
        return {
            "query": query,
            "type": query_type,
            "success": False,
            "total_chunks": 0,
            "total_files": 0,
            "avg_distance": None,
            "min_distance": None,
            "max_distance": None,
            "has_content": False
        }
    
    chunks = []
    for file_info in results.get("files", []):
        for chunk in file_info.get("matched_chunks", []):
            chunks.append({
                "distance": chunk.get("distance", 1.0),
                "content": chunk.get("content", ""),
                "filename": chunk.get("filename", "")
            })
    
    if not chunks:
        return {
            "query": query,
            "type": query_type,
            "success": True,
            "total_chunks": 0,
            "total_files": results.get("total_files", 0),
            "top_distance": None,
            "min_distance": None,
            "max_distance": None,
            "has_content": False
        }
    
    distances = [chunk["distance"] for chunk in chunks]
    distances.sort()  # Sort to get top (lowest) distance first
    
    return {
        "query": query,
        "type": query_type,
        "success": True,
        "total_chunks": len(chunks),
        "total_files": results.get("total_files", 0),
        "top_distance": distances[0],  # Top (best) distance
        "min_distance": min(distances),
        "max_distance": max(distances),
        "has_content": any(len(chunk["content"].strip()) > 0 for chunk in chunks),
        "distances": distances
    }

# Removed threshold sensitivity testing as we focus on top results

def create_distance_plot(related_distances: List[float], unrelated_distances: List[float], save_path: str = "test_results/distance_distribution.png"):
    """Create a plot showing the distribution of distances for related vs unrelated queries."""
    print(f"\n📊 Creating distance distribution plot...")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Histogram comparison
    ax1.hist(related_distances, bins=20, alpha=0.7, label='Related Queries', color='green', edgecolor='black')
    ax1.hist(unrelated_distances, bins=20, alpha=0.7, label='Unrelated Queries', color='red', edgecolor='black')
    ax1.set_xlabel('Distance Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distance Distribution: Related vs Unrelated Queries')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    data_to_plot = [related_distances, unrelated_distances]
    labels = ['Related Queries', 'Unrelated Queries']
    box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('green')
    box_plot['boxes'][1].set_facecolor('red')
    ax2.set_ylabel('Distance Score')
    ax2.set_title('Distance Score Distribution (Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    related_mean = statistics.mean(related_distances)
    unrelated_mean = statistics.mean(unrelated_distances)
    related_std = statistics.stdev(related_distances)
    unrelated_std = statistics.stdev(unrelated_distances)
    
    stats_text = f"""Statistics:
Related Queries: μ={related_mean:.3f}, σ={related_std:.3f}
Unrelated Queries: μ={unrelated_mean:.3f}, σ={unrelated_std:.3f}
Recommended Threshold: {(related_mean + unrelated_mean)/2:.3f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plot saved as: {save_path}")
    
    # Also create a simple ASCII plot for command line
    create_ascii_plot(related_distances, unrelated_distances)
    
    return save_path

def create_ascii_plot(related_distances: List[float], unrelated_distances: List[float]):
    """Create a simple ASCII plot for command line display."""
    print(f"\n📊 ASCII Distance Distribution:")
    print("=" * 60)
    
    # Create bins
    all_distances = related_distances + unrelated_distances
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    bin_size = (max_dist - min_dist) / 20
    
    # Count frequencies
    related_bins = [0] * 20
    unrelated_bins = [0] * 20
    
    for dist in related_distances:
        bin_idx = min(int((dist - min_dist) / bin_size), 19)
        related_bins[bin_idx] += 1
    
    for dist in unrelated_distances:
        bin_idx = min(int((dist - min_dist) / bin_size), 19)
        unrelated_bins[bin_idx] += 1
    
    max_freq = max(max(related_bins), max(unrelated_bins))
    
    # Print ASCII histogram
    for i in range(20):
        bin_start = min_dist + i * bin_size
        bin_end = min_dist + (i + 1) * bin_size
        
        related_bar = "█" * int(related_bins[i] * 30 / max_freq)
        unrelated_bar = "▓" * int(unrelated_bins[i] * 30 / max_freq)
        
        print(f"{bin_start:.3f}-{bin_end:.3f}: {related_bar} (R:{related_bins[i]}) {unrelated_bar} (U:{unrelated_bins[i]})")
    
    print("Legend: █ = Related Queries, ▓ = Unrelated Queries")
    print("=" * 60)

def run_comprehensive_evaluation():
    """Run comprehensive evaluation of the vector search system."""
    print("🚀 Starting Vector Search Evaluation")
    print("=" * 60)
    
    # Upload test file first
    if not upload_test_file():
        print("❌ Failed to upload test file. Exiting.")
        return
    
    print("\n⏳ Waiting for file processing...")
    time.sleep(3)  # Give time for processing
    
    all_results = {
        "related": [],
        "unrelated": []
    }
    
    # Test related queries
    print("\n📊 Testing Related Queries")
    print("-" * 40)
    for query in TEST_QUERIES["related"]:
        print(f"\n🔍 Testing: '{query}'")
        response = make_rag_request(query, ktop=10, threshold=0.3)
        analysis = analyze_search_results(response, query, "related")
        all_results["related"].append(analysis)
        
        if analysis["success"] and analysis["top_distance"] is not None:
            print(f"  ✅ Query: '{query}' → Distance: {analysis['top_distance']:.3f} (Chunks: {analysis['total_chunks']})")
        else:
            print(f"  ❌ Query: '{query}' → Failed")
        
        time.sleep(0.5)
    
    # Test unrelated queries
    print("\n📊 Testing Unrelated Queries")
    print("-" * 40)
    for query in TEST_QUERIES["unrelated"]:
        print(f"\n🔍 Testing: '{query}'")
        response = make_rag_request(query, ktop=10, threshold=0.3)
        analysis = analyze_search_results(response, query, "unrelated")
        all_results["unrelated"].append(analysis)
        
        if analysis["success"] and analysis["top_distance"] is not None:
            print(f"  ✅ Query: '{query}' → Distance: {analysis['top_distance']:.3f} (Chunks: {analysis['total_chunks']})")
        else:
            print(f"  ❌ Query: '{query}' → Failed")
        
        time.sleep(0.5)
    
    # Analyze results
    print("\n📈 Analysis Results")
    print("=" * 60)
    
    # Related queries analysis
    related_results = [r for r in all_results["related"] if r["success"] and r["top_distance"] is not None]
    if related_results:
        related_distances = [r["top_distance"] for r in related_results]
        print(f"\n✅ Related Queries ({len(related_results)} successful):")
        print(f"  Average Top Distance: {statistics.mean(related_distances):.3f}")
        print(f"  Min Top Distance: {min(related_distances):.3f}")
        print(f"  Max Top Distance: {max(related_distances):.3f}")
        print(f"  Std Deviation: {statistics.stdev(related_distances):.3f}")
    
    # Unrelated queries analysis
    unrelated_results = [r for r in all_results["unrelated"] if r["success"] and r["top_distance"] is not None]
    if unrelated_results:
        unrelated_distances = [r["top_distance"] for r in unrelated_results]
        print(f"\n❌ Unrelated Queries ({len(unrelated_results)} successful):")
        print(f"  Average Top Distance: {statistics.mean(unrelated_distances):.3f}")
        print(f"  Min Top Distance: {min(unrelated_distances):.3f}")
        print(f"  Max Top Distance: {max(unrelated_distances):.3f}")
        print(f"  Std Deviation: {statistics.stdev(unrelated_distances):.3f}")
    
    # Threshold recommendation
    if related_results and unrelated_results:
        related_avg = statistics.mean(related_distances)
        unrelated_avg = statistics.mean(unrelated_distances)
        
        # Calculate optimal threshold (midpoint with some buffer)
        optimal_threshold = (related_avg + unrelated_avg) / 2
        
        print(f"\n🎯 Threshold Recommendation:")
        print(f"  Related queries avg top distance: {related_avg:.3f}")
        print(f"  Unrelated queries avg top distance: {unrelated_avg:.3f}")
        print(f"  Recommended threshold: {optimal_threshold:.3f}")
        print(f"  Suggested range: {optimal_threshold - 0.1:.3f} - {optimal_threshold + 0.1:.3f}")
    
    # Create distance distribution plots
    if related_results and unrelated_results:
        related_distances = [r["top_distance"] for r in related_results if r["top_distance"] is not None]
        unrelated_distances = [r["top_distance"] for r in unrelated_results if r["top_distance"] is not None]
        
        if related_distances and unrelated_distances:
            create_distance_plot(related_distances, unrelated_distances, "test_results/distance_distribution.png")
    
    # Save detailed results
    with open("test_results/vector_search_evaluation_results.json", "w") as f:
        json.dump({
            "timestamp": time.time(),
            "all_results": all_results,
            "analysis": {
                "related_distances": related_distances if 'related_distances' in locals() else [],
                "unrelated_distances": unrelated_distances if 'unrelated_distances' in locals() else [],
                "optimal_threshold": optimal_threshold if 'optimal_threshold' in locals() else None
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: test_results/vector_search_evaluation_results.json")
    
    # Cleanup test file
    cleanup_test_file()
    
    print("\n🎉 Evaluation Complete!")

if __name__ == "__main__":
    run_comprehensive_evaluation()
