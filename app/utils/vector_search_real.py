"""Real Vector Search service using Google Cloud AI Platform."""

import os
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1 import UpsertDatapointsRequest, UpsertDatapoint, IndexDatapoint
from google.cloud.aiplatform_v1 import SearchRequest, SearchResponse
from google.cloud.aiplatform_v1 import DeleteDatapointsRequest
from google.cloud.aiplatform_v1.services.match_service import MatchServiceClient
from google.cloud.aiplatform_v1.services.index_service import IndexServiceClient
from google.cloud.aiplatform_v1.types import FindNeighborsRequest, FindNeighborsResponse
import asyncio
import json

from app.core.config import settings
from app.core.exceptions import RAGAPIException


class VectorSearchService:
    """Real Vector Search service using Google Cloud AI Platform."""
    
    def __init__(self):
        """Initialize the Vector Search service."""
        self.project_id = settings.google_cloud_project_id
        self.location = settings.google_cloud_region
        self.index_id = settings.vector_search_index_id
        self.endpoint_id = settings.vector_search_index_endpoint_id
        
        # Initialize AI Platform
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Initialize clients
        self.match_client = MatchServiceClient()
        self.index_client = IndexServiceClient()
        
        # Get index and endpoint resource names
        self.index_name = f"projects/{self.project_id}/locations/{self.location}/indexes/{self.index_id}"
        self.endpoint_name = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_id}"
        
        print(f"VectorSearchService initialized for project {self.project_id}")
        print(f"Index: {self.index_name}")
        print(f"Endpoint: {self.endpoint_name}")
    
    async def upsert_embeddings(
        self, 
        embeddings: List[Dict[str, Any]], 
        index_id: str, 
        endpoint_id: str
    ) -> bool:
        """
        Upsert embeddings to Vector Search.
        
        Args:
            embeddings: List of embedding dictionaries with id, content, metadata, and embedding
            index_id: Vector Search index ID
            endpoint_id: Vector Search endpoint ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Upserting {len(embeddings)} embeddings to Vector Search")
            
            # Prepare datapoints for upsert
            datapoints = []
            for embedding_data in embeddings:
                datapoint = UpsertDatapoint(
                    datapoint_id=embedding_data["id"],
                    feature_vector=embedding_data["embedding"],
                    restricts=[],  # No restrictions for now
                    numeric_restricts=[]  # No numeric restrictions for now
                )
                datapoints.append(datapoint)
            
            # Create upsert request
            request = UpsertDatapointsRequest(
                index=self.index_name,
                datapoints=datapoints
            )
            
            # Execute upsert
            response = self.index_client.upsert_datapoints(request=request)
            
            print(f"Successfully upserted {len(embeddings)} datapoints to Vector Search")
            return True
            
        except Exception as e:
            print(f"Error upserting embeddings to Vector Search: {e}")
            return False
    
    async def search_similar(
        self,
        query_embedding: List[float],
        index_id: str,
        endpoint_id: str,
        top_k: int = 5,
        filter_expression: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in Vector Search.
        
        Args:
            query_embedding: Query embedding vector
            index_id: Vector Search index ID
            endpoint_id: Vector Search endpoint ID
            top_k: Number of results to return
            filter_expression: Optional filter expression
            
        Returns:
            List of search results
        """
        try:
            print(f"Searching Vector Search with {len(query_embedding)}-dim vector")
            print(f"  Top K: {top_k}")
            print(f"  Filter: {filter_expression}")
            
            # Create search request
            request = SearchRequest(
                deployed_index_id=endpoint_id,
                queries=[{
                    "datapoint": {
                        "feature_vector": query_embedding
                    },
                    "neighbor_count": top_k
                }],
                return_full_datapoint=True
            )
            
            # Execute search
            response = self.match_client.search(
                index_endpoint=self.endpoint_name,
                deployed_index_id=endpoint_id,
                queries=[{
                    "datapoint": {
                        "feature_vector": query_embedding
                    },
                    "neighbor_count": top_k
                }],
                return_full_datapoint=True
            )
            
            # Process results
            results = []
            for query_result in response.nearest_neighbors:
                for neighbor in query_result.neighbors:
                    result = {
                        "id": neighbor.datapoint.datapoint_id,
                        "score": neighbor.distance,
                        "metadata": {}
                    }
                    
                    # Extract metadata if available
                    if hasattr(neighbor.datapoint, 'metadata') and neighbor.datapoint.metadata:
                        result["metadata"] = dict(neighbor.datapoint.metadata)
                    
                    results.append(result)
            
            print(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            print(f"Error searching Vector Search: {e}")
            return []
    
    async def remove_embeddings_by_metadata(
        self,
        metadata_filter: Dict[str, str],
        index_id: str,
        endpoint_id: str
    ) -> bool:
        """
        Remove embeddings by metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata filters
            index_id: Vector Search index ID
            endpoint_id: Vector Search endpoint ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Removing embeddings with filter: {metadata_filter}")
            print(f"  Index ID: {index_id}")
            print(f"  Endpoint ID: {endpoint_id}")
            
            # First, search for embeddings matching the filter
            # Note: This is a simplified approach. In production, you might want to
            # maintain a separate metadata index or use more sophisticated filtering
            
            # For now, we'll use a broad search and filter client-side
            # This is not ideal for large datasets but works for MVP
            
            # Create a search request to find all datapoints
            # We'll use a dummy query to get all datapoints
            dummy_query = [0.0] * 768  # Assuming 768-dimensional embeddings
            
            search_request = SearchRequest(
                deployed_index_id=endpoint_id,
                queries=[{
                    "datapoint": {
                        "feature_vector": dummy_query
                    },
                    "neighbor_count": 1000  # Large number to get many results
                }],
                return_full_datapoint=True
            )
            
            # Execute search
            response = self.match_client.search(
                index_endpoint=self.endpoint_name,
                deployed_index_id=endpoint_id,
                queries=[{
                    "datapoint": {
                        "feature_vector": dummy_query
                    },
                    "neighbor_count": 1000
                }],
                return_full_datapoint=True
            )
            
            # Find datapoints matching the filter
            datapoints_to_remove = []
            for query_result in response.nearest_neighbors:
                for neighbor in query_result.neighbors:
                    datapoint = neighbor.datapoint
                    
                    # Check if metadata matches filter
                    if hasattr(datapoint, 'metadata') and datapoint.metadata:
                        metadata = dict(datapoint.metadata)
                        matches = True
                        for key, value in metadata_filter.items():
                            if metadata.get(key) != value:
                                matches = False
                                break
                        
                        if matches:
                            datapoints_to_remove.append(datapoint.datapoint_id)
            
            if not datapoints_to_remove:
                print("No embeddings found matching the filter")
                return True
            
            # Remove the matching datapoints
            delete_request = DeleteDatapointsRequest(
                index=self.index_name,
                datapoint_ids=datapoints_to_remove
            )
            
            response = self.index_client.delete_datapoints(request=delete_request)
            
            print(f"Successfully removed {len(datapoints_to_remove)} embeddings")
            return True
            
        except Exception as e:
            print(f"Error removing embeddings from Vector Search: {e}")
            return False
    
    async def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """
        Get statistics about the Vector Search index.
        
        Args:
            index_id: Vector Search index ID
            
        Returns:
            Dictionary with index statistics
        """
        try:
            # Get index information
            index = self.index_client.get_index(name=self.index_name)
            
            stats = {
                "index_id": index_id,
                "display_name": index.display_name,
                "description": index.description,
                "metadata_schema_uri": index.metadata_schema_uri,
                "state": index.state.name,
                "create_time": index.create_time,
                "update_time": index.update_time,
                "etag": index.etag
            }
            
            print(f"Index stats retrieved for {index_id}")
            return stats
            
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {"error": str(e)}
