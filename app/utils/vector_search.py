"""Real Vector Search service using Google Cloud AI Platform."""

import os
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1.types import IndexDatapoint
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
            
            # Get the index
            index = MatchingEngineIndex(index_name=self.index_name)
            
            # Prepare datapoints for upsert
            datapoints = []
            for embedding_data in embeddings:
                datapoint = IndexDatapoint(
                    datapoint_id=embedding_data["id"],
                    feature_vector=embedding_data["embedding"],
                    restricts=[],  # No restrictions for now
                    numeric_restricts=[]  # No numeric restrictions for now
                )
                datapoints.append(datapoint)
            
            # Upsert datapoints
            index.upsert_datapoints(datapoints=datapoints)
            
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
            
            # Get the endpoint
            endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=self.endpoint_name)
            
            # Execute search
            response = endpoint.find_neighbors(
                deployed_index_id=endpoint_id,
                queries=[query_embedding],
                num_neighbors=top_k,
                return_full_datapoint=True
            )
            
            # Process results
            results = []
            for query_result in response:
                for neighbor in query_result:
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
            
            # Get the index
            index = MatchingEngineIndex(index_name=self.index_name)
            
            # Since Vector Search doesn't support direct metadata filtering for deletion,
            # we need to search for the embeddings first, then delete them by ID
            
            # First, let's search for embeddings with the filename metadata
            filename = metadata_filter.get("filename", "")
            if not filename:
                print("No filename provided in metadata filter")
                return False
            
            # Clean the filename to match the format used in upload
            clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
            
            # Search for embeddings with this filename
            # We'll use a dummy query to get all embeddings, then filter by metadata
            dummy_query = [0.0] * 3072  # 3072-dimensional dummy query
            
            # Get the endpoint for searching
            endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=self.endpoint_name)
            
            # Search for a large number of embeddings to find the ones we want
            response = endpoint.find_neighbors(
                deployed_index_id=endpoint_id,
                queries=[dummy_query],
                num_neighbors=1000,  # Large number to get many results
                return_full_datapoint=True
            )
            
            # Find datapoints matching the filename filter
            datapoints_to_remove = []
            for query_result in response:
                for neighbor in query_result:
                    datapoint = neighbor.datapoint
                    
                    # Check if metadata matches the filter
                    if hasattr(datapoint, 'metadata') and datapoint.metadata:
                        metadata = dict(datapoint.metadata)
                        matches = True
                        for key, value in metadata_filter.items():
                            if metadata.get(key) != value:
                                matches = False
                                break
                        
                        if matches:
                            datapoints_to_remove.append(datapoint.datapoint_id)
                            print(f"Found matching datapoint: {datapoint.datapoint_id}")
            
            if not datapoints_to_remove:
                print("No embeddings found matching the filter")
                return True
            
            # Remove the matching datapoints
            index.remove_datapoints(datapoint_ids=datapoints_to_remove)
            
            print(f"Successfully removed {len(datapoints_to_remove)} embeddings")
            return True
            
        except Exception as e:
            print(f"Error removing embeddings from Vector Search: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def remove_embeddings_by_ids(
        self,
        datapoint_ids: List[str],
        index_id: str,
        endpoint_id: str
    ) -> bool:
        """
        Remove embeddings by datapoint IDs.
        
        Args:
            datapoint_ids: List of datapoint IDs to remove
            index_id: Vector Search index ID
            endpoint_id: Vector Search endpoint ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Removing embeddings by IDs: {datapoint_ids[:3]}...")
            print(f"  Index ID: {index_id}")
            print(f"  Endpoint ID: {endpoint_id}")
            
            # Get the index
            index = MatchingEngineIndex(index_name=self.index_name)
            
            # Remove the datapoints
            index.remove_datapoints(datapoint_ids=datapoint_ids)
            
            print(f"Successfully removed {len(datapoint_ids)} embeddings")
            return True
            
        except Exception as e:
            print(f"Error removing embeddings by IDs from Vector Search: {e}")
            import traceback
            traceback.print_exc()
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
            index = MatchingEngineIndex(index_name=self.index_name)
            
            stats = {
                "index_id": index_id,
                "display_name": index.display_name,
                "description": index.description,
                "metadata_schema_uri": index.metadata_schema_uri,
                "state": str(index.state),
                "create_time": str(index.create_time),
                "update_time": str(index.update_time),
                "etag": index.etag
            }
            
            print(f"Index stats retrieved for {index_id}")
            return stats
            
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {"error": str(e)}