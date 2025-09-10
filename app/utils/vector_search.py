"""
Vector Search service for Vertex AI Matching Engine integration.

This module provides a service for managing vector embeddings in Google Cloud's
Vertex AI Matching Engine, including upsert, search, and delete operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Union

from google.api_core.retry import Retry
from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.services.index_service import IndexServiceClient
from google.cloud.aiplatform_v1.services.match_service import MatchServiceClient
from google.cloud.aiplatform_v1.types import (
    FindNeighborsRequest,
    Index as GCPIndex,
    IndexDatapoint,
    RemoveDatapointsRequest,
    UpsertDatapointsRequest,
)

from app.core.config import settings
from app.core.exceptions import RAGAPIException

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _as_list(v: Union[str, int, float, List[Any], None]) -> List[str]:
    """Convert a value to a list of strings."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def _build_restricts(facets: Optional[Dict[str, Union[str, int, float, List[Any]]]]) -> List[IndexDatapoint.Restriction]:
    """Convert a facets dict into Matching Engine Restriction objects."""
    restricts: List[IndexDatapoint.Restriction] = []
    if not facets:
        return restricts
    for k, v in facets.items():
        values = _as_list(v)
        if not values:
            continue
        restricts.append(IndexDatapoint.Restriction(namespace=str(k), allow_list=values))
    return restricts


class VectorSearchService:
    """
    Vertex AI Matching Engine service for vector search operations.
    
    This service provides methods for:
    - Upserting embeddings to the vector index
    - Searching for similar vectors
    - Removing embeddings by ID or metadata filters
    
    Notes:
        - Only store filterable facets in `restricts`. Keep rich metadata in your own DB keyed by `datapoint_id`.
        - Distances returned are cosine distances; similarity = 1 - distance.
    """

    # Configuration constants
    UPSERT_BATCH_SIZE = 500
    DEFAULT_RETRY = Retry(
        initial=1.0, maximum=30.0, multiplier=2.0, deadline=300.0
    )

    def __init__(self) -> None:
        """Initialize the Vector Search service with configuration from settings."""
        self.project_id = settings.google_cloud_project_id
        self.location = settings.google_cloud_region
        self.index_id = settings.vector_search_index_id
        self.endpoint_id = settings.vector_search_index_endpoint_id
        self.api_endpoint = settings.vector_search_api_endpoint

        # Optional: set in settings; if None we skip dimensionality validation
        self.vector_dims: Optional[int] = getattr(settings, "vector_embedding_dimensions", None)

        self.index_name = f"projects/{self.project_id}/locations/{self.location}/indexes/{self.index_id}"
        self.endpoint_name = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_id}"

        # Initialize low-level clients
        self.index_client = IndexServiceClient(client_options={"api_endpoint": self.api_endpoint})
        self.match_client = MatchServiceClient(client_options={"api_endpoint": self.api_endpoint})

        logger.info("VectorSearchService ready. index=%s endpoint=%s", self.index_name, self.endpoint_name)

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics and metadata about the vector search index."""
        try:
            idx: GCPIndex = self.index_client.get_index(name=self.index_name, retry=self.DEFAULT_RETRY)
            out = {
                "index_id": self.index_id,
                "display_name": idx.display_name,
                "description": idx.description,
                "metadata_schema_uri": idx.metadata_schema_uri,
                "state": str(idx.state),
                "create_time": str(idx.create_time),
                "update_time": str(idx.update_time),
                "etag": idx.etag,
            }
            return out
        except Exception as e:
            logger.exception("Failed to get index stats")
            raise RAGAPIException(f"get_index_stats failed: {e}") from e

    def _validate_dims(self, vector: List[float]) -> None:
        """Validate that the vector has the expected number of dimensions."""
        if self.vector_dims is not None and len(vector) != int(self.vector_dims):
            raise RAGAPIException(
                f"Vector has {len(vector)} dimensions; index expects {self.vector_dims}."
            )

    def upsert_embeddings(self, embeddings: List[Dict[str, Any]]) -> None:
        """
        Upsert datapoints to the vector search index using the high-level API.

        Args:
            embeddings: List of embedding dictionaries, each containing:
                - id: str - Unique identifier for the datapoint
                - embedding: List[float] - Vector embedding
                - metadata: Dict[str, Any] - Optional metadata stored as restricts facets
        """
        if not embeddings:
            return

        try:
            from google.cloud import aiplatform
            from google.cloud.aiplatform import MatchingEngineIndex
            
            # Initialize AI Platform
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Get the index
            index = MatchingEngineIndex(index_name=self.index_name)
            
            def to_datapoint(e: Dict[str, Any]) -> IndexDatapoint:
                dp_id = e["id"]
                vec = e["embedding"]
                self._validate_dims(vec)
                restricts = _build_restricts(e.get("metadata") or {})
                return IndexDatapoint(datapoint_id=dp_id, feature_vector=vec, restricts=restricts)

            for i in range(0, len(embeddings), self.UPSERT_BATCH_SIZE):
                batch = [to_datapoint(e) for e in embeddings[i : i + self.UPSERT_BATCH_SIZE]]
                index.upsert_datapoints(datapoints=batch)
                logger.info("Upserted batch %d..%d (%d)", i, i + len(batch) - 1, len(batch))
        except Exception as e:
            logger.exception("Upsert failed")
            raise RAGAPIException(f"upsert_embeddings failed: {e}") from e

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Union[str, int, float, List[Any]]]] = None,
        return_full_datapoint: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using server-side filtered nearest-neighbor search.

        Args:
            query_embedding: The vector to search with
            top_k: Number of neighbors to return
            filters: Optional dict of filterable facets (translated to server-side restricts)
            return_full_datapoint: Whether to include restricts in response for metadata reconstruction

        Returns:
            List of dictionaries containing:
                - id: str - Datapoint ID
                - similarity: float - Similarity score (1 - distance)
                - distance: float - Cosine distance
                - metadata: Dict[str, Any] - Extracted metadata from restricts
        """
        self._validate_dims(query_embedding)
        restricts = _build_restricts(filters)

        try:
            q = FindNeighborsRequest.Query(
                datapoint=IndexDatapoint(feature_vector=query_embedding),
                neighbor_count=top_k,
            )

            resp = self.match_client.find_neighbors(
                request=FindNeighborsRequest(
                    index_endpoint=self.endpoint_name,
                    deployed_index_id=settings.vector_search_deployed_index_id,
                    queries=[q],
                    return_full_datapoint=return_full_datapoint,
                ),
                retry=self.DEFAULT_RETRY,
            )

            results: List[Dict[str, Any]] = []
            for qr in resp.nearest_neighbors:
                for nb in qr.neighbors:
                    dist = nb.distance
                    meta: Dict[str, Union[str, List[str]]] = {}
                    if return_full_datapoint and nb.datapoint.restricts:
                        for r in nb.datapoint.restricts:
                            # keep list to avoid lossy comma-joining
                            if r.allow_list:
                                meta[r.namespace] = list(r.allow_list)
                    results.append(
                        {
                            "id": nb.datapoint.datapoint_id,
                            "distance": dist,
                            "metadata": meta,
                        }
                    )
            return results
        except Exception as e:
            logger.exception("Search failed")
            raise RAGAPIException(f"search_similar failed: {e}") from e

    def remove_embeddings_by_ids(self, datapoint_ids: Iterable[str]) -> int:
        """
        Remove datapoints by ID using the high-level API.

        Args:
            datapoint_ids: Iterable of datapoint IDs to remove

        Returns:
            Number of datapoints successfully removed
        """
        ids = [str(x) for x in datapoint_ids if str(x)]
        if not ids:
            return 0
        try:
            from google.cloud import aiplatform
            from google.cloud.aiplatform import MatchingEngineIndex
            
            # Initialize AI Platform
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Get the index
            index = MatchingEngineIndex(index_name=self.index_name)
            
            index.remove_datapoints(datapoint_ids=ids)
            logger.info("Removed %d datapoints by ID", len(ids))
            return len(ids)
        except Exception as e:
            logger.exception("Remove by IDs failed")
            raise RAGAPIException(f"remove_embeddings_by_ids failed: {e}") from e

    def remove_embeddings_by_metadata(
        self,
        filters: Dict[str, Union[str, int, float, List[Any]]],
        max_candidates: int = 1000,
        probe_vector: Optional[List[float]] = None,
    ) -> int:
        """
        Remove datapoints that match the given metadata filters.

        This method uses a filtered neighbor query to discover matching datapoint IDs,
        then removes them. This is a best-effort approach since Matching Engine
        doesn't support pure faceted browsing.

        Args:
            filters: Dict of facets to match (server-side restricts)
            max_candidates: Maximum number of candidates to retrieve
            probe_vector: Optional probe vector; if None, uses zero-vector of expected dimensionality

        Returns:
            Number of datapoints successfully removed
        """
        if not filters:
            raise RAGAPIException("remove_embeddings_by_metadata requires non-empty filters")

        # Choose a probe vector
        if probe_vector is None:
            if self.vector_dims is None:
                raise RAGAPIException(
                    "vector_embedding_dimensions is not set; provide probe_vector or set the dimension."
                )
            probe_vector = [0.0] * int(self.vector_dims)
        self._validate_dims(probe_vector)

        # Search with server-side restricts to gather candidate IDs
        try:
            q = FindNeighborsRequest.Query(
                datapoint=IndexDatapoint(feature_vector=probe_vector),
                neighbor_count=max_candidates,
                restricts=_build_restricts(filters),
            )
            resp = self.match_client.find_neighbors(
                request=FindNeighborsRequest(
                    index_endpoint=self.endpoint_name,
                    deployed_index_id=settings.vector_search_deployed_index_id,
                    queries=[q],
                    return_full_datapoint=False,
                ),
                retry=self.DEFAULT_RETRY,
            )

            ids: List[str] = []
            for qr in resp.nearest_neighbors:
                for nb in qr.neighbors:
                    ids.append(nb.datapoint.datapoint_id)

            if not ids:
                logger.info("No datapoints matched filters; nothing to remove.")
                return 0

            req = RemoveDatapointsRequest(index=self.index_name, datapoint_ids=ids)
            self.index_client.remove_datapoints(request=req, retry=self.DEFAULT_RETRY)
            logger.info("Removed %d datapoints by metadata filters=%s", len(ids), filters)
            return len(ids)

        except Exception as e:
            logger.exception("Remove by metadata failed")
            raise RAGAPIException(f"remove_embeddings_by_metadata failed: {e}") from e

    def remove_embeddings_by_filename(self, filename: str) -> int:
        """
        Remove all embeddings associated with a specific filename.

        Args:
            filename: The filename to remove embeddings for

        Returns:
            Number of datapoints successfully removed
        """
        return self.remove_embeddings_by_metadata(filters={"filename": filename})
