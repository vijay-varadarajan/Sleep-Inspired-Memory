"""
Schema Store

Manages abstract knowledge schemas extracted from consolidated memories.
Inspired by semantic memory and abstract knowledge representation.

Design Principles:
- Schemas represent patterns, concepts, and relationships
- Formed by identifying commonalities across multiple consolidated memories
- Enable generalization and transfer learning
- Support hierarchical organization of knowledge
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from uuid import uuid4
import json


@dataclass
class Schema:
    """
    A schema representing abstract knowledge patterns.
    
    Schemas are induced from multiple consolidated memories that share
    common themes, concepts, or relationships.
    
    Attributes:
        id: Unique identifier
        timestamp: When this schema was created
        name: Short descriptive name for the schema
        description: Detailed description of the pattern/concept
        core_concepts: Key concepts that define this schema
        related_memory_ids: Consolidated memory IDs that support this schema
        examples: Specific examples demonstrating the schema
        confidence: Confidence in this schema (based on supporting evidence)
        access_count: Number of times accessed
        last_access: Timestamp of last access
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    name: str = ""
    description: str = ""
    core_concepts: List[str] = field(default_factory=list)
    related_memory_ids: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.5
    access_count: int = 0
    last_access: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'name': self.name,
            'description': self.description,
            'core_concepts': self.core_concepts,
            'related_memory_ids': self.related_memory_ids,
            'examples': self.examples,
            'confidence': self.confidence,
            'access_count': self.access_count,
            'last_access': self.last_access.isoformat() if self.last_access else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Schema':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_access'):
            data['last_access'] = datetime.fromisoformat(data['last_access'])
        return cls(**data)


class SchemaStore:
    """
    Storage and management system for knowledge schemas.
    
    Schemas represent abstracted, generalized knowledge that emerges
    from patterns across multiple experiences and consolidated memories.
    
    Key Operations:
    - add_schema: Create a new schema
    - update_schema: Add supporting evidence to existing schema
    - find_related_schemas: Find schemas related to given concepts
    - merge_schemas: Combine overlapping schemas
    """
    
    def __init__(self):
        self.schemas: Dict[str, Schema] = {}
    
    def add_schema(
        self,
        name: str,
        description: str,
        core_concepts: List[str],
        related_memory_ids: List[str],
        examples: Optional[List[str]] = None,
        confidence: float = 0.5
    ) -> Schema:
        """
        Add a new schema.
        
        Args:
            name: Short name for the schema
            description: Detailed description
            core_concepts: Key concepts defining the schema
            related_memory_ids: Supporting consolidated memory IDs
            examples: Specific examples
            confidence: Initial confidence score (0-1)
            
        Returns:
            The created Schema object
        """
        schema = Schema(
            name=name,
            description=description,
            core_concepts=core_concepts,
            related_memory_ids=related_memory_ids,
            examples=examples or [],
            confidence=confidence
        )
        self.schemas[schema.id] = schema
        return schema
    
    def get_schema(self, schema_id: str) -> Optional[Schema]:
        """Retrieve a specific schema by ID."""
        return self.schemas.get(schema_id)
    
    def get_all_schemas(self) -> List[Schema]:
        """Get all schemas sorted by confidence (highest first)."""
        return sorted(
            self.schemas.values(),
            key=lambda s: s.confidence,
            reverse=True
        )
    
    def find_by_concepts(self, concepts: List[str], min_overlap: int = 1) -> List[Schema]:
        """
        Find schemas related to given concepts.
        
        Args:
            concepts: List of concepts to search for
            min_overlap: Minimum number of concept overlaps required
            
        Returns:
            List of schemas with sufficient concept overlap
        """
        matching_schemas = []
        
        for schema in self.schemas.values():
            overlap = len(set(concepts) & set(schema.core_concepts))
            if overlap >= min_overlap:
                matching_schemas.append((schema, overlap))
        
        # Sort by overlap count (most relevant first)
        matching_schemas.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in matching_schemas]
    
    def update_schema(
        self,
        schema_id: str,
        new_memory_ids: Optional[List[str]] = None,
        new_examples: Optional[List[str]] = None,
        confidence_boost: float = 0.0
    ) -> bool:
        """
        Update an existing schema with new supporting evidence.
        
        Args:
            schema_id: ID of schema to update
            new_memory_ids: Additional memory IDs that support this schema
            new_examples: Additional examples
            confidence_boost: Amount to increase confidence (can be negative)
            
        Returns:
            True if update succeeded, False if schema not found
        """
        if schema_id not in self.schemas:
            return False
        
        schema = self.schemas[schema_id]
        
        if new_memory_ids:
            schema.related_memory_ids.extend(new_memory_ids)
            # Remove duplicates
            schema.related_memory_ids = list(set(schema.related_memory_ids))
        
        if new_examples:
            schema.examples.extend(new_examples)
        
        # Update confidence (clamp to [0, 1])
        schema.confidence = max(0.0, min(1.0, schema.confidence + confidence_boost))
        
        return True
    
    def find_similar_schemas(self, schema: Schema, threshold: float = 0.3) -> List[Schema]:
        """
        Find schemas similar to a given schema based on concept overlap.
        
        Args:
            schema: Reference schema
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar schemas
        """
        similar = []
        
        for other_schema in self.schemas.values():
            if other_schema.id == schema.id:
                continue
            
            # Calculate Jaccard similarity of core concepts
            concepts_a = set(schema.core_concepts)
            concepts_b = set(other_schema.core_concepts)
            
            if not concepts_a or not concepts_b:
                continue
            
            intersection = len(concepts_a & concepts_b)
            union = len(concepts_a | concepts_b)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= threshold:
                similar.append((other_schema, similarity))
        
        # Sort by similarity (most similar first)
        similar.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in similar]
    
    def merge_schemas(self, schema_id_1: str, schema_id_2: str, new_name: str) -> Optional[Schema]:
        """
        Merge two similar schemas into one.
        
        Args:
            schema_id_1: First schema ID
            schema_id_2: Second schema ID
            new_name: Name for the merged schema
            
        Returns:
            The merged schema, or None if either schema not found
        """
        schema1 = self.get_schema(schema_id_1)
        schema2 = self.get_schema(schema_id_2)
        
        if not schema1 or not schema2:
            return None
        
        # Combine concepts (union)
        merged_concepts = list(set(schema1.core_concepts) | set(schema2.core_concepts))
        
        # Combine memory IDs
        merged_memory_ids = list(set(schema1.related_memory_ids) | set(schema2.related_memory_ids))
        
        # Combine examples
        merged_examples = schema1.examples + schema2.examples
        
        # Average confidence
        merged_confidence = (schema1.confidence + schema2.confidence) / 2
        
        # Create description mentioning merger
        merged_description = f"Merged schema combining:\n1. {schema1.name}: {schema1.description}\n2. {schema2.name}: {schema2.description}"
        
        # Create new merged schema
        merged = self.add_schema(
            name=new_name,
            description=merged_description,
            core_concepts=merged_concepts,
            related_memory_ids=merged_memory_ids,
            examples=merged_examples,
            confidence=merged_confidence
        )
        
        # Remove old schemas
        del self.schemas[schema_id_1]
        del self.schemas[schema_id_2]
        
        return merged
    
    def mark_accessed(self, schema_id: str) -> None:
        """Mark a schema as accessed."""
        if schema_id in self.schemas:
            schema = self.schemas[schema_id]
            schema.access_count += 1
            schema.last_access = datetime.now()
    
    def save_to_file(self, filepath: str) -> None:
        """Serialize schemas to JSON file."""
        data = {
            'schemas': [s.to_dict() for s in self.schemas.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load schemas from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.schemas = {}
        for schema_data in data.get('schemas', []):
            schema = Schema.from_dict(schema_data)
            self.schemas[schema.id] = schema
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the schema store."""
        schemas_list = list(self.schemas.values())
        return {
            'total_schemas': len(schemas_list),
            'avg_confidence': sum(s.confidence for s in schemas_list) / len(schemas_list) if schemas_list else 0,
            'total_accesses': sum(s.access_count for s in schemas_list),
            'avg_supporting_memories': sum(len(s.related_memory_ids) for s in schemas_list) / len(schemas_list) if schemas_list else 0
        }
