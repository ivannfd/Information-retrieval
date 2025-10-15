from dataclasses import dataclass
from typing import Dict, List, Any
import uuid

@dataclass
class Document:
    id: str
    fields: Dict[str, str]
    
    @classmethod
    def create(cls, **fields) -> 'Document':
        return cls(str(uuid.uuid4()), fields)
    
    def get_field_text(self, field: str) -> str:
        return self.fields.get(field, "")