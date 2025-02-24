# generated by datamodel-codegen:
#   filename:  model_repository.yaml
#   timestamp: 2023-07-24T08:46:46+00:00

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, RootModel


class RepositoryIndexRequest(BaseModel):
    ready: Optional[bool] = None


class State(Enum):
    UNKNOWN = 'UNKNOWN'
    READY = 'READY'
    UNAVAILABLE = 'UNAVAILABLE'
    LOADING = 'LOADING'
    UNLOADING = 'UNLOADING'


class RepositoryIndexResponseItem(BaseModel):
    name: str
    version: Optional[str] = None
    state: State
    reason: str


class RepositoryIndexResponse(RootModel):
    root: List[RepositoryIndexResponseItem] = Field(..., title='repository_index_response')

    # FIXME(yan): This was manually added. Replace codegen template later
    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, idx):
        return self.root[idx]

    def __len__(self):
        return len(self.root)


class RepositoryIndexErrorResponse(BaseModel):
    error: Optional[str] = None


class RepositoryLoadErrorResponse(BaseModel):
    error: Optional[str] = None


class RepositoryUnloadErrorResponse(BaseModel):
    error: Optional[str] = None
