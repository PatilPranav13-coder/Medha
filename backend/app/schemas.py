from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

Mode = Literal["code", "research", "architecture", "engineering"]
BuildTarget = Literal["auto", "apple", "android", "windows", "cross_platform"]


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=200)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class UserView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: EmailStr
    created_at: datetime


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=4000)
    instructions: str = Field(default="", max_length=8000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=4000)
    instructions: str | None = Field(default=None, max_length=8000)


class ProjectView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str
    instructions: str
    created_at: datetime
    updated_at: datetime


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=180)
    project_id: str | None = None
    mode: Mode = "code"
    model_name: str | None = None


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=180)


class ConversationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    project_id: str | None
    title: str
    mode: str
    model_name: str
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    mode: Mode | None = None
    model_name: str | None = None
    web_search: bool = True
    build_target: BuildTarget = "auto"


class MessageEdit(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


class ResendOptions(BaseModel):
    mode: Mode | None = None
    model_name: str | None = None
    web_search: bool = True
    build_target: BuildTarget = "auto"


class MessageView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    model_name: str
    input_tokens: int | None
    output_tokens: int | None
    sources: list[dict[str, Any]]
    created_at: datetime


class DocumentView(BaseModel):
    id: str
    filename: str
    extension: str
    size_bytes: int
    created_at: datetime
    chunk_count: int = 0


class MemoryCreate(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=4000)
    project_id: str | None = None


class MemoryView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    project_id: str | None
    label: str
    content: str
    created_at: datetime
