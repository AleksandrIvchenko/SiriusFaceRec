from pydantic import BaseModel


class UserBase(BaseModel):
    name: str
    emb: str
    filename: str


class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int

    class Config:
        orm_mode = True
