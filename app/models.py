from sqlalchemy import Column, Integer, String

from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    emb = Column(String)
    filename = Column(String, default=True)
