from fastapi import FastAPI
from receive import ws
api = FastAPI()
api.include_router(ws)
