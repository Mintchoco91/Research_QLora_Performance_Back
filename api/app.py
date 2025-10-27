from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from route import router as infer_router

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우트 등록
app.include_router(infer_router)

# 실행 명령: uvicorn app:app --host 0.0.0.0 --port 8000
