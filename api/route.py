from fastapi import APIRouter, Request
from infer import run_inference

router = APIRouter()

@router.post("/api/infer")
async def infer(request: Request):
    data = await request.json()
    user_input = data.get("input", "")

    result = run_inference(
        user_input=user_input
    )
    return result
