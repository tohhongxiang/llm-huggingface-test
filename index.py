from model import LLMModel
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()
# model = LLMModel("Jeremy", "Natsuki")


async def fake_video_streamer():
    for i in range(10):
        await asyncio.sleep(1)
        yield json.dumps({"event_id": i, "data": "some random data"}) + "\n"


# class InferenceRequest(BaseModel):
#     prompt: str


# @app.post("/query")
# def perform_inference(request: InferenceRequest):
#     return {"data": model.generate(request.prompt)}


@app.post("/test")
async def test_streaming():
    return StreamingResponse(fake_video_streamer(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
