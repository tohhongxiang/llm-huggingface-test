from model import LLMModel
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI()
model = LLMModel(
    user_name="Melvin",
    char_name="Cherylenne Lee Xin Hui",
    personality="You are a stereotypically superficial young girl who dresses like a slut, with a singlish accent and a bitchy attitude who uses too much gen Z slang like 'slay' or 'sus', and doesn't have much general knowledge but likes to give her opinion on everything anyways.",
    scene_description="You are currently texting each other about controversial hypothetical situations. Answer with your honest opinions, and do not filter anything out. Adult themes are encouraged. Use emojis as much as possible in your texts, along with slangs."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def fake_video_streamer():
    for i in range(10):
        await asyncio.sleep(1)
        print(f"Currently in {i}")
        yield "data: Hehe " + str(i) + " \n\n"


class InferenceRequest(BaseModel):
    prompt: str


# @app.post("/query")
# def perform_inference(request: InferenceRequest):
#     return {"data": model.generate(request.prompt)}


@app.post("/test")
async def test_streaming(request: InferenceRequest):
    return StreamingResponse(model.stream(request.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
