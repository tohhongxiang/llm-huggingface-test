from model import LLMModel
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

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


class InferenceRequest(BaseModel):
    prompt: str


@app.post("/api/test")
def test(request: InferenceRequest):
    print(request)


def stream_wrapper(prompt: str):
    for token in model.stream(prompt):
        yield f"data: {token}\n\n"


@app.post("/api/query")
async def query_stream(request: InferenceRequest):
    return StreamingResponse(stream_wrapper(request.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
