# Setup

```sh
git clone https://github.com/tohhongxiang/llm-huggingface-test
cd llm-huggingface-test

python -m venv venv
venv/Scripts/activate

pip install transformers bitsandbytes accelerate peft trl fastapi pydantic
```

Install `pytorch` by following the settings [here](https://pytorch.org/get-started/locally/). For windows and CUDA 12.1:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Resources

- https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146
- https://huggingface.co/SanjiWatsuki/Silicon-Maid-7B
- https://medium.com/@hxu296/serving-openai-stream-with-fastapi-and-consuming-with-react-js-part-1-8d482eb89702