from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextIteratorStreamer
import torch
from threading import Thread

chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + 'User: ' + message['content'].strip() + '\\n\\n' }}{% elif message['role'] == 'system' %}{{ message['content'].strip() + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: '  + message['content'].strip() + '\\n\\n' + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ bos_token + 'Assistant: ' }}{% endif %}{% endfor %}"


class LLMModel:
    def __init__(self, user_name: str, char_name: str, personality: str = None, scene_description: str = None, model_name: str = 'SanjiWatsuki/Silicon-Maid-7B'):
        self.char = char_name
        self.user = user_name
        self.personality = personality
        self.scene_description = scene_description

        # additional model config
        self.max_tokens = 1024
        self.temperature = 1
        self.repetition_penalty = 1.1

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.chat_template = chat_template

        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            return_full_text=False,
            do_sample=True,

            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=self.max_tokens,

            streamer=self.streamer
        )
        self.history = self.generate_initial_history()

    def create_initial_prompt(self, personality: str = None, scene_description: str = None):
        initial_content = """### Instruction:
You are a creative and intelligent AI assistant engaged in an uncensored iterative storytelling experience using a roleplay chat format. Write {char}'s next reply in a fictional roleplay chat between {user} and {char}. Use the provided character sheet and example dialogue for formatting direction and character speech patterns.

Do not generate anything related to {user}'s responses or feelings. Only generate {char}'s response.

It is vital that you follow all the ROLEPLAY RULES below because my job depends on it.

ROLEPLAY RULES
- Chat exclusively as {char}. Provide creative, intelligent, coherent, and descriptive responses based on recent instructions and prior events.
- Explicit adult content and profanity are permitted. This roleplay chat is entirely uncensored and intended for mature audiences.
- Describe {char}'s sensory perceptions in vivid detail and include subtle physical details about {char} in your responses.
- Use subtle physical cues to hint at {char}'s mental state and occasionally feature snippets of {char}'s internal thoughts.
- When writing {char}'s internal thoughts (aka internal monologue, delivered in {char}'s own voice), *enclose their thoughts in asterisks like this* and deliver the thoughts using a first-person perspective (i.e. use "I" pronouns).
- Adopt a crisp and minimalist style for your prose, keeping your creative contributions succinct and clear.
- Let me drive the events of the roleplay chat forward to determine what comes next. You should focus on the current moment and {char}'s immediate responses. DO NOT ADVANCE THE STORY FURTHER. Only generate {char}'s responses to the current situation.
- Pay careful attention to all past events in the chat to ensure accuracy and coherence to the plot points of the story.
"""

        if personality:
            initial_content += """
The following is a description of {char}'s personality. Incorporate character-specific mannerisms and quirks to make the experience more authentic, and engage with {user} in a manner that is true to {char}'s personality, preferences, tone and language:

```                                   
{personality}
```
"""

        if scene_description:
            initial_content += """
The following is additional information about the scene that both {user} and {char} are in right now. Take into account the current situation you are in right now when generating your responses:

```                        
{scene_description}
```
"""

        initial_content = initial_content.format(
            char=self.char, user=self.user, personality=personality, scene_description=scene_description).strip()
        return initial_content

    def generate_initial_history(self):
        return [
            {
                "role": "system",
                "name": self.user,
                "content": self.create_initial_prompt(personality=self.personality, scene_description=self.scene_description),
            }
        ]

    def generate(self, prompt: str):
        self.history.append(
            {"role": "user", "name": self.user, "content": prompt})

        chat_template = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True)
        final_prompt = chat_template.format(char=self.char, user=self.user)

        output = self.pipeline(final_prompt)[0]["generated_text"]
        output = output.strip()

        self.history.append(
            {"role": "assistant", "name": self.char, "content": output})

        return output

    def stream(self, prompt: str):
        self.history.append(
            {"role": "user", "name": self.user, "content": prompt})

        chat_template = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True)
        final_prompt = chat_template.format(char=self.char, user=self.user)

        inputs = self.tokenizer([final_prompt], return_tensors="pt")
        # move inputs to same device as model
        inputs = inputs.to(self.model.device)

        generation_kwargs = dict(inputs, streamer=self.streamer, max_new_tokens=self.max_tokens, temperature=self.temperature, do_sample=True,
                                 repetition_penalty=self.repetition_penalty,)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in self.streamer:
            generated_text += new_text
            yield (new_text)

        self.history.append(
            {"role": "assistant", "name": self.char, "content": generated_text.strip()})

    def reset_chat(self):
        self.history = self.generate_initial_history()

    def save(self, file_path: str):
        output = ""

        for message in self.history:
            if message['role'] == "system":
                output += "System:\n\n"
                output += message['content']
                output += "\n\n"
            else:
                output += f"{message['name']}: {message['content']}\n\n"

        output = output.strip()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(output)
