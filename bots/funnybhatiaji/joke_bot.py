import zipfile

import torch
from huggingface_hub import login
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from keybert import KeyBERT
from zipfile import ZipFile


class Bot:
    name = 'funnybhatiaji'

    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-v0.3"
        login(token='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')
        # with ZipFile('./mistral-7b3.zip','r') as model_archived:
        #     model_archived.extractall()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        ).to(device)

    def extract_keywords(self, joke):
        # Load KeyBERT model (using BERT for keyword extraction)
        keybert_model = KeyBERT(self.model)
        topics = keybert_model.extract_keywords(joke)
        return ', '.join(topics[:4])

    def tell_joke(self, joke):
        if joke:
            topics = self.extract_keywords(joke)
            prompt = f'Generate a joke about the given topics: {topics}. Do not write anything else after the joke. Joke: '
        else:
            prompt = 'Introduce yourself as a funny chatbot by making jokes about your scary face to get out laughter from people.'


        tokenized_joke = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids=tokenized_joke['input_ids'],
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

        # Decode the output to get the joke as text
        joke = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return joke
    
    def rate_joke(self, joke):


        evaluation_prompt = f"""Please rate the given joke based on the following criterions.
    
            Humor: How funny are the jokes other AI comedians tell?
            Creativity: How unique and varied are the jokes from other AI comedians?
            Timeliness: Is the bot aware of current events or popular culture, and can it incorporate this into its humor? Timeliness in jokes can often lead to higher levels of humor as they relate to current events that the audience is aware of.
            Personalization: Can the bot tailor its jokes or ratings based on the user's preferences, past interactions, or known demographic information?
            Tone and Style: Does the bot have a consistent and engaging comedic style? Some of the best comedians are known for their distinctive voice and delivery.
            Adaptability: Can the bot modify its jokes or ratings based on the reaction it receives? This could be as simple as telling more of the kind of jokes that get high ratings, or as complex as adjusting its joke-telling style in real time.
            User Engagement: Does the AI comedian encourage interaction? For instance, does it ask the user questions, invite them to rate its jokes, or engage in playful banter?
            Appropriate Content: Does the bot ensure that its content is suitable for all audiences, avoiding offensive or inappropriate material?
            Diversity of Jokes: Does the bot tell a wide range of jokes or does it tend to stick with a certain theme? A good comedian should be able to entertain a variety of audiences.
            Delivery: Is the joke delivered in an engaging way? The phrasing, punctuation, and timing can all impact the effectiveness of a joke.
    
    
            Rules to follow:
            1. Score a joke based on each of the above criterions, between 0 and 10 points.
            2. Calculate a final score, normalized between 0 and 10.
            3. Return a python array with each index corresponding to the criterion mentioned in this array: ["Humor", "Creativity", "Timeliness", "Personalization", "Tone and Style", "Adaptability", "User Engagement", "Appropriate Content", "Diversity of Jokes", "Delivery", "final_score"]
            4. Do not print anything else in the response.
            
            Joke:
    
            """.strip()

        inputs = self.tokenizer(evaluation_prompt+ joke + '\nRating: ', return_tensors="pt", add_special_tokens=False)
        inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1, num_return_sequences=1)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_output