from huggingface_hub import login, InferenceClient
import json


class Bot:
    name = 'funnybhatiaji'

    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        login(token='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')

        self.client = InferenceClient(
            self.model_name,
            token="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",
        )

    def extract_keywords(self, joke):

        chat_topics = [{'role': 'user',
                        'content': f'''Extract the keywords from the given joke: {joke}. 
                 Return different topics that joke contains.
                 '''},
               ]
        response_format = {'type': 'json', 'value':
            {
                'properties':
                    {
                        'Topic1': {'type': 'string'},
                        'Topic2': {'type': 'string'},
                        'Topic3': {'type': 'string'},
                        'Topic4': {'type': 'string'},
                    },
                'required': ['Topic1', 'Topic2', 'Topic3', 'Topic4']}
           }
        keywords_object = self.client.chat_completion(chat_topics, response_format=response_format, max_tokens=300)
        keywords_str = keywords_object['choices'][0]['message']['content']
        keywords_str_dict = keywords_str.replace('\n','')
        keywords_json = json.loads(keywords_str_dict)
        keywords_final = []
        for key,value in keywords_json.items():
            keywords_final.append(value)
        keywords = ','.join(keywords_final)
        return keywords

    def tell_joke(self, joke=None):
        if joke:
            topics = self.extract_keywords(joke)
            prompt = f'''
            Generate a joke about the given topics: {topics}. 
            Do not write anything else after the joke. 
            '''.strip()
        else:
            prompt = '''
            Introduce yourself as Funnybhatiaji, an Indian english 
            speaking standup comedian. Finish the introduction with a funny english
            joke which is not longer than 250 words. Write the joke about how to find
            love in India.
            '''.strip()

        chat = [{'role': 'user',
                 'content': prompt},
               ]

        response_format = {'type': 'json', 'value':
            {
                'properties':
                    {
                        'Joke': {'type': 'string'},
                    },
                'required': ['Joke']}
            }

        joke_object = self.client.chat_completion(chat, response_format=response_format, max_tokens=300)
        joke = joke_object['choices'][0]['message']['content']
        joke_obj = json.loads(joke)

        return joke_obj['Joke']
    
    def rate_joke(self, joke):
        prompt_eval = f"""
        Please rate the given joke based on the following criterions.
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
        
        Return a json object, with each key being the criterion mentioned above and a corresponding integer score between 0 (worst) and 10 (best)

        Joke: {joke}
        """.strip()

        response_format = {
            "type": "json",
            "value": {
                "properties": {
                    "Humor": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Creativity": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Timeliness": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Personalization": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Tone and Style": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Adaptability": {"type": "integer", "minimum": 1, "maximum": 10},
                    "User Engagement": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Appropriate Content": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Diversity of Jokes": {"type": "integer", "minimum": 1, "maximum": 10},
                    "Delivery": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ['Humor', 'Creativity', 'Timeliness', 'Personalization',
                             'Tone and Style', 'Adaptability', 'User Engagement',
                             'Appropriate Content', 'Diversity of Jokes', 'Delivery'],
            },
        }

        chat_eval = [{'role': 'user', 'content': f"{prompt_eval}"}]
        rating_object = self.client.chat_completion(chat_eval, response_format=response_format, max_tokens=300)

        rating = rating_object['choices'][0]['message']['content']
        rating_dict = json.loads(rating)
        total_score = 0
        for key, value in rating_dict.items():
            total_score += value

        return int(total_score/10)