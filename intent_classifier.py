# -*- coding: utf-8 -*-
from openai import OpenAI
import json
import logging
import dotenv

# Load OpenAI API-key from .env file
dotenv.load_dotenv()

# create streamhandler logger
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
LOG = logging.getLogger("intent_classifier")

# jsonl file paths
prompt_jsonl = 'prompt.jsonl'
prompt_history_jsonl = 'history.jsonl'


class IntentClassifier:
    '''
    Class to predict the intent from a text message using OpenAI's API.
    '''

    def __init__(self) -> None:
        '''
        Initialize the OpenAI client and class variables.

        Raises:
            Exception: If the OpenAI client cannot be initialized.
        '''

        self.client = None
        self.model_id = None
        self.intents_list = None
        self.prompt = None

        try:
            self.client = OpenAI()

        except Exception as e:
            LOG.error(f"Error initializing OpenAI client: {e}")
            return

    def is_ready(self) -> bool:
        '''
        Check if the model is available.

        Returns:
            bool: True if the model is available, False otherwise.

        Raises:
            Exception: If the list of available models cannot be accessed.
        '''

        if not self.client:
            LOG.error("OpenAI client not initialized.")
            return False

        # Check if the model is available
        try:
            available_models = [model.id for model in self.client.models.list()]

            if self.model_id in available_models:
                return True
            else:
                LOG.error(f"Model {self.model_id} not available. Make sure the name of the model is correct.")

        except Exception as e:
            LOG.error(f"Error fetching the list of available models: {e}.")

        return False

    def load(self, model, intent_list=None) -> None:
        '''
        Store the name of the OpenAI model and load the intent list.

        Args:
            model (str): Name of the OpenAI model.
            intent_list (list): List of intents to classify. Loaded from file if not provided.

        Raises:
            Exception: If the list cannot be loaded.
        '''

        # load model
        self.model_id = model
        LOG.debug(f"Loaded model {self.model_id}")

        # load intents list
        if intent_list is None:
            # load the intent list from the text file
            try:
                with open('intents.txt', 'r') as file:
                    intent_list = [line.strip() for line in file.readlines() if line.strip()]
                    intent_list.append('none')
                    LOG.debug(f"Loaded intents list: {self.intents_list}")

            except Exception as e:
                LOG.error(f"Error loading intents list: {e}")
                return

        self.intents_list = intent_list
        LOG.debug(f"Loaded intents list: {self.intents_list}")

    def load_prompt(self) -> None:
        '''
        Load the custom prompt from the jsonl file and replace the INTENT_LIST placeholder.

        Raises:
            Exception: If the prompt cannot be loaded.
        '''

        self.prompt = []

        try:
            with open(prompt_jsonl, 'r') as file:
                for line in file.readlines():
                    if 'INTENT_LIST' in line:
                        line = line.replace('INTENT_LIST', ', '.join(self.intents_list))
                    self.prompt.append(json.loads(line))

            LOG.debug(self.prompt)
            LOG.info('Prompt loaded successfully.')

        except Exception as e:
            LOG.error(f"Error loading prompt: {e}")
            return

    def predict_intent(self, text) -> None | dict:
        '''
        Predict the intent of the text message.

        Args:
            text (str): Text message to classify.

        Returns:
            dict: predicted intent as JSON object.

        Raises:
            ValueError: If no intents are returned from the completion call.
            Exception: If the history file cannot be saved.
        '''

        # load initial prompt if needed
        if self.prompt is None:
            LOG.debug('Prompt not loaded. Loading prompt.')
            self.load_prompt()

        try:
            # we create a copy in case the request doest not work
            prompt = self.prompt.copy()

            # add the user input to the prompt
            user_dict = {
                "role": "user",
                "content": f'{{"text": "{text}"}}'
            }
            prompt.append(user_dict)

            # send prompt to OpenAI and predict the 3 most likely intents
            response = self.client.chat.completions.create(
                model=self.model_id,
                response_format={"type": "json_object"},
                messages=prompt,
                max_tokens=1000,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            # parse the response
            response_json = json.loads(response.choices[0].message.content)
            LOG.debug(f"Model response: {response_json}")

            # exit function if no intents are returned (text cannot be classified)
            if response_json['intent'] == 'none' or response_json['intent'] not in self.intents_list:
                raise ValueError("No intents returned from completion call.")

        except Exception as e:
            LOG.error(f"Error predicting intent: {e}")
            return None

        # add user and assistant messages to the prompt to keep track of the conversation
        self.prompt.append(user_dict)

        answer_dict = {
            "role": "assistant",
            "content": f'{response_json}'
        }
        self.prompt.append(answer_dict)

        # save prompt history to keep a local copy of the conversation (could be resumed later)
        try:
            with open(prompt_history_jsonl, 'w') as file:
                for p in self.prompt:
                    file.write(json.dumps(p) + '\n')
            LOG.debug('Prompt history saved successfully.')

        except Exception as e:
            LOG.warning(f"Error saving prompt history: {e}.")

        return response_json


if __name__ == '__main__':
    pass
