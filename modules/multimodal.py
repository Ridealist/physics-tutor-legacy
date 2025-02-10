from langchain_teddynote.models import MultiModal


class MultiModalwithHistory(MultiModal):
    messages_history = [] # Class-level attribute (shared across all instances)

    def __init__(self, model, system_prompt=None, user_prompt=None):
        super().__init__(model, system_prompt, user_prompt)

    def _init_message(
         self, image_url, user_prompt=None, system_prompt=None, display_image=True
    ):   
        # Check if image_url is provided
        if image_url:
            encoded_image = self.encode_image(image_url)
            if display_image:
                self.display_image(encoded_image)
        else:
            encoded_image = None  # Default to None if no image_url is provided

        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
        if encoded_image:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"{encoded_image}"},
                        },
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]
        MultiModalwithHistory.messages_history.extend(messages)

    def create_messages(
        self, user_prompt, image_url=None, system_prompt=None, display_image=True
    ):
        if len(self.messages_history) == 0:
            self._init_message(image_url, user_prompt, system_prompt, display_image)
        else:
            if image_url:
                encoded_image = self.encode_image(image_url)
                if display_image:
                    self.display_image(encoded_image)

                user_prompt = user_prompt if user_prompt is not None else self.user_prompt
                # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{encoded_image}"},
                            },
                        ],
                    },
                ]
            else:
                user_prompt = user_prompt if user_prompt is not None else self.user_prompt
                # 인코딩된 이미지를 사용하여 다른 처리를 수행할 수 있습니다.
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt,
                            },
                        ],
                    },
                ]
            MultiModalwithHistory.messages_history.extend(messages)

        return self.messages_history


    def stream(
        self, user_prompt, image_url=None, system_prompt=None, display_image=True
    ):
        messages = self.create_messages(
            user_prompt, image_url, system_prompt, display_image
        )
        response = self.model.stream(messages)
        # print('-'*50)
        # print(self.messages_history)
        # print('-'*50)
        return response


    def add_messages(self, role, content):
        if role == "ai":
            message = {
                'role': 'assistant',
                'content': content
            }
        else:
            message = {
                'role': 'user',
                'content': content
            }
        MultiModalwithHistory.messages_history.append(message)
        