import os
import json
from typing import Iterable
from openai import AsyncOpenAI
from .route_specs import  NavigationContext

class RouteGenConfig:
    def __init__(self,
                 api_key: str,
                 model: str,
                 base_url: str | None = None,
                 cache_dir: str = "./cache",
                 temperature: float = 0.7,
                 max_tokens: int | None = None
                ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir

    @classmethod
    def fromEnv(cls, dir: str = "../build/cache"):
        return cls(
            api_key=os.getenv("AI_API_KEY"), # type: ignore
            model=os.getenv("AI_MODEL"), # type: ignore
            base_url=os.getenv("AI_MODEL_URL"),
            cache_dir=dir)


class SailorDataEngineer:
    def __init__(self, config: RouteGenConfig):
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url).beta

    async def _generate_data(
            self,
            route_context: str,
            route_count: int,
            session_count: int
            ) -> NavigationContext | None:
        system_context = """
            Act as a UX data synthesis specialist for complex administrative systems.
            To generate the route data, you must follow the following rules:
            - Each route must have a unique id and path.
            - Route path must mock a real route in the system, it can be base or a nested path.
            - Each route must have a list of tags that must be a mix of the route function and the route path.
            - For each route, you must generate at least 5 tags up to 10 tags.
            To generate the session data, you must follow the following rules:
            - Each session must have a unique id and reference to the route id.
            - Each session must have an intention with a type and a context.
            - Each route can have multiple sessions.
            To generate the intention data, you must follow the following rules:
            - The intention type must be one of the following: "search", "navigation".
            - If the intention type is "search", the context should mock a user search intention based on that route.
            - If the intention type is "navigation", the context should mock a user action of navigation. Like clicking on a link or a button.
        """

        user_context = (
            f"Generate {route_count} routes for a {route_context}."
            f"Should generate at least {session_count} sessions."
        )

        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_context}
        ]

        response = await self._client.chat.completions.parse(
            model=self._config.model,
            messages=messages, # type: ignore
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            response_format=NavigationContext
        )

        return response.choices[0].message.parsed

    async def _get_cached_data(self, cache_file: str) -> NavigationContext | None:
        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return NavigationContext(**data)
        except json.JSONDecodeError:
            return None

    def _save_cache(self, cache_file: str, data: NavigationContext):
        os.makedirs(self._config.cache_dir, exist_ok=True)

        json_data = data.model_dump()
        with open(cache_file, "w") as f:
            json.dump(json_data, f)

    async def generate_data(
            self,
            route_context: str,
            cache_key: str,
            route_count: int = 10,
            session_count: int = 60
            ) -> NavigationContext | None:
        cache_file = os.path.join(self._config.cache_dir, f"{cache_key}.json")

        if cached_data := await self._get_cached_data(cache_file):
            return cached_data

        data = await self._generate_data(route_context, route_count, session_count)
        if data:
            self._save_cache(cache_file, data)

        return data
