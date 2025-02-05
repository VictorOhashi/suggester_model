import os
import json
from typing import List, Tuple
from openai import AsyncOpenAI
from .route_specs import RouteSpec, SessionSpec, NavigationContext

class SearchConfig:
    def __init__(self,
                 api_key: str,
                 base_url: str | None = None,
                 model: str = "gpt-4o-mini",
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

class SearchDataEngineer:
    def __init__(self, config: SearchConfig):
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url).beta

    async def _generate_data(self, route_context: str, route_count: int, session_count: int) -> NavigationContext:
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
            - Each route can have multiple sessions, with different intention and time_spent.
            - The session time_spent should be a random number between 5 minutes and 1 hour in milliseconds.
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
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            response_format=NavigationContext
        )

        return response.choices[0].message.parsed

    async def _get_cached_data(self) -> NavigationContext | None:
        cache_file = os.path.join(self._config.cache_dir, "navigation_context.json")

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return NavigationContext(**data)
        except json.JSONDecodeError:
            return None

    async def generate_data(self, route_context: str, route_count: int, session_count: int) -> Tuple[List[RouteSpec], List[SessionSpec]]:
        if cached_data := await self._get_cached_data():
            return cached_data.routes, cached_data.sessions

        data = await self._generate_data(route_context, route_count, session_count)
        self._save_cache(data)
        return data.routes, data.sessions

    def _save_cache(self, data: NavigationContext):
        cache_file = os.path.join(self._config.cache_dir, "navigation_context.json")
        os.makedirs(self._config.cache_dir, exist_ok=True)

        json_data = data.model_dump()
        with open(cache_file, "w") as f:
            json.dump(json_data, f)
