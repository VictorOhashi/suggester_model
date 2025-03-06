import asyncio
import os
import json
from typing import Awaitable, List, Optional
from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletionMessageParam
from sailor.route_context import NavigationContext
from sailor.route_specs import RouteSpec, SessionSpec

class RouteGenConfig:
    def __init__(self,
                 api_key: str,
                 model: str,
                 base_url: str | None = None,
                 cache_dir: str = "./cache",
                 temperature: float = 0.7,
                ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.cache_dir = cache_dir

    @classmethod
    def from_env(cls, dir: str = "../build/cache"):
        return cls(
            api_key=os.getenv("AI_API_KEY"), # type: ignore
            model=os.getenv("AI_MODEL"), # type: ignore
            base_url=os.getenv("AI_MODEL_URL"),
            cache_dir=dir)

class RouteResponse(BaseModel):
    routes: List[RouteSpec]

class SessionResponse(BaseModel):
    sessions: List[SessionSpec]

class SailorDataEngineer:
    def __init__(self, config: RouteGenConfig, cache_key: str, route_description: str):
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url).beta
        self._cache_key = cache_key
        self._route_description = route_description

    async def _generate_routes(self, context: str, count: int) -> RouteResponse | None:
        system_context = """
            Act as a UX data synthesis specialist for complex administrative systems.
            To generate the route data, you must follow the following rules:
            - Each route must have a unique id and path.
            - Route path must mock a real route in the system, it can be base or a nested path.
            - Each route must have a list of tags that must be a mix of the route function and the route path.
            - For each route, you must generate at least 5 tags up to 10 tags.
        """

        user_context = f"Generate {count} routes for a {context}."

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_context}
        ]

        response = await self._client.chat.completions.parse(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            response_format=RouteResponse
        )

        return response.choices[0].message.parsed

    async def _get_routes(self, context: str, count: int) -> List[RouteSpec]:
        cache_file = os.path.join(self._config.cache_dir, f"{self._cache_key}_routes.json")
        routes: Optional[RouteResponse] = None

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    routes = RouteResponse(**data)
                    return routes.routes
            except json.JSONDecodeError:
                pass

        routes = await self._generate_routes(context, count)
        if routes is None:
            raise ValueError("No routes generated")

        with open(cache_file, "w") as f:
            json.dump(routes.model_dump(), f)

        return routes.routes

    async def _generate_sessions(self, context: RouteSpec, count: int) -> Optional[SessionResponse]:
        system_context = """
            Act as a UX data synthesis specialist for complex administrative systems.
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
            f"Generate {count} sessions for the given route:"
            f"ID: {context.id}, Path: {context.path}, Tags: {', '.join(context.tags)}"
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_context}
        ]

        response = await self._client.chat.completions.parse(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            response_format=SessionResponse
        )

        return response.choices[0].message.parsed

    async def _get_sessions(self, context: List[RouteSpec], count: int) -> List[SessionSpec]:
        cache_file = os.path.join(self._config.cache_dir, f"{self._cache_key}_sessions.json")
        sessions: Optional[SessionResponse] = None

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    sessions = SessionResponse(**data)
                    return sessions.sessions
            except json.JSONDecodeError:
                pass

        sessions_semaphore = asyncio.Semaphore(5)
        async def wrapped_generate(route: RouteSpec, batch_count: int) -> Optional[SessionResponse]:
            async with sessions_semaphore:
                return await self._generate_sessions(route, batch_count)

        sessions_coroutine: List[Awaitable[Optional[SessionResponse]]] = []
        for route in context:
            remaining = count
            while remaining > 0:
                batch_count = min(remaining, 50)
                sessions_coroutine.append(wrapped_generate(route, batch_count))
                remaining -= batch_count

        sessions = SessionResponse(sessions=[])
        sessions_responses = await asyncio.gather(*sessions_coroutine, return_exceptions=False)
        for response in sessions_responses:
            if response is None:
                print(f"No sessions generated for route: {route.id}")
                continue
            sessions.sessions.extend(response.sessions)

        with open(cache_file, "w") as f:
            json.dump(sessions.model_dump(), f)

        return sessions.sessions

    async def generate_data(self, route_count: int, session_count: int) -> Optional[NavigationContext]:
        os.makedirs(self._config.cache_dir, exist_ok=True)

        routes = await self._get_routes(self._route_description, route_count)
        sessions = await self._get_sessions(routes, session_count)

        data = NavigationContext(routes=routes, sessions=sessions)

        return data
