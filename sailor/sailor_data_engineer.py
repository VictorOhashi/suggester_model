import asyncio
import os
import sqlite3
import hashlib
import uuid
from typing import Awaitable, List, Optional
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from sailor.types import RouteSpec, SessionSpec, RouteResponse, SessionResponse

class RouteGenConfig:
    def __init__(self,
                 api_key: str,
                 model: str,
                 base_url: str | None = None,
                 temperature: float = 0.7,
                ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("AI_API_KEY"), # type: ignore
            model=os.getenv("AI_MODEL"), # type: ignore
            base_url=os.getenv("AI_MODEL_URL"))

class SailorDataEngineer:
    def __init__(self, config: RouteGenConfig):
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url).beta

    async def generate_routes(self, context: str, count: int) -> List[RouteSpec]:
        routes = await self._generate_routes(context, count)
        if routes is None:
            raise ValueError("No routes generated")

        return routes.routes

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

    async def generate_sessions(self, context: List[RouteSpec], count: int) -> List[SessionSpec]:
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

        return sessions.sessions

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

class SailorDataWarehouse:
    def __init__(self, config: RouteGenConfig, db_path: str):
        self.enginner = SailorDataEngineer(config)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self._init()

    def _init(self):
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routes_registry (
                id TEXT PRIMARY KEY,
                context_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                tags TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions_registry (
                id TEXT PRIMARY KEY,
                route_id INTEGER NOT NULL,
                intention_context TEXT NOT NULL
            )
        ''')
        self.db.commit()

    async def _get_routes(self, context_id: str) -> List[RouteSpec]:
        cursor = self.db.cursor()
        cursor.execute('SELECT id, path, tags FROM routes_registry WHERE context_id = ?', (context_id,))

        routes_data = cursor.fetchall()
        if not routes_data: return []

        routes = []
        for route_data in routes_data:
            route = RouteSpec(
                id=route_data[0],
                path=route_data[1],
                tags=route_data[2].split(',')
            )
            routes.append(route)

        return routes

    async def create_routes(self, context: str, count: int = 10, force_new: bool = False) -> List[RouteSpec]:
        context_hash = hashlib.sha256(context.encode()).hexdigest()

        if not force_new:
            routes = await self._get_routes(context_hash)
            if routes: return routes

        routes = await self.enginner.generate_routes(context, count=count)

        cursor = self.db.cursor()
        for route in routes:
            route_id = uuid.uuid4()
            input = (route_id.hex, context_hash, route.path, ','.join(route.tags))
            cursor.execute("INSERT INTO routes_registry (id, context_id, path, tags) VALUES (?, ?, ?, ?)", input)

        self.db.commit()

        routes = await self._get_routes(context_hash)
        return routes

    async def _get_sessions(self, route_id: str) -> List[SessionSpec]:
        cursor = self.db.cursor()
        cursor.execute('SELECT id, intention_context FROM sessions_registry WHERE route_id = ?', (route_id,))

        sessions_data = cursor.fetchall()
        if not sessions_data: return []

        sessions = []
        for session_data in sessions_data:
            session = SessionSpec(
                id=session_data[0],
                route_id=route_id,
                context=session_data[1]
            )
            sessions.append(session)

        return sessions

    async def create_sessions(self, route: RouteSpec, count: int = 10, force_new: bool = False) -> List[SessionSpec]:
        if not force_new:
            sessions = await self._get_sessions(route.id)
            if sessions: return sessions

        sessions = await self.enginner.generate_sessions([route], count=count)

        return sessions
