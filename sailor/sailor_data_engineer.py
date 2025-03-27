import asyncio
import os
import sqlite3
import hashlib
import uuid
from typing import AsyncGenerator, Awaitable, List, Optional
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from sailor.types import RouteSpec, SessionSpec, RouteResponse, SessionResponse

_rate_limit_timeout = 60
_max_sessions_per_fetch = 50
_max_semaphore = 5

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
    def __init__(self, config: RouteGenConfig, verbose: bool = False):
        self._verbose = verbose
        self._config = config
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url).beta
        self._fetch_semaphore = asyncio.Semaphore(_max_semaphore)

    async def generate_routes(self, context: str, count: int) -> List[RouteSpec]:
        if self._verbose: print("[GENERATE_ROUTE] Generating", count, "routes for context:", context)

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

        if self._verbose: print("[GENERATE_ROUTE] Generating routes for context:", context)
        response = await self._client.chat.completions.parse(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            response_format=RouteResponse
        )

        return response.choices[0].message.parsed

    async def generate_sessions(self, context: RouteSpec, count: int) -> AsyncGenerator[List[SessionSpec], None]:
        if self._verbose: print("[GENERATE_SESSION] Generating", count, "sessions for route:", context.id)

        remaining = count
        while remaining > 0:
            try:
                batch_count = min(remaining, _max_sessions_per_fetch)
                async with self._fetch_semaphore:
                    response = await self._generate_sessions(context, batch_count)
                if response: yield response.sessions
                remaining -= batch_count
            except RateLimitError:
                if self._verbose: print("[GENERATE_SESSION] Rate limit reached. Pausing for 1 minute...")
                async with self._fetch_semaphore:
                    await asyncio.sleep(_rate_limit_timeout)

    async def _generate_sessions(self, context: RouteSpec, count: int) -> Optional[SessionResponse]:
        system_context = """
            Act as a UX data synthesis specialist for complex administrative systems.
            To generate the session data, you must follow the following rules:
            - Each session must have a unique id and reference to the route id.
            - Each session must have a context.
            - Each route can have multiple sessions.
            To generate the context data, you must follow the following rules:
            - The context should mock a user search intention based on that route.
            - Introduce variations in 25% of the contexts, including typo errors, synonyms, or related keywords.
        """

        user_context = (
            f"Generate {count} sessions for the given route:"
            f"ID: {context.id}, Path: {context.path}, Tags: {', '.join(context.tags)}"
        )

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_context}
        ]

        if self._verbose: print("[GENERATE_SESSION] Generating sessions for route:", context.id)
        response = await self._client.chat.completions.parse(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            response_format=SessionResponse
        )

        return response.choices[0].message.parsed

class SailorDataWarehouse:
    def __init__(self, config: RouteGenConfig, db_path: str, verbose: bool = False):
        self._verbose = verbose
        self.enginner = SailorDataEngineer(config, verbose=verbose)

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

    async def _get_routes(self, context_id: str, count: int) -> List[RouteSpec]:
        if self._verbose: print("[QUERY_ROUTE] Getting routes for context:", context_id)

        cursor = self.db.cursor()
        query = 'SELECT id, path, tags FROM routes_registry WHERE context_id = ?'
        params = (context_id,)
        if count:
            query += ' LIMIT ?'
            params += (count,)
        cursor.execute(query, params)

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

        if self._verbose: print("[QUERY_ROUTE] Found routes:", len(routes), "for context:", context_id)

        return routes

    async def create_routes(self, context: str, count: int, force_new: bool = False) -> List[RouteSpec]:
        context_hash = hashlib.sha256(context.encode()).hexdigest()
        routes = []
        if not force_new:
            routes = await self._get_routes(context_hash, count)
            if routes and len(routes) >= count: return routes

        count -= len(routes)
        routes = await self.enginner.generate_routes(context, count=count)

        cursor = self.db.cursor()
        for route in routes:
            route_id = uuid.uuid4()
            input = (route_id.hex, context_hash, route.path, ','.join(route.tags))
            cursor.execute("INSERT INTO routes_registry (id, context_id, path, tags) VALUES (?, ?, ?, ?)", input)

        self.db.commit()

        routes = await self._get_routes(context_hash, count)
        return routes

    async def _get_sessions(self, route_id: str, count: Optional[int] = None) -> List[SessionSpec]:
        if self._verbose: print("[QUERY_SESSION] Getting sessions for route:", route_id)

        cursor = self.db.cursor()
        query = 'SELECT id, intention_context FROM sessions_registry WHERE route_id = ?'
        params = (route_id,)
        if count:
            query += ' LIMIT ?'
            params += (count,)
        cursor.execute(query, params)

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

        if self._verbose: print("[QUERY_SESSION] Found sessions:", len(sessions), "for route:", route_id)

        return sessions

    async def create_route_sessions(self, route: RouteSpec, count: int, force_new: bool = False) -> List[SessionSpec]:
        if not force_new:
            cache = await self._get_sessions(route.id)
            if cache and len(cache) >= count: return cache

        count -= len(cache)
        cursor = self.db.cursor()
        async for session_batch in self.enginner.generate_sessions(route, count=count):
            for session in session_batch:
                session_id = uuid.uuid4()
                input = (session_id.hex, route.id, session.context)
                cursor.execute("INSERT INTO sessions_registry (id, route_id, intention_context) VALUES (?, ?, ?)", input)
            self.db.commit()

        sessions = await self._get_sessions(route.id)
        return sessions

    async def create_sessions(self, routes: List[RouteSpec], count: int, force_new: bool = False) -> List[List[SessionSpec]]:
        coroutine: List[Awaitable[List[SessionSpec]]] = []
        for route in routes:
            coroutine.append(self.create_route_sessions(route, count, force_new=force_new))

        responses = await asyncio.gather(*coroutine, return_exceptions=False)
        return responses
