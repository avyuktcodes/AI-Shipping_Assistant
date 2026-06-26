import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agent import app


async def main():
    runner = InMemoryRunner(app=app)
    session = await runner.session_service.create_session(
        app_name="app", user_id="test_user"
    )

    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part.from_text(text="What are your shipping rates?")],
        ),
    ):
        print(event)


asyncio.run(main())
