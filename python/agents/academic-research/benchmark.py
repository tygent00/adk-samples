import asyncio
import time

from google.adk.runners import InMemoryRunner
from google.genai import types

from tygent import accelerate

from academic_research.agent import academic_coordinator


async def _run(agent, question: str):
    runner = InMemoryRunner(agent=agent, app_name="academic-research")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="benchmark"
    )
    content = types.Content(parts=[types.Part(text=question)])
    start = time.perf_counter()
    tokens = 0
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
        if getattr(event, "usage_metadata", None):
            usage = event.usage_metadata
            tokens += (
                (usage.prompt_token_count or 0)
                + (usage.candidates_token_count or 0)
                + (usage.tool_use_prompt_token_count or 0)
                + (usage.cached_content_token_count or 0)
            )
    return time.perf_counter() - start, tokens


async def main():
    question = "Who are you?"
    base_time, base_tokens = await _run(academic_coordinator, question)

    accelerated_run = accelerate(_run)
    acc_time, acc_tokens = await accelerated_run(academic_coordinator, question)

    print(
        f"Baseline: {base_time:.2f}s, {base_tokens} tokens\n"
        f"Tygent accelerated: {acc_time:.2f}s, {acc_tokens} tokens"
    )


if __name__ == "__main__":
    asyncio.run(main())
