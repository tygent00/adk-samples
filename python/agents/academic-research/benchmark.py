import asyncio
import time

from google.adk.runners import InMemoryRunner
from google.genai import types

from academic_research.agent import create_agent


async def _run(agent, question: str):
    runner = InMemoryRunner(agent=agent, app_name="academic-research")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="benchmark"
    )
    content = types.Content(parts=[types.Part(text=question)])
    start = time.perf_counter()
    tokens = 0
    response_text = ""
    async for event in runner.run_async(
        user_id=session.user_id, session_id=session.id, new_message=content
    ):
        if event.content.parts and event.content.parts[0].text:
            response_text = event.content.parts[0].text
        if getattr(event, "usage_metadata", None):
            usage = event.usage_metadata
            tokens += (
                (usage.prompt_token_count or 0)
                + (usage.candidates_token_count or 0)
                + (usage.tool_use_prompt_token_count or 0)
                + (usage.cached_content_token_count or 0)
            )
    return time.perf_counter() - start, tokens, response_text


async def main():
    question = "Who are you?"

    baseline_agent = create_agent()
    base_time, base_tokens, base_output = await _run(baseline_agent, question)

    accelerated_agent = create_agent(accelerated=True)
    acc_time, acc_tokens, acc_output = await _run(accelerated_agent, question)

    print(
        f"Baseline: {base_time:.2f}s, {base_tokens} tokens\n"
        f"Output: {base_output}\n\n"
        f"Tygent accelerated: {acc_time:.2f}s, {acc_tokens} tokens\n"
        f"Output: {acc_output}"
    )


if __name__ == "__main__":
    asyncio.run(main())
