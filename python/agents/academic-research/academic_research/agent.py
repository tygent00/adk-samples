# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Academic_Research: Research advice, related literature finding, research area proposals, web knowledge access."""

import importlib
import time

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
from tygent import accelerate

from . import prompt
from .sub_agents.academic_newresearch import academic_newresearch_agent
from .sub_agents.academic_websearch import academic_websearch_agent

MODEL = "gemini-2.5-pro"


def create_agent(*, accelerated: bool = False) -> LlmAgent:
    """Constructs the academic coordinator agent.

    Args:
        accelerated: Whether to wrap the agent with Tygent acceleration.

    Returns:
        The initialized (optionally accelerated) agent.
    """

    agent = LlmAgent(
        name="academic_coordinator",
        model=MODEL,
        description=(
            "analyzing seminal papers provided by the users, "
            "providing research advice, locating current papers "
            "relevant to the seminal paper, generating suggestions "
            "for new research directions, and accessing web resources "
            "to acquire knowledge"
        ),
        instruction=prompt.ACADEMIC_COORDINATOR_PROMPT,
        output_key="seminal_paper",
        tools=[
            AgentTool(agent=academic_websearch_agent),
            AgentTool(agent=academic_newresearch_agent),
        ],
    )
    if accelerated:
        return accelerate(agent)
    return agent


academic_coordinator = create_agent()

# Default exported agent
root_agent = academic_coordinator


async def _run(agent: LlmAgent, question: str) -> tuple[float, int, str]:
    """Executes the agent and returns timing, token usage, and output."""
    runner = InMemoryRunner(agent=agent, app_name="academic-research")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="comparison"
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


async def run_with_and_without_acceleration(question: str) -> dict:
    """Runs the agent normally and with Tygent acceleration for comparison."""
    importlib.reload(prompt)
    base_agent = create_agent()
    base_time, base_tokens, base_output = await _run(base_agent, question)

    importlib.reload(prompt)
    accelerated_agent = create_agent(accelerated=True)
    acc_time, acc_tokens, acc_output = await _run(accelerated_agent, question)

    return {
        "baseline": {
            "time": base_time,
            "tokens": base_tokens,
            "output": base_output,
        },
        "accelerated": {
            "time": acc_time,
            "tokens": acc_tokens,
            "output": acc_output,
        },
    }


def main() -> None:
    """Command-line entry point for comparing runs."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Compare default and accelerated academic research agent runs",
    )
    parser.add_argument("prompt", nargs="?", help="Question to ask the agent")
    args = parser.parse_args()

    question = args.prompt or input("Enter your question: ")
    results = asyncio.run(run_with_and_without_acceleration(question))
    print("Baseline:", results["baseline"])
    print("Accelerated:", results["accelerated"])


if __name__ == "__main__":
    main()
