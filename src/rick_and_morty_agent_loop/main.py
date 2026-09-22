"""Rick and Morty agent loop.

Send the conversation to the model, append its reply, run whatever tools it
asked for, repeat until it stops asking for tools (or we run out of turns).

The model is whatever `client.resolve` hands back -- OpenRouter, OpenAI or
Anthropic -- and the loop below is written against pydantic-ai's message
types, so none of it changes when the provider does.

The tools wrap https://rickandmortyapi.com through the `ramapi` client
(https://github.com/curiousrohan/ramapi).
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from functools import partial
from typing import Any

import ramapi
from dotenv import load_dotenv
from pydantic_ai import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    ToolDefinition,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.direct import model_request_sync
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from rick_and_morty_agent_loop import client

# API keys come from .env at the project root; a real environment variable, if
# one is already set, wins over the file.
load_dotenv()

# Both optional, and both come from .env: blank PROVIDER means whichever API
# key is set, blank MODEL means that provider's default.
PROVIDER = os.getenv("PROVIDER", "")
MODEL = os.getenv("MODEL", "")

MAX_TURNS = 20

# Turns held back at the end. The tools switch off this many turns before the
# wall and the answer comes due, so a run that would have spent its last turn on
# one more search writes the answer instead -- with turns still in hand to have
# another go if that first attempt comes back blank.
RESERVE_TURNS = 6

# The turn the tools go away and the answer is due. Never below 1: a reserve as
# big as the budget would otherwise leave no turn to call a tool in.
ANSWER_TURN = max(1, MAX_TURNS - RESERVE_TURNS)

# A turn with neither an answer nor a tool call is a malfunction, not a stop.
# Usually it is a hiccup, so nudge the model rather than ending the run -- but
# a model that keeps doing it is stuck, and each nudge costs a turn.
MAX_BLANK_TURNS = 2

# What to type at the prompt to leave, or to drop the conversation history.
EXIT = {"exit", "quit", "q"}
FORGET = {"new", "reset", "clear"}

NUDGE = """\
That turn came back empty. Either call a tool to get what you still need, or
write the answer with what you already have."""

DEADLINE = f"""\
Turn {ANSWER_TURN} of {MAX_TURNS}: the tools are switched off from here, so this
turn is the answer. Write it from the observations you already have, and say
plainly what you could not confirm rather than filling the gap."""

# The API links resources by URL, and a popular location can list hundreds of
# residents. We hand the model ids instead, capped -- the matching *_count field
# keeps the true total visible.
MAX_IDS = 50

DEFAULT_QUESTION = "Which main character am I like if I like salads"

SYSTEM_PROMPT = f"""\
You are a Rick and Morty research agent. Questions about the show's characters,
episodes and locations get answered from the tools you are given, not from
memory.

Work the ReAct way — reason, act, observe, repeat:

1. Thought. Before each tool call, say in a sentence or two what you still need
   and which tool gets it. This is scratch work, not the answer; keep it short.
2. Action. Call one tool with the narrowest arguments that satisfy the thought.
   Prefer a filtered query over fetching everything and sifting it yourself.
3. Observation. The result comes back to you as a tool message. Read it before
   deciding anything — never assume a call succeeded or guess what it returned.
4. Repeat until the observations answer the question, then stop calling tools
   and write the answer.

Ground every claim in an observation. If the tools cannot support a claim, say
so instead of filling the gap from memory. An empty result or an error is
information: loosen the filter, try a different spelling, or reach for another
tool, rather than repeating the same call unchanged. Never invent a tool, an
argument, or a result.

You get {MAX_TURNS} turns, and the tools only last for the first
{ANSWER_TURN - 1} of them: on turn {ANSWER_TURN} they switch off and the answer
is due. Spend the tool turns on distinct questions rather than retries of the
same one, and when the tools go, answer with what you have and name what is
still missing.

The final answer is plain prose: no Thought/Action labels, and state what you
found rather than narrating the search.
"""


# --- tools ------------------------------------------------------------------
# Each API resource gets two tools: a filtered search and a fetch-by-id. The
# payloads are trimmed to what an answer needs -- no image URLs, no self-links.

Json = dict[str, Any]


def _ids(urls: list[str]) -> list[int]:
    """["https://.../character/2", ...] -> [2, ...], capped at MAX_IDS."""
    return [int(url.rsplit("/", 1)[-1]) for url in urls[:MAX_IDS]]


def _character(raw: Json) -> Json:
    return {
        "id": raw["id"],
        "name": raw["name"],
        "status": raw["status"],
        "species": raw["species"],
        "type": raw["type"],
        "gender": raw["gender"],
        "origin": raw["origin"]["name"],
        "location": raw["location"]["name"],
        "episode_count": len(raw["episode"]),
        "episode_ids": _ids(raw["episode"]),
    }


def _location(raw: Json) -> Json:
    return {
        "id": raw["id"],
        "name": raw["name"],
        "type": raw["type"],
        "dimension": raw["dimension"],
        "resident_count": len(raw["residents"]),
        "resident_ids": _ids(raw["residents"]),
    }


def _episode(raw: Json) -> Json:
    return {
        "id": raw["id"],
        "name": raw["name"],
        "air_date": raw["air_date"],
        "episode": raw["episode"],
        "character_count": len(raw["characters"]),
        "character_ids": _ids(raw["characters"]),
    }


def _search(resource: Any, shrink: Callable[[Json], Json], args: Json) -> Json:
    """One page of filtered results, or the API's own {"error": ...} payload."""
    # ramapi pastes params straight into the query string, so they must be str.
    query = {k: str(v) for k, v in args.items() if v not in (None, "")}
    raw = resource.filter(**query)
    if "error" in raw:
        return raw
    return {
        "count": raw["info"]["count"],
        "pages": raw["info"]["pages"],
        "results": [shrink(result) for result in raw["results"]],
    }


def _fetch(resource: Any, shrink: Callable[[Json], Json], args: Json) -> Json:
    raw = resource.get(args["id"])
    return raw if "error" in raw else shrink(raw)


TOOL_SPECS: list[ToolDefinition] = []
TOOL_IMPLS: dict[str, Callable[[Json], Json]] = {}

_ID = {"type": "integer", "description": "Resource id, as it appears in search results and in the id lists on other resources."}
_PAGE = {"type": "integer", "description": "1-based page number. Omit for the first page."}


def _tool(
    name: str,
    run: Callable[[Json], Json],
    description: str,
    properties: Json,
    required: tuple[str, ...] = (),
) -> None:
    """Register a tool: the schema the model sees, and the code behind it.

    Pydantic AI renders one `ToolDefinition` into whichever shape the provider
    wants -- an OpenAI function, an Anthropic tool -- so the schema is written
    once here.
    """
    TOOL_SPECS.append(
        ToolDefinition(
            name=name,
            description=description,
            parameters_json_schema={
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        )
    )
    TOOL_IMPLS[name] = run


_tool(
    "search_characters",
    partial(_search, ramapi.Character, _character),
    "Search the ~800 characters in the show. Text filters are case-insensitive "
    "substring matches and combine with AND; with no filters at all you get "
    "everything, 20 per page. Returns `count`, `pages`, and for each character "
    "their id, status, species, type (subspecies), gender, origin, last known "
    "location and the ids of the episodes they appear in.",
    {
        "name": {"type": "string", "description": "Full or partial character name, e.g. 'rick' or 'morty smith'."},
        "status": {"type": "string", "enum": ["alive", "dead", "unknown"]},
        "species": {"type": "string", "description": "e.g. 'Human', 'Alien', 'Robot', 'Cronenberg'."},
        "type": {"type": "string", "description": "Subspecies or variant, e.g. 'Parasite', 'Clone'."},
        "gender": {"type": "string", "enum": ["female", "male", "genderless", "unknown"]},
        "page": _PAGE,
    },
)

_tool(
    "get_character",
    partial(_fetch, ramapi.Character, _character),
    "Fetch one character by id — the way to turn an id from another result "
    "(a location's resident_ids, an episode's character_ids) into a name and "
    "details. Same fields as search_characters returns per character.",
    {"id": _ID},
    required=("id",),
)

_tool(
    "search_locations",
    partial(_search, ramapi.Location, _location),
    "Search the planets, dimensions, space stations and microverses characters "
    "come from or live on. Text filters are case-insensitive substring matches "
    "and combine with AND; 20 results per page. Returns `count`, `pages`, and "
    "for each location its id, type, dimension and the ids of the characters "
    "last seen there.",
    {
        "name": {"type": "string", "description": "Full or partial location name, e.g. 'earth' or 'citadel'."},
        "type": {"type": "string", "description": "Kind of place, e.g. 'Planet', 'Space station', 'Microverse'."},
        "dimension": {"type": "string", "description": "e.g. 'C-137', 'Replacement Dimension'."},
        "page": _PAGE,
    },
)

_tool(
    "get_location",
    partial(_fetch, ramapi.Location, _location),
    "Fetch one location by id, e.g. to resolve a character's origin or current "
    "location into its dimension and resident list.",
    {"id": _ID},
    required=("id",),
)

_tool(
    "search_episodes",
    partial(_search, ramapi.Episode, _episode),
    "Search episodes by title or by production code. With no filters you get "
    "all episodes in air order, 20 per page. Returns `count`, `pages`, and for "
    "each episode its id, title, air date, code and the ids of the characters "
    "who appear in it. Episode ids run in air order, so the lowest id in a "
    "character's episode_ids is their first appearance.",
    {
        "name": {"type": "string", "description": "Full or partial episode title, e.g. 'pilot' or 'pickle'."},
        "episode": {"type": "string", "description": "Season/episode code, e.g. 'S01E01'. A season prefix like 'S03' matches the whole season."},
        "page": _PAGE,
    },
)

_tool(
    "get_episode",
    partial(_fetch, ramapi.Episode, _episode),
    "Fetch one episode by id — the way to turn an id from a character's "
    "episode_ids into a title, air date and cast list.",
    {"id": _ID},
    required=("id",),
)


def run_tool(call: ToolCallPart) -> ToolReturnPart:
    """Run one tool call and shape the result as a tool-return part."""
    run = TOOL_IMPLS.get(call.tool_name)
    if run is None:
        result: Json = {"error": f"no such tool: {call.tool_name}"}
    else:
        try:
            # args arrive as a JSON string or an already-parsed dict depending
            # on the provider; args_as_dict flattens that difference.
            result = run(call.args_as_dict())
        except Exception as exc:  # hand the failure back; the model can adapt
            result = {"error": f"{type(exc).__name__}: {exc}"}
    return ToolReturnPart(
        tool_name=call.tool_name,
        content=result,
        tool_call_id=call.tool_call_id,
    )


# --- tracing -----------------------------------------------------------------


def _cost(response: ModelResponse) -> Decimal | None:
    """What one turn cost: the provider's own figure, else the price list."""
    if response.usage.cost is not None:
        return response.usage.cost
    try:
        return response.cost().total_price
    except (LookupError, AssertionError):
        # genai-prices does not know every model -- OpenRouter's free ones, for
        # one -- and an unpriced turn is not worth failing a run over.
        return None


@dataclass
class Trace:
    """Running totals for one `run`, reported however the run ends."""

    usage: RunUsage = field(default_factory=RunUsage)
    spent: Decimal = Decimal(0)
    # Goes False the moment a turn comes back without a price, so the total can
    # admit it is a floor rather than the bill.
    priced: bool = True
    turns: int = 0
    started: float = field(default_factory=time.monotonic)

    def record(self, response: ModelResponse, seconds: float) -> None:
        """Fold one turn into the totals and print its line of accounting."""
        cost = _cost(response)
        self.usage.incr(response.usage)
        self.spent += cost or Decimal(0)
        self.priced = self.priced and cost is not None
        self.turns += 1

        counts = [f"in={response.usage.input_tokens:,}"]
        # Only mention the cache when something hit or filled it, so the line
        # stays short on providers that do not cache at all.
        if response.usage.cache_read_tokens:
            counts.append(f"cached={response.usage.cache_read_tokens:,}")
        if response.usage.cache_write_tokens:
            counts.append(f"wrote={response.usage.cache_write_tokens:,}")
        counts.append(f"out={response.usage.output_tokens:,}")

        line = f"  usage {' '.join(counts)} | {seconds:.1f}s"
        print(line if cost is None else f"{line} | ${cost:.4f}")

    def merge(self, other: "Trace") -> None:
        """Fold one question's totals into the session's."""
        self.usage.incr(other.usage)
        self.spent += other.spent
        self.priced = self.priced and other.priced
        self.turns += other.turns

    def summary(self) -> str:
        bits = [
            f"{self.turns} turn{'' if self.turns == 1 else 's'}",
            f"{self.usage.total_tokens:,} tokens "
            f"(in {self.usage.input_tokens:,} / out {self.usage.output_tokens:,})",
        ]
        if self.usage.cache_read_tokens:
            bits.append(f"{self.usage.cache_hit_ratio:.0%} cached")
        if self.spent:
            bits.append(("$" if self.priced else "~$") + f"{self.spent:.4f}")
        bits.append(f"{time.monotonic() - self.started:.1f}s")
        return "-- " + " | ".join(bits)


# --- loop --------------------------------------------------------------------


def run(question: str, provider: str = PROVIDER, model_name: str = MODEL) -> str:
    # Resolve the blank cases first, so the line below names what actually ran.
    provider = provider or client.detect()
    model = client.resolve(provider, model_name)
    print(f"[{provider}] {model.model_name}")

    messages: list[ModelMessage] = []

    trace = Trace()
    try:
        return ask(model, messages, question, trace)
    finally:
        # However the run ended -- an answer, a giveup, or an exception on the
        # way out -- the totals are worth seeing.
        print(trace.summary())


def ask(model: Model, messages: list[ModelMessage], question: str, trace: Trace) -> str:
    """Put a question to an ongoing conversation and answer it.

    `messages` is carried between questions, so follow-ups ("what episodes is
    she in?") land with the earlier answers still in view.
    """
    # Only the newest question carries the cache point, so everything before it
    # -- system prompt, tool schemas, every earlier turn -- is read from cache.
    # It has to move rather than accumulate: Anthropic caps a request at four
    # cache_control blocks and pydantic-ai does not prune stale ones, so a long
    # session would start failing at the fifth question.
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                part.content = [c for c in part.content if not isinstance(c, CachePoint)]

    # The system prompt rides along with the first question and nothing after.
    parts: list[ModelRequestPart] = [] if messages else [SystemPromptPart(content=SYSTEM_PROMPT)]
    parts.append(UserPromptPart(content=[question, CachePoint()]))
    messages.append(ModelRequest(parts=parts))

    return _loop(model, messages, trace)


def _say(messages: list[ModelMessage], text: str) -> None:
    """Put a word in from our side, without stacking two requests in a row.

    Providers expect the conversation to alternate, and a request carrying tool
    results is already our turn to speak -- so the note joins it rather than
    following it.
    """
    part = UserPromptPart(content=text)
    last = messages[-1] if messages else None
    if isinstance(last, ModelRequest):
        last.parts.append(part)
    else:
        messages.append(ModelRequest(parts=[part]))


def _loop(model: Model, messages: list[ModelMessage], trace: Trace) -> str:
    params = ModelRequestParameters(function_tools=TOOL_SPECS)
    blanks = 0

    for turn in range(1, MAX_TURNS + 1):
        # From ANSWER_TURN on, prose is the only thing left to produce. Saying
        # so, once, keeps the model from spending the turn reaching for a tool
        # that is no longer on offer.
        due = turn >= ANSWER_TURN
        if turn == ANSWER_TURN:
            _say(messages, DEADLINE)

        note = " -- answer due, tools off" if due else ""
        print(f"[turn {turn}/{MAX_TURNS}] messages={len(messages)}{note}")

        clock = time.monotonic()
        response = model_request_sync(
            model,
            messages,
            model_settings=ModelSettings(
                max_tokens=10000,
                # Once the answer is due another tool call buys nothing, so
                # spend the turn -- and the reserve behind it -- on writing.
                tool_choice="none" if due else "auto",
            ),
            model_request_parameters=params,
        )
        trace.record(response, time.monotonic() - clock)

        # A ModelResponse is itself a message, so appending it whole keeps the
        # tool calls -- and any reasoning the provider wants echoed back --
        # intact for the next request.
        messages.append(response)

        # The finishes that mean "this turn is not an answer", each named so a
        # failure never comes back looking like an empty answer.
        if response.finish_reason == "content_filter":
            return "refused: content filter"
        if response.finish_reason == "length":
            return "stopped: ran out of output tokens mid-message"
        if response.finish_reason == "error":
            return "stopped: the provider reported an error"

        # Some providers report finish_reason "stop" alongside tool calls, so
        # trust the calls themselves rather than the reason.
        if calls := response.tool_calls:
            blanks = 0
            for call in calls:
                print(f"  -> {call.tool_name}({call.args_as_json_str()})")
            # All the results for one response go back in a single request,
            # which is what providers expect when they call tools in parallel.
            messages.append(ModelRequest(parts=[run_tool(call) for call in calls]))
            continue

        # Nothing left to run, so this turn was meant to be the answer.
        if answer := (response.text or "").strip():
            return answer

        # It said nothing at all. Ask again -- but only so many times, and
        # never silently: a blank run should not read like an empty answer.
        blanks += 1
        if blanks > MAX_BLANK_TURNS:
            return (
                f"stopped: {blanks} blank turns in a row, "
                f"last finished on {response.finish_reason!r}"
            )
        print(f"  (blank turn {blanks}/{MAX_BLANK_TURNS}, nudging)")
        _say(messages, NUDGE)

    return "stopped: hit max turns without an answer"


def main() -> None:
    provider = PROVIDER or client.detect()
    model = client.resolve(provider, MODEL)

    print(f"[{provider}] {model.model_name}")
    print(f"Ask about Rick and Morty -- e.g. {DEFAULT_QUESTION!r}")
    print(f"{'/'.join(sorted(EXIT))} to leave, {'/'.join(sorted(FORGET))} to start over.\n")

    messages: list[ModelMessage] = []
    session = Trace()

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D or Ctrl-C at the prompt is a way out, not a crash.
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT:
            break
        if question.lower() in FORGET:
            messages = []
            print("(conversation forgotten)\n")
            continue

        # Each question gets its own totals; the session keeps the running sum.
        mark, trace = len(messages), Trace()
        try:
            print()
            print(ask(model, messages, question, trace))
        # KeyboardInterrupt is not an Exception, so it needs naming separately.
        except (KeyboardInterrupt, Exception) as exc:
            # Ctrl-C mid-answer, or a provider error like a rate limit, ends the
            # question and not the session. Rewind the conversation to where the
            # question started, so a run that did not finish leaves nothing
            # behind -- no unanswered question, no half-finished tool exchange
            # for the next question to trip over.
            del messages[mark:]
            print(
                "\n(interrupted)"
                if isinstance(exc, KeyboardInterrupt)
                else f"\nfailed: {type(exc).__name__}: {exc}"
            )
        finally:
            print(trace.summary(), "\n")
            session.merge(trace)

    if session.turns:
        print("session:", session.summary().removeprefix("-- "))


if __name__ == "__main__":
    main()
