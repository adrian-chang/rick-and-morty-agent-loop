# rick-and-morty-agent-loop

A hand-written ReAct agent loop that answers Rick and Morty questions from
[rickandmortyapi.com](https://rickandmortyapi.com).

The loop is provider-agnostic: [Pydantic AI](https://ai.pydantic.dev) hands each
provider a client that speaks its own protocol behind one interface, so the same
loop, the same tool schemas and the same message list run against OpenRouter,
OpenAI or Claude. Which one you get is decided by the API key you set.

## Setup

```sh
uv sync
cp .env.example .env
```

Then set the key for the provider you want. That's the whole config — the key
you set is the provider you get:

```sh
OPENROUTER_API_KEY=...   # -> OpenRouter
OPENAI_API_KEY=...       # -> OpenAI
ANTHROPIC_API_KEY=...    # -> Claude
```

## Run

```sh
uv run rick-and-morty-agent-loop
# or
uv run python -m rick_and_morty_agent_loop
```

It opens a prompt and keeps answering until you leave:

```
[claude] claude-opus-5
Ask about Rick and Morty -- e.g. 'Which main character am I like if I like salads'
exit/q/quit to leave, clear/new/reset to start over.

> who is beth smith

[turn 1/20] messages=1
  usage in=4,200 wrote=4,100 out=95 | 1.3s | $0.0285
  -> search_characters({"name":"beth smith"})
[turn 2/20] messages=3
  usage in=4,800 cached=4,100 out=210 | 2.0s | $0.0133
Beth Smith is Rick's daughter, a human from Earth (Replacement Dimension)...
-- 2 turns | 9,305 tokens (in 9,000 / out 305) | 45% cached | $0.0418 | 3.3s

> what episodes is she in
```

The conversation carries between questions, so follow-ups like that one work.
`new` forgets it and starts fresh; `exit` leaves and prints the session total.
A question that fails or that you Ctrl-C is rolled back, so it leaves nothing
behind for the next one to trip over. There are no command-line flags —
everything is configured in `.env`.

### The turn budget

A question gets 20 turns, and the tools are only there for the first 13. On turn
14 the loop says so in the conversation and stops offering tools, so the model
writes its answer from what it already has instead of spending its last turn on
one more search. The six turns behind the deadline are slack: if that first
attempt comes back blank there is room to nudge and try again, so a run reaches
`hit max turns without an answer` only if every one of them did. `MAX_TURNS` and
`RESERVE_TURNS` in `main.py` set both numbers, and the turn the answer falls due
is the difference:

```
[turn 13/20] messages=25
  -> search_episodes({"episode":"S03"})
[turn 14/20] messages=27 -- answer due, tools off
```

### What the tracing shows

Per turn: tokens in and out, cache reads (`cached=`) and writes (`wrote=`), wall
time, and cost. Then a total per question, and one for the session on the way
out. Cost comes from the provider when it reports one, otherwise from
[genai-prices](https://github.com/pydantic/genai-prices); a model that isn't in
the price list just shows no figure, and a total that mixes priced and unpriced
turns is marked `~$` to say it is a floor rather than the bill.

## Overriding the provider or model

Each provider runs a default model:

| provider     | key                  | default model                        |
| ------------ | -------------------- | ------------------------------------ |
| `openrouter` | `OPENROUTER_API_KEY` | `inclusionai/ling-3.0-flash-vl:free` |
| `openai`     | `OPENAI_API_KEY`     | `gpt-5.2`                            |
| `claude`     | `ANTHROPIC_API_KEY`  | `claude-opus-5`                      |

Two optional settings in `.env` change that:

```sh
MODEL=claude-haiku-4-5   # a different model on whichever provider was picked
PROVIDER=claude          # only needed if several keys are set, to say which wins
```

With several keys set and no `PROVIDER`, they break in the order above. Keys are
checked before the first request, so a missing one is a single line rather than
a stack trace. To change a provider's default model, edit its entry in
`client.py`.

## Layout

- `src/rick_and_morty_agent_loop/main.py` — the agent loop, the tools, the prompt
- `src/rick_and_morty_agent_loop/client.py` — the three providers, their models and keys
- add deps with `uv add <pkg>`

## Dependencies

- `pydantic-ai-slim[anthropic,openai,openrouter]` — model clients and message types
- `ramapi` — the Rick and Morty API client the tools wrap
- `python-dotenv` — loads `.env`
