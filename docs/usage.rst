Usage
=====

Simple Chat Completions (OpenAI)
--------------------------------

The repository includes a simple example at ``example/simple-chat-completion/chat.py``.

Key lines::

    from aiobs import observer

    observer.observe()
    # Call OpenAI Chat Completions via openai>=1
    observer.end()
    observer.flush()

Gemini Generate Content
-----------------------

Example using Google Gemini at ``example/gemini/main.py``.

Key lines::

    from aiobs import observer
    from google import genai

    observer.observe()

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain quantum computing"
    )

    observer.end()
    observer.flush()

Function Tracing with @observe
------------------------------

Trace any function (sync or async) with the ``@observe`` decorator::

    from aiobs import observer, observe

    @observe
    def research(query: str) -> list:
        # your logic here
        return results

    @observe(name="custom_name")
    async def fetch_data(url: str) -> dict:
        # async logic here
        return data

    observer.observe(session_name="my-pipeline")
    research("What is an API?")
    observer.end()
    observer.flush()

Decorator Options
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``name``
     - function name
     - Custom display name for the traced function
   * - ``capture_args``
     - ``True``
     - Whether to capture function arguments
   * - ``capture_result``
     - ``True``
     - Whether to capture the return value

Examples::

    # Don't capture sensitive arguments
    @observe(capture_args=False)
    def login(username: str, password: str):
        ...

    # Don't capture large return values
    @observe(capture_result=False)
    def get_large_dataset():
        ...

Pipeline Example
----------------

Chained tasks with multiple API calls::

    python -m example.pipeline.main "Your prompt here"

This runs a three-step pipeline (research → summarize → critique) and writes a single JSON file with all events.

Output
------

By default, ``observer.flush()`` writes ``./llm_observability.json``. Override with the ``LLM_OBS_OUT`` environment variable::

    LLM_OBS_OUT=/path/to/output.json python your_script.py

What Gets Captured
------------------

For each LLM API call:

- **Provider**: ``openai`` or ``gemini``
- **API**: e.g., ``chat.completions`` or ``models.generateContent``
- **Request**: model, messages/contents, core parameters
- **Response**: text, model, token usage (when available)
- **Timing**: start/end timestamps, ``duration_ms``
- **Errors**: exception name and message if the call fails
- **Callsite**: file path, line number, and function name where the API was called

For decorated functions (``@observe``):

- Function name and module
- Input arguments (args/kwargs)
- Return value
- Timing: start/end timestamps, ``duration_ms``
- Errors: exception name and message if the call fails
- Callsite: file path, line number where the function was defined
