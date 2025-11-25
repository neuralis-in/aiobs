Getting Started
===============

Install
-------

- Core only::

    pip install aiobs

- With OpenAI support::

    pip install aiobs[openai]

- With Gemini support::

    pip install aiobs[gemini]

- With all providers and examples::

    pip install aiobs[openai,gemini,example]

Configure
---------

Create a ``.env`` at your project root with provider credentials:

**OpenAI**::

    OPENAI_API_KEY=your_key_here
    # Optional
    OPENAI_MODEL=gpt-4o-mini

**Google Gemini**::

    GOOGLE_API_KEY=your_key_here
    # Or use service account credentials
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

Hello, Observability
--------------------

Minimal example with OpenAI::

    from aiobs import observer
    from openai import OpenAI

    observer.observe()

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    observer.end()
    observer.flush()  # writes llm_observability.json

Minimal example with Gemini::

    from aiobs import observer
    from google import genai

    observer.observe()

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hello!"
    )

    observer.end()
    observer.flush()  # writes llm_observability.json
