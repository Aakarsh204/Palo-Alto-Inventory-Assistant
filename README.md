# CafeTrack Inventory Assistant

Candidate Name: Aakarsh Pathak    
Scenario Chosen: Green-Tech Inventory Assistant     
Estimated Time Spent: 7 hours

## Quick Start

### Prerequisites
- Python 3.11+
- pip
- Virtual environment support (venv)
- Flask
- Optional but recommended: Groq API key for AI insights and integration tests

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set environment variables:

1. Copy .env.example to .env

    ```powershell
    cp .env.example .env
    ```

2. Fill required values, especially GROQ_API_KEY if AI insights are needed

    Example .env values:

    ```env
    GROQ_API_KEY=YOUR_PERSONAL_GROQ_API_KEY
    FLASK_ENV=development
    FLASK_APP=app.py
    DEVELOPMENT_MODE=False
    ```

### Run Commands

Start the Flask app:

```powershell
python app.py
```

Open the app at:

http://127.0.0.1:5000

### Test Commands

Run all non-integration tests (recommended default):

```powershell
pytest -m "not integration"
```

Run all tests, including external API integration checks:

```powershell
pytest tests/
```

Run only Groq connectivity checks:

```powershell
pytest tests/test_api_connectivity.py
```

## AI Disclosure

- Did you use an AI assistant (Copilot, ChatGPT, etc.)? (Yes/No)
	- Yes, Github Copilot with Claude models.
- How did you verify the suggestions?
	- Verified by running targeted pytest suites for prediction/fallback/usage modules.
	- Reviewed code paths manually for data loading, route behavior, and fallback safety.
	- Checked environment-dependent behavior (API-key and network requirements) before accepting changes.
- Give one example of a suggestion you rejected or changed:
	- Rejected a suggestion to run only full integration tests by default; changed to non-integration tests as the default command to keep local verification fast and reliable without requiring external API/network access.
    - Rejected a design suggestion to have a primarily Groq generated summary, instead using a RAG approach by using Machine Learning to generate usage reports and providing the results to Groq in a structured prompt.
    - Rejected a suggestion to generate random data, instead relying on realistic trends (such as more consumption on weekends) over a 6 month period for the data generation.

## Tradeoffs & Prioritization

- What did you cut to stay within the 4-6 hour limit?
	- Production-grade auth and role-based access controls.
	- Full deployment setup (containerization/CI pipeline/infrastructure-as-code).
	- Advanced frontend polish and accessibility audit.
    - Scraping from static but real sites for supplier alternatives.

- What would you build next if you had more time?
	- Add persistent database storage with migrations instead of CSV-backed storage.
	- Add background jobs for scheduled forecasting and supplier scraping refresh.
	- Add richer observability (structured logging, metrics, and API call tracing).
	- Expand tests with end-to-end browser tests and negative/edge-case coverage.

- Known limitations:
	- Data persistence is CSV-based and not concurrency-safe for multi-user production workloads.
	- AI insights depend on external Groq API availability and valid credentials.
	- Integration tests can fail in restricted corporate networks due to DNS/SSL/proxy constraints.
	- The app currently runs Flask debug mode in the default local entrypoint.
    - The machine learning algorithms require a minimum threshold of data for accurate insights and forecasting.
    - The same algorithms do not have 100% accuracy and may generate slightly misleading results.
