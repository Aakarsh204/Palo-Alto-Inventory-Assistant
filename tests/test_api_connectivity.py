import os
import socket
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')


class TestGroqApiConnectivity:
    """Diagnostics for Groq API availability and configuration."""

    def test_groq_api_key_present(self):
        """Fail fast when GROQ_API_KEY is missing or malformed."""
        api_key = os.environ.get('GROQ_API_KEY', '').strip()

        assert api_key, (
            'GROQ_API_KEY is missing. Add it to .env and retry. '
            'Expected format: gsk_...'
        )
        assert api_key.startswith('gsk_'), (
            'GROQ_API_KEY looks invalid (must start with gsk_). '
            'Please verify the key in .env.'
        )

    def test_groq_sdk_import_and_client_init(self):
        """Verify groq SDK import and basic client creation."""
        try:
            from groq import Groq
        except Exception as exc:
            pytest.fail(
                'Could not import groq SDK. Install dependencies with '
                '`pip install -r requirements.txt`. '
                f'Import error: {type(exc).__name__}: {exc}'
            )

        api_key = os.environ.get('GROQ_API_KEY', '').strip()
        try:
            client = Groq(api_key=api_key)
        except Exception as exc:
            pytest.fail(
                'Groq client initialization failed. '
                f'{type(exc).__name__}: {exc}'
            )

        assert client is not None

    def test_dns_and_tcp_to_groq(self):
        """Check DNS resolution and TCP connectivity to api.groq.com:443."""
        host = 'api.groq.com'

        try:
            resolved_ip = socket.gethostbyname(host)
        except Exception as exc:
            pytest.fail(
                f'DNS resolution failed for {host}. '
                f'{type(exc).__name__}: {exc}'
            )

        assert resolved_ip, 'DNS returned empty IP for api.groq.com'

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            result = sock.connect_ex((host, 443))
        finally:
            sock.close()

        assert result == 0, (
            f'TCP connection to {host}:443 failed (connect_ex={result}). '
            'This usually indicates firewall/proxy/network restrictions.'
        )

    @pytest.mark.integration
    def test_chat_completion_roundtrip(self):
        """Perform a real minimal chat completion call to Groq."""
        from groq import Groq

        api_key = os.environ.get('GROQ_API_KEY', '').strip()
        client = Groq(api_key=api_key)

        model_candidates = [
            'llama-3.1-8b-instant',
            'llama3-8b-8192',
        ]

        last_error = None
        completion = None
        used_model = None

        for model_name in model_candidates:
            for attempt in range(1, 4):
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {'role': 'system', 'content': 'You are concise.'},
                            {'role': 'user', 'content': 'Reply with the single word: ok'}
                        ],
                        max_tokens=10,
                        temperature=0
                    )
                    used_model = model_name
                    break
                except Exception as exc:
                    last_error = exc
                    error_name = type(exc).__name__.lower()
                    is_connection_error = (
                        'connection' in str(exc).lower() or
                        'connect' in error_name
                    )

                    if is_connection_error and attempt < 3:
                        time.sleep(0.5 * attempt)
                    else:
                        break

            if completion is not None:
                break

        if completion is None:
            cause = getattr(last_error, '__cause__', None)
            detail = f'{type(last_error).__name__}: {last_error}'
            if cause is not None:
                detail += f' | cause={type(cause).__name__}: {cause}'

            remediation = ''
            lower_detail = detail.lower()
            if 'certificate_verify_failed' in lower_detail or 'ssl' in lower_detail:
                remediation = (
                    ' SSL trust issue detected. If you are behind a corporate proxy/SSL inspection, '
                    'install python-certifi-win32 and restart the environment, or set GROQ_CA_BUNDLE '
                    'to your corporate root CA PEM path. As a temporary diagnostic only, you may set '
                    'GROQ_SKIP_SSL_VERIFY=true.'
                )

            pytest.fail(
                'Groq completion request failed for all candidate models. '
                f'Last error: {detail}.{remediation}'
            )

        content = ''
        if completion.choices and completion.choices[0].message:
            content = (completion.choices[0].message.content or '').strip()

        assert content, (
            f'Groq responded with empty content using model {used_model}. '
            'Request reached API but no text was returned.'
        )
