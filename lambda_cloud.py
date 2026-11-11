"""Small Lambda API client. Credentials stay in the parent checkout's .env.

Reads only LAMBDA_API_KEY (never executes .env); never logs credentials.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

API = 'https://cloud.lambda.com/api/v1/'


def key_from_env():
    key = os.environ.get('LAMBDA_API_KEY')
    if key:
        return key
    for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
        path = parent / '.env'
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.startswith('LAMBDA_API_KEY='):
                return line.split('=', 1)[1].strip().strip('\"\'')
    raise RuntimeError('LAMBDA_API_KEY is not configured')


def request(endpoint, body=None):
    key = key_from_env()
    # Lambda documents curl as an API client. Pass its configuration through
    # stdin so neither credentials nor mutation payloads appear in process args.
    values = {'url': API + endpoint.lstrip('/'),
              'header': 'Authorization: Bearer ' + key}
    config = ''.join(f'{name} = {json.dumps(value)}\n' for name, value in values.items())
    config += 'header = "Accept: application/json"\nheader = "Content-Type: application/json"\n'
    if body is not None:
        config += 'request = "POST"\ndata = ' + json.dumps(json.dumps(body)) + '\n'
    result = subprocess.run(['curl', '--silent', '--show-error', '--max-time', '45',
                             '--config', '-', '--write-out', '\n%{http_code}'],
                            input=config, capture_output=True, text=True, timeout=50)
    text, _, status = result.stdout.rpartition('\n')
    if result.returncode or not status.startswith('2'):
        detail = (text or result.stderr).replace(key, '[REDACTED]')
        raise RuntimeError(f'Lambda HTTP {status or "unavailable"}: {detail[:1000]}')
    return json.loads(text)['data']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('endpoint', choices=['instance-types', 'instances', 'ssh-keys', 'account', 'images'])
    args = parser.parse_args()
    result = request(args.endpoint)
    if args.endpoint == 'instance-types':
        result = [dict(**row['instance_type'], regions=row.get('regions_with_capacity_available', []))
                  for name, row in result.items()]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
