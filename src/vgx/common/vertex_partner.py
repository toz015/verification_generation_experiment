"""Native managed Vertex partner endpoints, sharing durable request accounting.

Sources: Google Cloud partner-models/mistral and partner-models/claude/use-claude.
No tools, prompt caching, or extended thinking are requested here.
"""
from __future__ import annotations

from vgx.common.vertex import VertexBatchRunner


class VertexPartnerRunner(VertexBatchRunner):
    def __init__(self, model, location, project, max_tokens=1024, temperature=0.0):
        super().__init__(model, location, project, max_tokens=max_tokens, temperature=temperature)
        parts = model.split('/')
        if len(parts) != 2 or parts[0] not in ('mistralai', 'anthropic'):
            raise ValueError('explicit supported publisher/model required')
        self.publisher, self.model_id = parts
        if not self.model_id or not all(c.isalnum() or c in '-.@' for c in self.model_id):
            raise ValueError('invalid partner model ID')

    @property
    def params(self):
        return {'max_tokens': self.max_tokens, 'temperature': self.temperature}

    @property
    def identity(self):
        return {'schema': 1, 'provider': 'vertex_ai_partner_raw_predict',
                'model': self.model, 'location': self.location,
                'project_sha256': __import__('hashlib').sha256(self.project.encode()).hexdigest(),
                **self.params, 'anthropic_version': 'vertex-2023-10-16' if self.publisher == 'anthropic' else None,
                'extended_thinking': False, 'prompt_caching': False}

    def _endpoint(self):
        host = 'aiplatform.googleapis.com' if self.location == 'global' else f'{self.location}-aiplatform.googleapis.com'
        return (f'https://{host}/v1/projects/{self.project}/locations/{self.location}'
                f'/publishers/{self.publisher}/models/{self.model_id}:rawPredict')

    def request_body(self, request):
        if self.publisher == 'anthropic':
            body = {'anthropic_version': 'vertex-2023-10-16', 'stream': False, **self.params,
                    'messages': [{'role': 'user', 'content': request.prompt}]}
            if request.system:
                body['system'] = request.system
            return body
        body = super().request_body(request)
        body['model'] = self.model_id
        return body

    def response_text(self, result):
        if self.publisher != 'anthropic':
            return super().response_text(result)
        content = result.get('content')
        if not isinstance(content, list):
            return ''
        return ''.join(block['text'] for block in content if isinstance(block, dict)
                       and block.get('type') == 'text' and isinstance(block.get('text'), str))
