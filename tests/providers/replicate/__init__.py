"""
GenOps Replicate Provider Tests

Comprehensive test suite for Replicate integration following CLAUDE.md standards.
Includes 125+ tests across all components with real-world scenario simulation.

Test Structure:
- Unit Tests (~35): Individual component validation
- Integration Tests (~17): End-to-end workflow verification
- Cost Aggregation Tests (~24): Multi-model cost tracking accuracy
- Validation Tests (~33): Setup diagnostics and error handling
- Pricing Tests (~30): All model type cost calculations

Coverage:
- All provider functionality with edge cases
- Multi-modal model support (text, image, video, audio)
- Cost calculation accuracy across pricing models
- Error handling and graceful degradation
- Performance benchmarking and optimization
- Production deployment scenarios
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

# Test configuration
TEST_CONFIG = {
    "use_mock_responses": True,  # Use mocks by default
    "replicate_api_token": "r8_test_token_for_unit_tests_only",
    "timeout_seconds": 10,
    "max_retry_attempts": 3
}

# Mock response templates for testing
MOCK_RESPONSES = {
    "text_generation": {
        "content": "This is a test response from the mocked Replicate text model.",
        "model": "meta/llama-2-7b-chat",
        "tokens_used": 150,
        "processing_time_ms": 1200
    },
    "image_generation": {
        "content": ["https://example.com/generated_image.png"],
        "model": "black-forest-labs/flux-schnell",
        "images_generated": 1,
        "processing_time_ms": 3000
    },
    "video_generation": {
        "content": ["https://example.com/generated_video.mp4"],
        "model": "google/veo-2",
        "duration_seconds": 5.0,
        "processing_time_ms": 15000
    },
    "audio_processing": {
        "content": "This is the transcribed text from the audio.",
        "model": "openai/whisper",
        "audio_duration_seconds": 30.0,
        "processing_time_ms": 2500
    }
}
