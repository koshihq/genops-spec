"""Tests for the framework detection system."""

from unittest.mock import Mock, patch

import pytest

from genops.providers.base.detector import (
    FrameworkDetector,
    FrameworkInfo,
    detect_frameworks,
    get_framework_detector,
    is_framework_available,
)


class TestFrameworkInfo:
    """Test FrameworkInfo class."""

    def test_framework_info_creation(self):
        """Test FrameworkInfo object creation."""
        info = FrameworkInfo(
            name="test_framework",
            import_path="test.framework",
            version="1.0.0",
            framework_type="testing",
            available=True
        )

        assert info.name == "test_framework"
        assert info.import_path == "test.framework"
        assert info.version == "1.0.0"
        assert info.framework_type == "testing"
        assert info.available is True

    def test_framework_info_string_repr(self):
        """Test string representations."""
        available_info = FrameworkInfo("test", "test", "1.0.0", "testing", True)
        unavailable_info = FrameworkInfo("test", "test", None, "testing", False)

        assert "✓ test (v1.0.0) [testing]" in str(available_info)
        assert "✗ test [testing]" in str(unavailable_info)


class TestFrameworkDetector:
    """Test FrameworkDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = FrameworkDetector()

    def test_detector_initialization(self):
        """Test detector initializes with known frameworks."""
        assert "langchain" in self.detector.FRAMEWORKS
        assert "torch" in self.detector.FRAMEWORKS
        assert "tensorflow" in self.detector.FRAMEWORKS
        assert "transformers" in self.detector.FRAMEWORKS

    def test_framework_registry_structure(self):
        """Test that framework registry has required fields."""
        for _name, config in self.detector.FRAMEWORKS.items():
            assert "import_path" in config
            assert "version_attr" in config
            assert "framework_type" in config
            assert "description" in config

    def test_get_framework_types(self):
        """Test getting unique framework types."""
        types = self.detector.get_framework_types()

        assert isinstance(types, set)
        assert "orchestration" in types
        assert "training" in types
        assert "inference" in types

    @patch('importlib.import_module')
    def test_detect_framework_available(self, mock_import):
        """Test detecting an available framework."""
        # Mock successful import
        mock_module = Mock()
        mock_module.__version__ = "1.0.0"
        mock_import.return_value = mock_module

        config = {
            "import_path": "test_framework",
            "version_attr": "__version__",
            "framework_type": "testing"
        }

        result = self.detector.detect_framework("test_framework", config)

        assert result.name == "test_framework"
        assert result.available is True
        assert result.version == "1.0.0"
        assert result.framework_type == "testing"
        assert result.module_obj is mock_module

    @patch('importlib.import_module')
    def test_detect_framework_unavailable(self, mock_import):
        """Test detecting an unavailable framework."""
        # Mock import error
        mock_import.side_effect = ImportError("Module not found")

        config = {
            "import_path": "missing_framework",
            "version_attr": "__version__",
            "framework_type": "testing"
        }

        result = self.detector.detect_framework("missing_framework", config)

        assert result.name == "missing_framework"
        assert result.available is False
        assert result.version is None
        assert result.module_obj is None

    @patch('importlib.import_module')
    def test_detect_framework_no_version(self, mock_import):
        """Test detecting framework without version attribute."""
        # Mock module without version
        mock_module = Mock()
        del mock_module.__version__  # Ensure no version attribute
        mock_import.return_value = mock_module

        config = {
            "import_path": "no_version_framework",
            "version_attr": "__version__",
            "framework_type": "testing"
        }

        result = self.detector.detect_framework("no_version_framework", config)

        assert result.available is True
        assert result.version is None

    @patch('genops.providers.base.detector.FrameworkDetector.detect_framework')
    def test_detect_all_frameworks(self, mock_detect):
        """Test detecting all frameworks."""
        # Mock detection results
        mock_detect.return_value = FrameworkInfo("test", "test", "1.0.0", "testing", True)

        results = self.detector.detect_all_frameworks()

        assert isinstance(results, dict)
        assert len(results) > 0
        # Should be called for each framework in registry
        assert mock_detect.call_count == len(self.detector.FRAMEWORKS)

    def test_detect_all_frameworks_caching(self):
        """Test that framework detection results are cached."""
        with patch.object(self.detector, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo("test", "test", "1.0.0", "testing", True)

            # First call
            results1 = self.detector.detect_all_frameworks()
            first_call_count = mock_detect.call_count

            # Second call should use cache
            results2 = self.detector.detect_all_frameworks()

            assert results1 == results2
            assert mock_detect.call_count == first_call_count  # No additional calls

    def test_detect_all_frameworks_force_refresh(self):
        """Test forcing refresh of framework detection."""
        with patch.object(self.detector, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo("test", "test", "1.0.0", "testing", True)

            # First call
            self.detector.detect_all_frameworks()
            first_call_count = mock_detect.call_count

            # Force refresh should re-detect
            self.detector.detect_all_frameworks(force_refresh=True)

            assert mock_detect.call_count == first_call_count * 2

    @patch('genops.providers.base.detector.FrameworkDetector.detect_all_frameworks')
    def test_get_available_frameworks(self, mock_detect_all):
        """Test filtering available frameworks."""
        # Mock mixed available/unavailable frameworks
        mock_detect_all.return_value = {
            "available1": FrameworkInfo("available1", "path1", "1.0.0", "type1", True),
            "unavailable1": FrameworkInfo("unavailable1", "path2", None, "type1", False),
            "available2": FrameworkInfo("available2", "path3", "2.0.0", "type2", True),
        }

        available = self.detector.get_available_frameworks()

        assert len(available) == 2
        assert all(info.available for info in available)
        assert any(info.name == "available1" for info in available)
        assert any(info.name == "available2" for info in available)

    @patch('genops.providers.base.detector.FrameworkDetector.detect_all_frameworks')
    def test_get_available_frameworks_filtered_by_type(self, mock_detect_all):
        """Test filtering available frameworks by type."""
        mock_detect_all.return_value = {
            "framework1": FrameworkInfo("framework1", "path1", "1.0.0", "type1", True),
            "framework2": FrameworkInfo("framework2", "path2", "2.0.0", "type2", True),
            "framework3": FrameworkInfo("framework3", "path3", "3.0.0", "type1", True),
        }

        type1_frameworks = self.detector.get_available_frameworks("type1")

        assert len(type1_frameworks) == 2
        assert all(info.framework_type == "type1" for info in type1_frameworks)

    @patch('genops.providers.base.detector.FrameworkDetector.detect_all_frameworks')
    def test_is_framework_available(self, mock_detect_all):
        """Test checking if specific framework is available."""
        mock_detect_all.return_value = {
            "available": FrameworkInfo("available", "path", "1.0.0", "type", True),
            "unavailable": FrameworkInfo("unavailable", "path", None, "type", False),
        }

        assert self.detector.is_framework_available("available") is True
        assert self.detector.is_framework_available("unavailable") is False
        assert self.detector.is_framework_available("nonexistent") is False

    @patch('genops.providers.base.detector.FrameworkDetector.detect_all_frameworks')
    def test_get_framework_version(self, mock_detect_all):
        """Test getting framework version."""
        mock_detect_all.return_value = {
            "versioned": FrameworkInfo("versioned", "path", "1.0.0", "type", True),
            "no_version": FrameworkInfo("no_version", "path", None, "type", True),
            "unavailable": FrameworkInfo("unavailable", "path", None, "type", False),
        }

        assert self.detector.get_framework_version("versioned") == "1.0.0"
        assert self.detector.get_framework_version("no_version") is None
        assert self.detector.get_framework_version("unavailable") is None
        assert self.detector.get_framework_version("nonexistent") is None

    def test_add_custom_framework(self):
        """Test adding custom framework to registry."""
        initial_count = len(self.detector.FRAMEWORKS)

        self.detector.add_custom_framework(
            name="custom_framework",
            import_path="custom.framework",
            framework_type="custom",
            version_attr="__version__",
            description="Custom test framework"
        )

        assert len(self.detector.FRAMEWORKS) == initial_count + 1
        assert "custom_framework" in self.detector.FRAMEWORKS

        config = self.detector.FRAMEWORKS["custom_framework"]
        assert config["import_path"] == "custom.framework"
        assert config["framework_type"] == "custom"
        assert config["description"] == "Custom test framework"

    def test_add_custom_framework_clears_cache(self):
        """Test that adding custom framework clears detection cache."""
        # First populate cache
        with patch.object(self.detector, 'detect_framework') as mock_detect:
            mock_detect.return_value = FrameworkInfo("test", "test", "1.0.0", "testing", True)
            self.detector.detect_all_frameworks()

        # Cache should be populated
        assert self.detector._detected_frameworks is not None

        # Add custom framework
        self.detector.add_custom_framework("custom", "custom.path", "custom")

        # Cache should be cleared
        assert self.detector._detected_frameworks is None


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_framework_detector_singleton(self):
        """Test that get_framework_detector returns singleton."""
        detector1 = get_framework_detector()
        detector2 = get_framework_detector()

        assert detector1 is detector2
        assert isinstance(detector1, FrameworkDetector)

    @patch('genops.providers.base.detector.get_framework_detector')
    def test_detect_frameworks_convenience(self, mock_get_detector):
        """Test detect_frameworks convenience function."""
        mock_detector = Mock()
        mock_detector.detect_all_frameworks.return_value = {"test": "result"}
        mock_get_detector.return_value = mock_detector

        result = detect_frameworks()

        assert result == {"test": "result"}
        mock_detector.detect_all_frameworks.assert_called_once()

    @patch('genops.providers.base.detector.get_framework_detector')
    def test_is_framework_available_convenience(self, mock_get_detector):
        """Test is_framework_available convenience function."""
        mock_detector = Mock()
        mock_detector.is_framework_available.return_value = True
        mock_get_detector.return_value = mock_detector

        result = is_framework_available("test_framework")

        assert result is True
        mock_detector.is_framework_available.assert_called_once_with("test_framework")


@pytest.fixture
def sample_frameworks():
    """Sample framework data for testing."""
    return {
        "langchain": {
            "import_path": "langchain",
            "version_attr": "__version__",
            "framework_type": "orchestration",
            "description": "LLM application orchestration framework"
        },
        "torch": {
            "import_path": "torch",
            "version_attr": "__version__",
            "framework_type": "training",
            "description": "PyTorch deep learning framework"
        }
    }


class TestFrameworkDetectorIntegration:
    """Integration tests for framework detector."""

    def test_real_framework_detection(self):
        """Test detection with real Python modules (if available)."""
        detector = FrameworkDetector()

        # Try to detect some common modules that should be available
        test_modules = {
            "os": {"import_path": "os", "version_attr": "__version__", "framework_type": "builtin"},
            "sys": {"import_path": "sys", "version_attr": "version", "framework_type": "builtin"},
        }

        for name, config in test_modules.items():
            result = detector.detect_framework(name, config)
            assert result.available is True  # These should always be available
            assert result.module_obj is not None

    def test_detection_with_missing_modules(self):
        """Test detection with modules that definitely don't exist."""
        detector = FrameworkDetector()

        config = {
            "import_path": "definitely_does_not_exist_module_12345",
            "version_attr": "__version__",
            "framework_type": "nonexistent"
        }

        result = detector.detect_framework("nonexistent", config)
        assert result.available is False
        assert result.module_obj is None
