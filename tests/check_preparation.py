"""Check preparation from a Git checkout containing provenance but no corpus."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import complete_hindi_english as preparation


class PreparationChecks(unittest.TestCase):
    def test_provenance_alone_does_not_skip_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'data_english_hf'
            base.mkdir()
            (base / 'provenance.json').write_text('{}')
            with patch.object(preparation, 'ROOT', root), patch.object(
                preparation.urllib.request, 'urlretrieve', side_effect=RuntimeError('download requested')
            ) as download:
                with self.assertRaisesRegex(RuntimeError, 'download requested'):
                    preparation.prepare('english')
                download.assert_called_once()

    def test_wrong_source_hash_is_rejected_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'data_english_hf'
            base.mkdir()
            (base / 'alpaca_data_cleaned.json').write_text('[]')
            with patch.object(preparation, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Source checksum mismatch'):
                    preparation.prepare('english')
            self.assertFalse((base / 'provenance.json').exists())


if __name__ == '__main__':
    unittest.main()
