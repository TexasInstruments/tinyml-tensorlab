"""Regression tests for two bugs in the compilation orchestrator (compilation.py).

1. gen_artifacts() changed the process cwd to args.output_dir but only restored
   it on the success path, not in a finally -- a compile failure left the whole
   (in-process, not subprocess) orchestrator's cwd permanently pointed at
   args.output_dir for the rest of its lifetime.
2. main() silently swallowed a failed decrypt() and proceeded as if it had
   succeeded, then unconditionally re-encrypted args.FILE regardless of whether
   it had actually been decrypted -- corrupting an already-encrypted file via
   double-encryption.
"""
import os
from argparse import Namespace
from unittest.mock import patch

import pytest

from tinyml_tinyverse.references.common import compilation


def test_gen_artifacts_restores_cwd_even_when_drive_compile_raises(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "artifacts").mkdir()
    original_cwd = os.getcwd()

    args = Namespace(output_dir=str(output_dir), output="model", keep_libc_files=True)

    with patch.object(compilation, "drive_compile", side_effect=RuntimeError("TVM blew up")):
        with pytest.raises(RuntimeError, match="TVM blew up"):
            compilation.gen_artifacts(args)

    assert os.getcwd() == original_cwd


def test_gen_artifacts_restores_cwd_on_success(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "artifacts").mkdir()
    original_cwd = os.getcwd()

    args = Namespace(output_dir=str(output_dir), output="model", keep_libc_files=True)

    with patch.object(compilation, "drive_compile"):
        compilation.gen_artifacts(args)

    assert os.getcwd() == original_cwd


def _minimal_main_args(tmp_path, generic_model):
    return Namespace(
        lis=None,
        output_dir=str(tmp_path),
        DEBUG=False,
        target="c",
        cross_compiler=str(tmp_path),  # any existing path passes the validation check
        cross_compiler_options="",
        target_cmsis_nn_mcpu="None",
        target_cmsis_nn_mattr="None",
        generic_model=generic_model,
        FILE="model.onnx",
        keep_intermittent_files=True,
    )


def test_main_aborts_without_encrypting_when_decrypt_failed(tmp_path):
    """A failed decrypt() must not be followed by an encrypt() (the file was
    never actually decrypted, so re-encrypting it would corrupt it), must not
    proceed into gen_artifacts() (compiling the still-encrypted blob can only
    produce garbage), and must report failure via a nonzero exit flag --
    matching the contract the modelmaker runners consume
    (exit_flag = compile_scr.run(args))."""
    args = _minimal_main_args(tmp_path, generic_model=False)

    with patch.object(compilation.utils, "decrypt", side_effect=RuntimeError("bad key")), \
         patch.object(compilation.utils, "encrypt") as mock_encrypt, \
         patch.object(compilation.utils, "get_crypt_key", return_value="key"), \
         patch.object(compilation, "gen_artifacts") as mock_gen_artifacts, \
         patch.object(compilation, "remove_intermittent_files"):
        assert compilation.main(args) == 1

    mock_encrypt.assert_not_called()
    mock_gen_artifacts.assert_not_called()


def test_main_encrypts_after_successful_decrypt_and_successful_compile(tmp_path):
    """Regression safety: the normal success path must still encrypt exactly once."""
    args = _minimal_main_args(tmp_path, generic_model=False)

    with patch.object(compilation.utils, "decrypt"), \
         patch.object(compilation.utils, "encrypt") as mock_encrypt, \
         patch.object(compilation.utils, "get_crypt_key", return_value="key"), \
         patch.object(compilation, "gen_artifacts"), \
         patch.object(compilation, "remove_intermittent_files"):
        compilation.main(args)

    mock_encrypt.assert_called_once_with("model.onnx", "key")


def test_main_encrypts_after_successful_decrypt_even_if_compile_fails(tmp_path):
    """Regression safety: if decrypt succeeded, the file must be re-encrypted
    even when gen_artifacts() itself fails (matches the original try/except's
    intent), and the original exception must still propagate."""
    args = _minimal_main_args(tmp_path, generic_model=False)

    with patch.object(compilation.utils, "decrypt"), \
         patch.object(compilation.utils, "encrypt") as mock_encrypt, \
         patch.object(compilation.utils, "get_crypt_key", return_value="key"), \
         patch.object(compilation, "gen_artifacts", side_effect=RuntimeError("compile failed")), \
         patch.object(compilation, "remove_intermittent_files"):
        with pytest.raises(RuntimeError, match="compile failed"):
            compilation.main(args)

    mock_encrypt.assert_called_once_with("model.onnx", "key")
