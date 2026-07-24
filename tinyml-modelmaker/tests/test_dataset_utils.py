"""Tests for dataset_utils: file listing and split creation."""

import os
import tempfile

import pytest

from tinyml_modelmaker.ai_modules.common.datasets import dataset_utils


class TestCreateFilelist:
    """Tests for create_filelist()."""

    def test_basic_listing(self, tmp_path):
        """Creates a simple directory tree and verifies the generated file list."""
        data_dir = tmp_path / "data"
        (data_dir / "classA").mkdir(parents=True)
        (data_dir / "classB").mkdir(parents=True)
        (data_dir / "classA" / "file1.csv").write_text("a")
        (data_dir / "classA" / "file2.csv").write_text("b")
        (data_dir / "classB" / "file3.csv").write_text("c")

        out_dir = str(tmp_path / "output")
        result = dataset_utils.create_filelist(str(data_dir), out_dir, ignore_str_list=[])
        assert os.path.isfile(result)

        with open(result) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_ignore_pattern(self, tmp_path):
        """Verify files matching the ignore pattern are excluded."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "keep.csv").write_text("k")
        (data_dir / "skip.log").write_text("s")

        out_dir = str(tmp_path / "output")
        result = dataset_utils.create_filelist(str(data_dir), out_dir, ignore_str_list=[r"\.log$"])
        with open(result) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        assert "keep.csv" in lines[0]


class TestCreateInterFileSplit:
    """Tests for create_inter_file_split() input validation."""

    def test_invalid_split_list_type(self, tmp_path):
        fl = tmp_path / "file_list.txt"
        fl.write_text("a.csv\nb.csv\n")
        with pytest.raises(TypeError, match="tuple or list"):
            dataset_utils.create_inter_file_split(
                str(fl), "not_a_list", 0.8
            )

    def test_split_factor_too_large(self, tmp_path):
        fl = tmp_path / "file_list.txt"
        fl.write_text("a.csv\nb.csv\n")
        split_files = (str(tmp_path / "train.txt"), str(tmp_path / "val.txt"))
        with pytest.raises(ValueError, match="less than 1"):
            dataset_utils.create_inter_file_split(str(fl), split_files, 1.5)

    def test_split_factor_list_sum_too_large(self, tmp_path):
        fl = tmp_path / "file_list.txt"
        fl.write_text("a.csv\nb.csv\n")
        split_files = (str(tmp_path / "train.txt"), str(tmp_path / "val.txt"))
        with pytest.raises(ValueError, match="<=1"):
            dataset_utils.create_inter_file_split(str(fl), split_files, [0.8, 0.5])


class TestCreateIntraFileSplit:
    """Tests for create_intra_file_split() input validation."""

    def test_invalid_split_list_type(self, tmp_path):
        fl = tmp_path / "file_list.txt"
        fl.write_text("a.csv\n")
        with pytest.raises(TypeError, match="tuple or list"):
            dataset_utils.create_intra_file_split(
                str(fl), "not_a_list", 0.8, "data", str(tmp_path), ("train",)
            )
