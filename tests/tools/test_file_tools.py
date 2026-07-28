import pytest
from pathlib import Path
from yapa.tools.core.read_file import read_file
from yapa.tools.core.grep import grep
from yapa.tools.core.list_dir import list_dir


class TestReadFile:
    async def test_reads_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await read_file.execute(path=str(f))
        assert result == "hello world"

    async def test_reads_with_limit(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await read_file.execute(path=str(f), limit=2)
        assert result == "line1\nline2\n"

    async def test_reads_with_offset(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await read_file.execute(path=str(f), offset=2)
        assert result == "line2\nline3\n"

    async def test_file_not_found(self):
        result = await read_file.execute(path="/nonexistent/file.txt")
        assert "Error" in result or "No such file" in result

    async def test_name_and_metadata(self):
        assert read_file.name == "read_file"
        assert read_file.needs_approval is False


class TestGrep:
    async def test_finds_pattern(self, tmp_path: Path):
        d = tmp_path / "sub"
        d.mkdir()
        f = d / "test.txt"
        f.write_text("apple\nbanana\napple pie\n")
        result = await grep.execute(pattern="apple", path=str(d))
        assert "test.txt" in result

    async def test_no_match(self, tmp_path: Path):
        d = tmp_path / "sub"
        d.mkdir()
        f = d / "test.txt"
        f.write_text("hello")
        result = await grep.execute(pattern="zzzz", path=str(d))
        assert not result or result == ""

    async def test_name_and_metadata(self):
        assert grep.name == "grep"
        assert grep.needs_approval is False


class TestListDir:
    async def test_lists_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = await list_dir.execute(path=str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result

    async def test_name_and_metadata(self):
        assert list_dir.name == "list_dir"
        assert list_dir.needs_approval is False