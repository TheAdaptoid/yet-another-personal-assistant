from pathlib import Path

from yapa.tools.core.bash import bash
from yapa.tools.core.edit_file import edit_file
from yapa.tools.core.grep import grep
from yapa.tools.core.list_dir import list_dir
from yapa.tools.core.read_file import read_file
from yapa.tools.core.write_file import write_file


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


class TestWriteFile:
    async def test_writes_file(self, tmp_path: Path):
        f = tmp_path / "out.txt"
        result = await write_file.execute(path=str(f), content="hello")
        assert f.read_text() == "hello"
        assert result == "ok"

    async def test_refuses_when_parent_missing(self, tmp_path: Path):
        f = tmp_path / "missing" / "out.txt"
        result = await write_file.execute(path=str(f), content="hello")
        assert "parent directory does not exist" in result
        assert not f.exists()

    async def test_name_and_metadata(self):
        assert write_file.name == "write_file"
        assert write_file.needs_approval is True


class TestBash:
    async def test_runs_command(self):
        result = await bash.execute(command="echo hello")
        assert "hello" in result

    async def test_failing_command(self):
        result = await bash.execute(command="exit 1")
        assert "exit code 1" in result

    async def test_name_and_metadata(self):
        assert bash.name == "bash"
        assert bash.needs_approval is True


class TestEditFile:
    async def test_replaces_string(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await edit_file.execute(
            path=str(f), old_string="world", new_string="there"
        )
        assert result == "ok"
        assert f.read_text() == "hello there"

    async def test_string_not_found(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = await edit_file.execute(
            path=str(f), old_string="zzz", new_string="aaa"
        )
        assert "could not find" in result

    async def test_name_and_metadata(self):
        assert edit_file.name == "edit_file"
        assert edit_file.needs_approval is True
