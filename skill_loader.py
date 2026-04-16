"""
Layout (matches Cursor Agent Skills):
  <skills_root>/<skill-name>/
    SKILL.md              # required — YAML frontmatter + markdown body
    reference.md          # optional — long-form docs (appended when bundling)
    examples.md           # optional — examples (appended when bundling)
    scripts/              # optional — helper scripts (paths only; not executed here)

Frontmatter (required keys, same rules as Cursor):
  name: lowercase-hyphen-id
  description: non-empty, used for routing / discovery (optional to inject into LLM)

Templating in SKILL.md / reference.md / examples.md:
  Use ``{{variable_name}}`` placeholders; pass them to ``.render(...)``.

Optional dependency: PyYAML (recommended). Without it, only simple ``key: value`` frontmatter
lines are supported (no folded blocks like ``description: >-``).
"""

from __future__ import annotations # 让Python支持延迟类型注解，写类型提示更方便，不会报未定义错误。

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - runtime optional
    yaml = None  # type: ignore


_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}") # 匹配全部{{variable}}


def _project_default_skills_root() -> Path:
    """Default: ``<this_file_dir>/.cursor/skills`` 自动找到技能文件夹：当前代码文件所在目录 → .cursor/skills."""
    return Path(__file__).resolve().parent / "skills"


def _split_frontmatter(markdown_text: str) -> Tuple[Dict[str, Any], str]:
    text = markdown_text.lstrip("\ufeff") # 去掉UTF隐藏字符
    if not text.startswith("---"):
        return {}, text
    # 查找 --- 开头和结尾
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    closing: Optional[int] = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing = i
            break
    if closing is None:
        return {}, text

    frontmatter_raw = "\n".join(lines[1:closing]) # 截取name,description元数据
    body = "\n".join(lines[closing + 1 :]) # 截取后面的正文（---下面）
    if body.startswith("\n"):
        body = body[1:]

    # 有YAML库就用，没有就用简易解析
    meta: Dict[str, Any]
    if yaml is not None:
        loaded = yaml.safe_load(frontmatter_raw)
        meta = dict(loaded) if isinstance(loaded, dict) else {}
    else:
        meta = _parse_simple_key_value_yaml(frontmatter_raw)

    return meta, body


def _parse_simple_key_value_yaml(raw: str) -> Dict[str, Any]:
    """
    Fallback when PyYAML is not installed: ``name: x`` / ``description: ...`` per line.
    Continuation lines for description: following lines indented by 2+ spaces merge.
    """
    out: Dict[str, Any] = {}
    current_key: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal current_key, buf
        if current_key is None:
            return
        val = "\n".join(buf).strip()
        out[current_key] = val
        current_key = None
        buf = []

    for line in raw.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped and not line.startswith(" ") and not line.startswith("\t"):
            flush()
            key, rest = stripped.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            current_key = key
            buf = [rest] if rest else []
        elif current_key is not None and (line.startswith("  ") or line.startswith("\t")):
            buf.append(line.strip() if line.strip() == "" else line.rstrip())
        elif ":" in stripped:
            flush()
            key, rest = stripped.split(":", 1)
            current_key = key.strip()
            buf = [rest.strip()] if rest.strip() else []
    flush()
    return out


def _validate_frontmatter(meta: Mapping[str, Any], skill_dir: Path) -> None:
    '''校验skill配置是否合法'''
    name = meta.get("name")
    desc = meta.get("description")
    if not name or not isinstance(name, str): # name存在检查
        raise ValueError(f"Skill {skill_dir}: frontmatter missing non-empty 'name' (str).")
    if len(name) > 64: # name长度检查
        raise ValueError(f"Skill {skill_dir}: 'name' exceeds 64 characters (Cursor rule).")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name): # name必须为数字、小写、横杠
        raise ValueError(
            f"Skill {skill_dir}: 'name' must be lowercase letters, numbers, hyphens only "
            f"(Cursor rule). Got: {name!r}"
        )
    if not desc or not isinstance(desc, str): # description存在检查
        raise ValueError(f"Skill {skill_dir}: frontmatter missing non-empty 'description' (str).")
    if not desc.strip():
        raise ValueError(f"Skill {skill_dir}: 'description' must be non-empty.")
    if len(desc) > 1024: # description长度检查
        raise ValueError(f"Skill {skill_dir}: 'description' exceeds 1024 characters (Cursor rule).")


def _read_if_exists(path: Path) -> Optional[str]:
    '''读取文件'''
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def render_template(text: str, variables: Mapping[str, Any]) -> str:
    """Replace ``{{var}}`` with ``str(variables['var'])``. Unknown keys stay unchanged."""

    def repl(m: re.Match[str]) -> str:
        key = m.group(1) # 拿到变量名
        if key not in variables:
            return m.group(0) # 不认识的变量保留原样
        val = variables[key]
        if val is None:
            return ""
        return str(val)

    return _PLACEHOLDER.sub(repl, text)


@dataclass(frozen=True) # 不可修改（frozen=True）
class CursorSkill:
    """Loaded Cursor-isomorphic skill directory."""

    skill_dir: Path # 目录
    frontmatter: Dict[str, Any] # name和description
    skill_md_body: str # 正文
    reference_md: Optional[str] = None # 参考
    examples_md: Optional[str] = None # 例子
    script_paths: Tuple[Path, ...] = field(default_factory=tuple) # 脚本路径

    @property
    def name(self) -> str:
        return str(self.frontmatter["name"])

    @property
    def description(self) -> str:
        return str(self.frontmatter["description"])

    def bundled_markdown(
        self,
        *,
        include_reference: bool = True,
        include_examples: bool = True,
        include_description_header: bool = False,
    ) -> str:
        """
        把 SKILL.md + reference.md + examples.md 合并成一个完整文档

        If ``include_description_header`` is True, prepends a short machine-readable header
        with name + description (useful for multi-skill routing prompts). Cursor itself uses
        description mainly for discovery, not always injected verbatim.
        """
        parts: List[str] = []
        if include_description_header:
            parts.append(
                f"<!-- skill: {self.name} -->\n"
                f"<!-- skill-description: {self.description} -->\n"
            )
        parts.append(self.skill_md_body.rstrip())
        if include_reference and self.reference_md:
            parts.append("\n\n## Reference material (reference.md)\n\n")
            parts.append(self.reference_md.rstrip())
        if include_examples and self.examples_md:
            parts.append("\n\n## Examples (examples.md)\n\n")
            parts.append(self.examples_md.rstrip())
        return "\n".join(parts).strip() + "\n"

    def render(
        self,
        variables: Optional[Mapping[str, Any]] = None,
        *,
        include_reference: bool = True,
        include_examples: bool = True,
        include_description_header: bool = False,
    ) -> str:
        """渲染变量(render_template) + 打包(self.bundled_markdown)"""
        base = self.bundled_markdown(
            include_reference=include_reference,
            include_examples=include_examples,
            include_description_header=include_description_header,
        )
        if not variables:
            return base
        return render_template(base, variables)

    def to_messages(
        self,
        *, # * 表示：它后面的所有参数，必须用「关键字 = 值」的形式传参，不能用位置传参。
        variables: Optional[Mapping[str, Any]] = None,
        user_content: str,
        include_description_header: bool = False,
    ) -> List[Dict[str, str]]:
        '''转成 OpenAI 格式消息'''
        """Convenience: OpenAI-style ``[{"role":"system",...},{"role":"user",...}]``."""
        system = self.render(
            variables,
            include_description_header=include_description_header,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]


class CursorSkillLoader:
    def __init__(self, skills_root: Optional[Union[str, Path]] = None) -> None:
        self.skills_root = Path(skills_root) if skills_root is not None else _project_default_skills_root() # 加载skills目录

    def skill_path(self, skill_name: str) -> Path:
        # skill_name is directory name under skills_root, e.g. "paper-keyword"
        if not skill_name or any(c in skill_name for c in ("/", "\\", "..")):
            raise ValueError(f"Invalid skill_name: {skill_name!r}")
        return self.skills_root / skill_name

    def load(self, skill_name: str) -> CursorSkill:
        d = self.skill_path(skill_name)
        skill_file = d / "SKILL.md"
        if not skill_file.is_file():
            raise FileNotFoundError(f"Missing SKILL.md for skill {skill_name!r}: {skill_file}")

        raw = skill_file.read_text(encoding="utf-8") # 打开 SKILL.md
        meta, body = _split_frontmatter(raw) # 拆分配置 + 正文
        _validate_frontmatter(meta, d) # 验证配置

        ref = _read_if_exists(d / "reference.md") # 读取参考、示例、脚本
        ex = _read_if_exists(d / "examples.md")

        scripts_dir = d / "scripts"
        script_paths: Tuple[Path, ...] = ()
        if scripts_dir.is_dir(): # .is_dir()：是不是一个存在的文件夹 .is_file()：是不是一个文件
            script_paths = tuple(sorted(p for p in scripts_dir.rglob("*") if p.is_file())) # .rglob("*")：递归遍历文件夹下所有东西（文件 + 子文件夹 + 子子文件夹）

        return CursorSkill(
            skill_dir=d.resolve(),
            frontmatter=dict(meta),
            skill_md_body=body,
            reference_md=ref,
            examples_md=ex,
            script_paths=script_paths,
        )

    def list_skills(self) -> List[str]:
        """Directory names under ``skills_root`` that contain ``SKILL.md``."""
        if not self.skills_root.is_dir():
            return []
        names: List[str] = []
        for child in sorted(self.skills_root.iterdir()): # .iterdir(): 遍历目录里的所有内容（文件 / 文件夹）
            if child.is_dir() and (child / "SKILL.md").is_file():
                names.append(child.name)
        return names

    def load_all_metadata(self) -> List[Dict[str, str]]:
        """For a router model: name + description of every installed skill."""
        rows: List[Dict[str, str]] = []
        for name in self.list_skills():
            sk = self.load(name)
            rows.append({"name": sk.name, "description": sk.description, "dir": name})
        return rows


def discover_skills(skills_root: Optional[Union[str, Path]] = None) -> Iterator[str]:
    '''获取所有技能名'''
    yield from CursorSkillLoader(skills_root).list_skills()

def build_keyword_messages(
    skill_dir_name: str,
    *,
    render: Dict[str, str] = None,
    skills_root: Path | None = None,
    user_input: str = None,
    include_skill_discovery_header: bool = False,
) -> List[Dict[str, str]]:
    loader = CursorSkillLoader(skills_root)
    skill = loader.load(skill_dir_name)
    system = skill.render(
        render,
        include_reference=True,
        include_examples=True,
        include_description_header=include_skill_discovery_header,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


if __name__ == '__main__':
    print(build_keyword_messages('paper-keyword'))
