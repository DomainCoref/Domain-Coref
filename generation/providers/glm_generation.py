# -*- coding: utf-8 -*-
"""
GLMAPI.py — 使用智谱 GLM 生成“中文指代消解”数据集（按文档二结构 + 五类指代 + 零代词 Ø + 防污染）

目录结构：
out/
  dataset_glm.json               # 最终数据（只放这里）
  dataset_glm.txt                # 最终纯文本（只放这里）
  logs_glm/                      # 其他日志/溯源文件都放这里
    genlog_glm.jsonl
    manifest_glm.json
    gen_counter_glm.txt
    topic_memory_glm.json
    raw/
       PA-000001-GLM-cand1.txt
       ...

主要特性：
1）五个领域 + 五种指代类型（人称代词 / 指示代词 / 有定描述 / 零指代 / 事件指代）
2）零指代 Ø 严格结构约束（1 antecedent + ≥1 Ø anaphor，且句中真实含 Ø）
3）category_primary 必须真实出现在至少一条链的 type 中（防止乱标主类型）
4）2025+ 年份约束（禁止 2024 及以前具体年份出现在正文中）
5）Jaccard7 + SimHash 去重
6）自动对齐 char_start / char_end
7）二阶段验证+修复（VERIFY_PROMPT）
8）批次内主类型配额控制：BATCH_SIZE = 50，每 50 条内五种主类型尽量均衡
"""

import os
import re
import time
import random
import argparse
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple


# --- robust stdlib json import (avoid local json.py shadowing) ---
def _import_stdlib_json():
    import importlib, os as _os, sys as _sys
    # if a local json.py was imported, purge it first
    if 'json' in _sys.modules:
        _m = _sys.modules.get('json')
        _mf = getattr(_m, '__file__', '') or ''
        try:
            _here = _os.path.dirname(__file__)
        except Exception:
            _here = ''
        if _mf and _here and _os.path.abspath(_mf).startswith(_os.path.abspath(_here)) and _os.path.basename(_mf) == 'json.py':
            del _sys.modules['json']
    removed = []
    candidates = ['', _os.getcwd(), _os.path.dirname(__file__)]
    for p in candidates:
        if p in _sys.path:
            _sys.path.remove(p)
            removed.append(p)
    try:
        return importlib.import_module('json')
    finally:
        # restore paths (prepend in reverse to keep priority)
        for p in reversed(removed):
            _sys.path.insert(0, p)

json = _import_stdlib_json()
# --- end json import helper ---

from openai import OpenAI
from openai import BadRequestError, APIError

# =============================================================================
# 内容安全/敏感词触发（code=1301）处理
# =============================================================================

class ContentFilterError(RuntimeError):
    """Provider blocked the request due to content/safety filtering (e.g., code=1301)."""
    pass


def _is_content_filter_error(exc: Exception) -> bool:
    s = str(exc)
    s_low = s.lower()
    return (
        ("1301" in s)
        or ("contentfilter" in s_low)
        or ("敏感内容" in s)
        or ("不安全" in s)
    )

def _is_param_error_1210(exc: Exception) -> bool:
    """Zhipu/BigModel 常见：1210 = 参数错误（常由 base_url 误填或不支持的字段SDK字段触发）"""
    s = str(exc)
    return ("'code': '1210'" in s) or ('"code": "1210"' in s) or ("code': '1210'" in s) or ("1210" in s and "参数" in s)


_SANITIZE_MAP = {
    "加密": "安全",
    "审计": "评估",
    "执法": "检查",
    "反垄断": "竞争合规",
    "信访": "反馈",
    "维稳": "秩序维护",
    "公安": "安全部门",
    "警察": "安保人员",
    "政务": "公共服务",
    "政府": "管理部门",
    "监管": "管理",
    "跨境": "跨地区",
    "出境": "外部共享",
    "黑客": "攻击者",
    "渗透": "入侵",
    "漏洞": "风险点",
    "个人信息": "用户信息",
    "隐私": "合规信息",
    "敏感信息": "重要信息",
    "涉密": "机密",
}


def sanitize_topic_brief(tb: str) -> str:
    if not tb:
        return tb
    out = tb
    for k, v in _SANITIZE_MAP.items():
        out = out.replace(k, v)
    return out


def build_safe_retry_prompt(original_prompt: str) -> str:
    tail = """\n\n【安全重试约束】\n- 仅生成与学术/业务场景相关的中性内容，不涉及政治敏感、暴力、违法犯罪、仇恨、色情等。\n- 不要出现真实国家/政党/政府机关/执法机关/具体地名/具体人物。\n- 不要给出任何攻击、入侵、绕过、破解、加密实现细节等可被滥用的操作性步骤。\n- 若话题含高风险词，请使用更中性的替代说法表达同一含义。\n"""
    return (original_prompt.rstrip() + tail)

# =============================================================================
# 路径配置
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "dataset_glm16.json"
OUT_TXT  = OUT_DIR / "dataset_glm16.txt"

LOG_DIR = OUT_DIR / "logs_glm16"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = LOG_DIR / "raw16"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_MEM_PATH   = LOG_DIR / "topic_memory_glm16.json"
TOPIC_BLOCKLIST_PATH = LOG_DIR / "blocked_topics_glm16.txt"
GEN_LOG_PATH     = LOG_DIR / "genlog_glm16.jsonl"
MANIFEST_PATH    = LOG_DIR / "manifest_glm16.json"
GEN_COUNTER_PATH = LOG_DIR / "gen_counter_glm16.txt"

# =============================================================================
# 环境加载 (.env)
# =============================================================================
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(SCRIPT_DIR / ".env", override=True)
    p = find_dotenv()
    if p:
        load_dotenv(p, override=False)
except Exception:
    pass

ZHIPU_API_KEY = (os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY") or "").strip()
ZHIPU_BASE_URL = (os.getenv("ZHIPU_BASE_URL") or os.getenv("GLM_BASE_URL") or "https://api.z.ai/api/paas/v4").strip()  # 官方 OpenAI 兼容网关根路径

def normalize_base_url(url: str) -> str:
    """Accept either gateway root or full endpoint, and normalize to gateway root for OpenAI client."""
    u = (url or "").strip()
    if not u:
        return "https://api.z.ai/api/paas/v4"
    u = re.sub(r"/chat/completions/?$", "", u)  # 用户误填完整接口时自动纠正
    u = u.rstrip("/")
    return u

ZHIPU_BASE_URL = normalize_base_url(ZHIPU_BASE_URL)
ZHIPU_MODEL    = os.getenv("ZHIPU_MODEL", "glm-4.5-air")  # 默认使用 glm-4.5-air

if not ZHIPU_API_KEY:
    raise RuntimeError("未读取到 ZHIPU_API_KEY，请在 .env 写入 ZHIPU_API_KEY=...")

client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)

# =============================================================================
# 生成参数
# =============================================================================
TEMP_RANGE = (0.60, 0.98)
TOPP_RANGE = (0.90, 0.96)
MAX_TOKENS = 1800
RETRIES = int((os.getenv("ZHIPU_RETRIES") or os.getenv("GLM_RETRIES") or "5"))  # 单条样本/话题的最大重试次数

N_CANDS    = 2
MAX_TRIES  = 10
SLEEP      = 0.15

DOC_MIN_LEN   = 110
JACC_DOC_THR  = 0.32
HAMM_THR      = 8
TOPIC_MEM_MAX = 400

# 每一轮按 50 条做批次内类型配额控制
BATCH_SIZE = 50

# =============================================================================
# 领域 + 类型
# =============================================================================
DOMAINS = [
    "新闻与公共事务",   # PA
    "商业与金融",       # BF
    "科技与数字治理",   # TDG
    "教育与学术",       # EDU
    "社会民生与文体"    # SLSC
]
DOMAIN_ABBR = {
    "新闻与公共事务": "PA",
    "商业与金融": "BF",
    "科技与数字治理": "TDG",
    "教育与学术": "EDU",
    "社会民生与文体": "SLSC"
}

COREF_TYPES = ["人称代词", "指示代词", "有定描述", "零指代", "事件指代"]

# =========================
# TOPIC_BANK_V59（继续一批：五域×10）
# =========================
TOPIC_BANK: Dict[str, List[str]] = {
    "新闻与公共事务": [
        "城市防洪排涝泵站远程监控掉线应急复盘", "乡镇文化站消防验收缺漏整改",
        "行政审批“电子材料自动校验”误判率评测与修正", "城市道路临时占道施工夜间警示不足整治",
        "公共卫生机构废弃针具回收台账抽检", "市级应急广播试播与覆盖盲区补点",
        "农村污水处理站药剂投放不规范专项纠偏", "社区居民议事会代表产生程序争议复核",
        "政府热线“跨部门转办循环”卡点治理", "城市新建人行天桥电梯故障率与维保考核"
    ],
    "商业与金融": [
        "银行对个人消费贷“用途证明过度收集”合规整改回访", "平台商户“预售尾款时间窗过短”投诉规则优化",
        "企业跨境收款“同名不同户”导致冻结案例复盘", "保险理赔“线上复核拖延”时限考核与问责",
        "上市公司大额关联采购定价偏离线索穿透审计", "消费金融“以物抵贷”估值泄露与风控复盘",
        "供应链金融“平台撮合”中介费过高争议听证", "券商客户风险揭示书线上签署走过场治理",
        "地方融资租赁公司资产分类偏松风险回访", "私募基金投后月报“关键指标缺失”整改"
    ],
    "科技与数字治理": [
        "政务大模型对“模糊主体提问”的消歧与追问策略评测", "公共数据共享平台“秒级限流触发过频”误伤复核",
        "智慧交通停车诱导算法对临时封场的自适应复盘", "AI辅助审批对跨附件佐证链的关联抽取漏配治理",
        "城市统一政务搜索“热词联想误导办事”纠偏", "公共算法推荐“不同年龄层曝光差异”公平性审计",
        "政务系统批量导入历史材料导致字符集乱码修复", "生成式AI政务场景输出“表格数值错位”自动校验策略",
        "城市物联网终端远程重启频繁导致业务中断复盘", "政务云对象存储跨项目共享桶审计边界重划"
    ],
    "教育与学术": [
        "高校研究生课程助教津贴发放延迟专项治理", "本科课堂期末展示评分“导师偏差”校准评测",
        "实验室危化品退库流程缺失导致库存偏差治理", "校企联合培养学生实习考核“企业口径变更”衔接复盘",
        "学术论文数据复现失败的责任认定与通报机制评估", "高校在线教学平台“AI批改回退手工”比例异常诊断",
        "跨学院共享课程期中补考安排冲突协调", "研究生学位论文盲审“评价维度不一致”模板统一",
        "青年教师教学督导反馈迟滞问题治理", "毕业论文图表重绘未标注来源线索抽检"
    ],
    "社会民生与文体": [
        "社区公共卫生站疫苗预约系统“排队号抖动”治理复盘", "城市公交换乘优惠“跨月失效”投诉专项纠偏",
        "公共医院问诊录音存储期限与取证需求冲突听证", "社区电动车充电棚“高峰限时”规则公平性评测",
        "城市慢行道新装路桩导致儿童通行风险整改", "老旧小区消防水压不足与管网改造协商",
        "社区文化活动讲师临时更换未告知投诉治理", "城市公园夜间照明“跳闸黑区”巡检与修复考核",
        "文体场馆公益票“验证码频繁失效”系统修复复盘", "文旅景区自助售票机故障导致排队过长应急分流"
    ]
}

ACTIONS = [
    "试点上线",
    "专项排查",
    "阶段性复盘",
    "听证/座谈协商",
    "跨部门联动",
    "政策细则发布",
    "技术迭代与回滚",
    "第三方评估",
]
CONFLICTS = [
    "效率与公平取舍",
    "隐私合规与数据共享矛盾",
    "预算约束与服务覆盖冲突",
    "安全风控与业务增长拉扯",
    "公众体验与管理成本矛盾",
    "短期成效与长期治理目标冲突",
]
EVIDENCE = [
    "监测与日志数据",
    "问卷/访谈纪要",
    "专家论证意见",
    "审计或测评报告",
    "工单/投诉统计",
    "现场抽检记录",
]

FORBID_SEEDS = [
    "“不会用手机”的老年人固定桥段",
    "“利用率不足15%”的模板数字",
    "“一刀切”式套话反复出现",
]

OPENING_STYLE_BANK = [
    "首句不要用年份开头，直接从事件或矛盾起笔；时间信息可后置。",
    "首句从人物或机构动作起笔，不要先写具体年份；可以用“近期”“本季度”等相对时间。",
    "首句从问题或冲突起笔，避免“在20xx年……”这类开头。",
    "首句可用“近期/新一轮/本季度”等相对时间开场，但不得出现 2024 及以前具体年份。",
]

# =============================================================================
# 批次内主类型配额：每 50 条内部尽量均衡
# =============================================================================
def pick_primary_for_batch(batch_type_counts: Dict[str, int], batch_size: int) -> str:
    per_type_target = max(batch_size // len(COREF_TYPES), 1)
    type_targets = {t: per_type_target for t in COREF_TYPES}
    gaps = {
        t: max(type_targets[t] - int(batch_type_counts.get(t, 0)), 0)
        for t in COREF_TYPES
    }
    total_gap = sum(gaps.values())
    if total_gap <= 0:
        return random.choice(COREF_TYPES)

    r = random.uniform(0, total_gap)
    acc = 0.0
    for t in COREF_TYPES:
        acc += gaps[t]
        if r <= acc:
            return t
    return COREF_TYPES[-1]

# =============================================================================
# 话题规划 + 记忆
# =============================================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_gen_counter() -> int:
    if GEN_COUNTER_PATH.exists():
        try:
            return int(GEN_COUNTER_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            return 0
    return 0


def save_gen_counter(v: int):
    GEN_COUNTER_PATH.write_text(str(v), encoding="utf-8")


def append_genlog(entry: dict):
    with GEN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_topic_memory() -> List[str]:
    if TOPIC_MEM_PATH.exists():
        try:
            mem = json.loads(TOPIC_MEM_PATH.read_text(encoding="utf-8"))
            if isinstance(mem, list):
                return mem[-TOPIC_MEM_MAX:]
        except Exception:
            pass
    return []


def save_topic_memory(mem: List[str]):
    mem = mem[-TOPIC_MEM_MAX:]
    TOPIC_MEM_PATH.write_text(
        json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_topic_blocklist() -> set[str]:
    if not TOPIC_BLOCKLIST_PATH.exists():
        return set()
    try:
        obj = json.loads(TOPIC_BLOCKLIST_PATH.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            arr = obj.get("blocked", [])
        else:
            arr = obj
        return set(arr) if isinstance(arr, list) else set()
    except Exception:
        return set()


def save_topic_blocklist(blocked: set[str]) -> None:
    try:
        TOPIC_BLOCKLIST_PATH.write_text(
            json.dumps({"blocked": sorted(blocked)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass



def topic_signature(topic_brief: str) -> str:
    t = re.sub(r"\s+", "", topic_brief.lower())
    t = re.sub(r"[：:|｜，、。；;,.!！?？\-—]+", "_", t)
    parts = sorted(set(filter(None, t.split("_"))))
    return "|".join(parts)


def make_topic_brief(domain: str) -> str:
    s = random.choice(TOPIC_BANK[domain])
    a = random.choice(ACTIONS)
    c = random.choice(CONFLICTS)
    e = random.choice(EVIDENCE)
    return f"{domain}｜{s}｜{a}｜关注：{c}｜证据：{e}"


def plan_topics_quota(total_limit: int, history: List[str],
                      domain_targets: Dict[str, int]) -> List[str]:
    hist_set = set(history)
    topics: List[str] = []
    tries = 0
    domain_left = dict(domain_targets)

    def pick_domain_by_gap() -> str:
        g = {d: max(domain_left[d], 0) for d in DOMAINS}
        tot = sum(g.values())
        if tot <= 0:
            return random.choice(DOMAINS)
        r = random.uniform(0, tot)
        acc = 0.0
        for d, w in g.items():
            acc += w
            if acc >= r:
                return d
        return random.choice(DOMAINS)

    while len(topics) < total_limit and tries < total_limit * 80:
        d = pick_domain_by_gap()
        tb = make_topic_brief(d)
        sig = topic_signature(tb)
        if sig in hist_set:
            tries += 1
            continue
        if any(seed in tb for seed in FORBID_SEEDS):
            tries += 1
            continue
        topics.append(tb)
        hist_set.add(sig)
        domain_left[d] -= 1
    return topics

# =============================================================================
# Prompt（五类指代 + 零代词 Ø）
# =============================================================================
SYSTEM_PROMPT = (
    "你是中文写作者。严格只返回一个合法 JSON 对象，不要 Markdown 代码块、不要解释。"
)

WRITE_PROMPT = """
你现在要围绕“{DOMAIN}”领域的话题“{TOPIC}”，生成一条包含**中文指代消解**现象的短文，并给出结构化标注。

本条数据的主类型是：**{PRIMARY}**（五类之一：人称代词、指示代词、有定描述、零指代、事件指代）。

====================【整体要求】====================
1. 文本与句子
- 写一段自然的新闻 / 通告 / 工作报告风格的短文，语体为：{OPEN_STYLE}，不要写成提示词或说明书。
- 分成 5~7 句，放在 "sentences" 数组中，每句用中文句号“。”结尾。
- "text" 字段等于所有句子用单个空格拼接后的结果，内容必须与 "sentences" 完全一致（只是分句不同）。
- 不要出现真实世界的机构全称、项目名或具体地址，可以使用虚构但合理的中文人名、机构名、地名。
- 所有年份必须是 **2025 年及以后**（例如 2025 / 2026），不要出现 2024 及以前的具体年份。
- 为避免话题高度重复，请避免与以下摘要雷同：{NEG_LIST}（这些只是提示，不要照抄）。

2. 指代链总体约束
- 至少生成 3 条以上的 "coreference_chains"。
- 必须覆盖至少 3 种不同的 "type"；**必须保证**包含主类型 {PRIMARY}。
- 每一条链：
  - "type" 必须是下列之一："人称代词"、"指示代词"、"有定描述"、"零指代"、"事件指代"；
  - 至少包含 2 个 mention；
  - "index" 从 1 开始递增；
  - "start_sentence" 和 "end_sentence" 是该链中最早 / 最晚出现的句子编号（从 1 开始）。

3. 字段结构（最终 JSON 顶层）
你必须输出一个 JSON 对象，键名固定如下：
{
  "category_primary": "{PRIMARY}",
  "domain": "{DOMAIN}",
  "text": "完整文本",
  "sentences": ["句1", "句2", ...],
  "coreference_chains": [
    {
      "type": "人称代词/指示代词/有定描述/零指代/事件指代",
      "index": 1,
      "start_sentence": 1,
      "end_sentence": 3,
      "mentions": [
        {
          "sentence_id": 1,
          "text": "表面字符串（或 Ø）",
          "role": "antecedent"  // 或 "anaphor"
        }
      ]
    }
  ],
  "entities": {
    "persons": ["人名1", "人名2"],
    "orgs": ["机构名1", "机构名2"],
    "locations": ["地名1", "地名2"]
  }
}

- 顶层 JSON 中 **不要** 出现 "char_start" / "char_end" 字段（这些由后续脚本自动补全）。
- "mentions" 里暂时也只需要给出 "sentence_id"、"text"、"role" 三个字段，位置索引由后续脚本自动计算。
- 同一个 mention 不能同时属于两条不同的链。

====================【五类指代类型的精确定义】====================

1. 人称代词
- 形式：具有明确人称和数特征的代词，如“我/我们/你/你们/他/她/它/他们/她们/它们”等。
- 指向具体的人或机构负责人，而不是事件或抽象概念。
- 在 coreference_chains 中：
  - antecedent 通常是带名词的人名 / 职务描述，如“局长张某”“项目负责人李某”等；
  - anaphor 是这些人称代词，如“他”“她”“他们”。

2. 指示代词
- 形式：含有指示标记的名词短语或代词，如：
  - 单独的“这/那/该/此/这些/那些/上述”；
  - 或者带名词的组合：“该项目”“这一数值”“上述措施”“这部分资金”等。
- 指向已经在文本中唯一确定的“实体”（人、机构、项目、金额、数值等），而不是“事件进程本身”。
- 只要表面形式中出现了“这/那/该/此/上述”等指示成分，就归入“指示代词”，即便后面还有名词。

3. 有定描述
- 形式：不含指示词，但在当前语境下唯一可识别的名词短语，如：
  - “基金负责人”“审计团队”“项目组”“金融监管部门内部审计处”等。
- 依靠共有背景或篇章语境可以确定唯一实体，但表面形式中 **没有** “这/那/该/此/上述”等指示成分，也不是人称代词。
- 常见模式：先给出实体的全称或专有名（如“青岚自然资源局”），后文用“该局局长”“局长喻某”“该局内部审计处”等描述同一实体，这些后续描述属于“有定描述”。

4. 零指代（Ø）
- 句法上本应出现主语 / 宾语 / 受事等成分，但在表层句子中被省略，仍然回指前文某个实体。
- 在 "sentences" 中，用**单个字符** “Ø” 直接占位，表示该处出现了一个省略的论元，例如：
  - “Ø 将在月底前完成拨款审批。”
  - “预计 Ø 在第三季度完成验收。”
- 严禁使用 “Ø(将通过)” 等形式，必须只写 “Ø”。
- 在 "coreference_chains" 中：
  - 至少包含一个带明示名词的 antecedent mention（如“基金会”“审计团队”“项目组”等），用于说明 Ø 指向谁；
  - 同一条“零指代”链内可以出现多个 "Ø" mention，这些 Ø 都回指到同一个 antecedent；
  - 零指代链的 "type" 固定为 "零指代"。

5. 事件指代
- 指向的是“动作 / 过程 / 状态”这一类事件，而不是实体。
- 触发词通常由“动词 / 动词短语 + 事件性名词 + 可选指示成分”构成，例如：
  - “启动专项审计”“完成首轮排查”“开展风险排查行动”“这项调整”“该轮整改”“上述措施”“这一结果”“相关结果”等。
- 在事件链中：
  - antecedent 可以是第一次完整描述该事件的短语（例如“启动专项执法行动”“完成首轮数据采集”）；
  - anaphor 往往是“这项措施/上述行动/该安排/这一结果/相关结果”等带指示色彩的事件短语；
  - 这些 mention 的共同指向是“同一个事件”，而不是机构或个人。

====================【类型区分决策流程】====================

(1) 先判断它有没有字面形式？
- 没有：如果在句法上有成分缺省，并且这个省略成分可以自然回指前文某个实体，则：
  - 在句子里写成 “Ø”；
  - 在 "coreference_chains" 中创建一条 type = "零指代" 的链；
  - 链中至少放入一个带名词的 antecedent + 一个或多个 "Ø" anaphor。
- 如果无法自然回指前文实体，就不要强行标注成任何指代链。

- 有字面形式 → 进入 (2)。

(2) 再判断它指向“实体”还是“事件”，以及是否带指示词：
- 若指的是“事件”（动作 / 过程 / 措施 / 结果等），尤其是带“这次/该项/这项/上述/这一结果/相关结果”等：统一归入 **"事件指代"**。
- 若指的是“实体”（人、机构、项目、金额、数值等），则：
  - 表面含有“这/那/该/此/上述”等指示成分 → **"指示代词"**；
  - 不含这些指示词，但在当前语境下是唯一可识别实体 → **"有定描述"**；
  - 如果是典型人称代词（我/我们/你/你们/他/她/它/他们/她们/它们）→ **"人称代词"**。

通过上述三条轴：
- 是否显式出现（零指代 vs 非零）、
- 是否带指示成分（指示代词 vs 有定描述）、
- 指向实体还是事件（实体三类 vs 事件指代），
把五类指代在同一共指框架下清晰地区分开。

====================【消极要求】====================
- 不要出现名单、编号式罗列。
- 不要写成对“指代消解任务”的说明或教学文字，文本应像一篇真实的领域新闻/通告。
- 避免价值判断或敏感话题，聚焦业务流程、预算、进度、指标等。

现在，请严格按照上述所有规则，直接输出一个 JSON 对象，不要有任何多余文本。
""".strip()

VERIFY_PROMPT = """
你现在充当第二阶段的 **JSON 质检与修复模型**。

输入是一段文本形式的 JSON（由另一模型生成），它应该满足以下条件：
- 结构与字段必须符合前面定义的格式；
- 指代链 type 只能是 "人称代词"、"指示代词"、"有定描述"、"零指代"、"事件指代"；
- 零指代在 sentences 中必须写成单独的字符 "Ø"，不能写成 "Ø(将通过)" 之类；
- 每条零指代链中必须至少有一个带明示名词的 antecedent，以及一个或多个 text="Ø" 的 anaphor；
- 事件指代必须指向“动作 / 过程 / 措施 / 结果”等事件，而不是实体；
- 指示代词必须带有“这/那/该/此/上述”等成分；
- 有定描述不能带这些指示词，但在语境中是唯一可识别实体；
- 人称代词必须是“我/我们/你/你们/他/她/它/他们/她们/它们”等形式；
- "text" 与 "sentences" 内容一致（拼接 sentences 即得 text）；
- 不出现 2024 及以前的年份；
- JSON 能够被严格解析。

你的任务：
1. 尝试解析输入 JSON；
2. 检查是否违反上述任何规则；
3. 在可以“局部修改”就修好的情况下，优先直接修正（例如：
   - 把 "Ø(将通过)" 改为 "Ø"；
   - 把本应是事件指代的链 type 从 "指示代词" 改成 "事件指代" 等）；
4. 如果结构严重错误、文本与句子完全错乱，或者无法在不重写全文的前提下修复，则判定为 reject。

输出格式 **只能是下面三种情况之一**（务必是一个 JSON 对象）：

1）完全合法：
{
  "status": "ok",
  "data": 原始JSON对象
}

2）已修复：
{
  "status": "fixed",
  "data": 修正后的JSON对象,
  "errors": ["简要说明你修改了哪些地方"]
}

3）无法修复：
{
  "status": "reject",
  "errors": ["原因1", "原因2", ...]
}

注意：
- 不要加入任何额外字段；
- 不要输出多余说明文字或 Markdown，只能输出一个 JSON 对象。
""".strip()

# =============================================================================
# 调用 GLM
# =============================================================================
def _pack_user_first(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sys_txt = ""
    rest = []
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system":
            sys_txt += (m.get("content") or "") + "\n\n"
        else:
            rest.append(m)
    if rest and (rest[0].get("role") or "").lower() == "user":
        rest[0]["content"] = sys_txt + rest[0].get("content", "")
        return rest
    return [{"role": "user", "content": sys_txt + (rest[0].get("content", "") if rest else "")}]



def glm_chat(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 1800,
    retries: int = 3,
    backoff_s: float = 1.2,
) -> Any:
    """单次 chat 调用（OpenAI 兼容）+ 更强的 1210 参数错误自愈。

    你遇到的报错：
      400 / code=1210 / “API 调用参数有误”
    在 BigModel/智谱网关里非常常见，触发原因通常是：
    - 采样参数越界（temperature/top_p）
    - payload 携带了某些网关/模型不支持的字段
    - 输入过长（prompt + max_tokens 超过上下文上限）
    本函数会按“从完整到最小”的方式降级重试，并在 1210 时自动降低 max_tokens/合并 system。
    """
    # 参数范围（Z.AI 文档约束的常见取值）：temperature ∈ [0,1]；top_p ∈ [0.01,1]
    temp = _clip_float(temperature, 0.0, 1.0, default=0.95)
    topp = _clip_float(top_p, 0.01, 1.0, default=0.95)

    # max_tokens 做一个保守裁剪（不同模型上限不同；裁剪能减少“输入过长”被归类为 1210 的概率）
    try:
        mt = int(max_tokens)
    except Exception:
        mt = 1200
    mt = max(1, min(mt, 2048))

    # vision 模型兼容：仅把 user 的字符串 content 包装为 content parts
    msgs = _normalize_messages_for_model(messages, model)

    def _merge_system(msgs_in: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把 system 迁移到 user（某些网关不支持 system 角色时可用）。"""
        sys_txt = ""
        rest: List[Dict[str, Any]] = []
        for mm in msgs_in:
            if (mm.get("role") == "system"):
                c = mm.get("content")
                if isinstance(c, list):
                    # parts -> text 拼起来
                    sys_txt += "".join([str(x.get("text", "")) for x in c if isinstance(x, dict)])
                else:
                    sys_txt += str(c or "")
            else:
                rest.append(mm)
        if not rest:
            return [{"role": "user", "content": sys_txt}]
        # 找第一条 user
        for i, mm in enumerate(rest):
            if mm.get("role") == "user":
                c = mm.get("content")
                if isinstance(c, list):
                    # 在 parts 前追加一段 system text
                    rest[i] = dict(mm)
                    rest[i]["content"] = [{"type": "text", "text": sys_txt}] + c
                else:
                    rest[i] = dict(mm)
                    rest[i]["content"] = sys_txt + str(c or "")
                return rest
        # 没有 user，就新建
        return [{"role": "user", "content": sys_txt}] + rest

    def _truncate_user(msgs_in: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
        """只截断 user 文本（保留尾部），避免超过上下文。"""
        out = []
        for mm in msgs_in:
            if mm.get("role") != "user":
                out.append(mm)
                continue
            m2 = dict(mm)
            c = m2.get("content")
            if isinstance(c, list):
                # 找到第一段 text，截断其 text
                parts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = str(part.get("text", ""))
                        if len(t) > max_chars:
                            t = t[-max_chars:]
                        parts.append({"type": "text", "text": t})
                    else:
                        parts.append(part)
                m2["content"] = parts
            else:
                t = str(c or "")
                if len(t) > max_chars:
                    t = t[-max_chars:]
                m2["content"] = t
            out.append(m2)
        return out

    # OpenAI-compat 常用 payload（尽量只用标准字段，降低 1210 概率）
    payload_full = {
        "model": model,
        "messages": msgs,
        "temperature": temp,
        "top_p": topp,
        "max_tokens": mt,
    }

    # 逐级降级 payload（字段越少越兼容）
    fallbacks: List[Dict[str, Any]] = [
        payload_full,
        {"model": model, "messages": msgs, "temperature": temp, "max_tokens": min(mt, 1200)},
        {"model": model, "messages": msgs, "temperature": temp},
        {"model": model, "messages": msgs},
        # 合并 system -> user
        {"model": model, "messages": _merge_system(msgs), "temperature": temp, "max_tokens": min(mt, 1200)},
        {"model": model, "messages": _merge_system(msgs)},
        # 仍失败就尝试截断输入（保留尾部，指令通常在尾部更关键）
        {"model": model, "messages": _truncate_user(_merge_system(msgs), 6000), "temperature": 0.7, "max_tokens": 800},
    ]

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        # 每次重试都从“最完整”开始；若报 1210 就顺序尝试 fallbacks
        try:
            return client.chat.completions.create(**payload_full)
        except BadRequestError as e:
            last_exc = e
            if _is_param_error_1210(e):
                # 1210：顺序尝试更“瘦”的 payload
                for fb in fallbacks[1:]:
                    try:
                        return client.chat.completions.create(**fb)
                    except Exception as e2:
                        last_exc = e2
                # 仍失败：稍作退避后进入下一轮 attempt
                time.sleep(backoff_s * (attempt + 1))
                continue
            # 非 1210：直接抛出让上层处理（例如配额/鉴权/其它 4xx）
            raise
        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as e:
            last_exc = e
            time.sleep(backoff_s * (attempt + 1))
            continue

    assert last_exc is not None
    raise last_exc


def call_glm(prompt: str, temperature: float, top_p: float) -> str:
    """对齐 DeepSeekAPI：返回 assistant 的纯文本 content；并避免 4xx 直接导致整个脚本崩溃。"""
    try:
        resp = glm_chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=ZHIPU_MODEL,
            temperature=temperature,
            top_p=top_p,
            max_tokens=MAX_TOKENS,
            retries=RETRIES,
        )
    except ContentFilterError:
        raise
    except BadRequestError as e:
        # 1210/其它参数错误：返回空字符串，让上层按“json_parse_failed”走重试/跳过逻辑
        # 同时打印最小诊断信息（不包含密钥）。
        msg = str(e)
        print(f"[WARN][GLM] BadRequestError: {msg[:180]} ...")
        return ""
    except Exception as e:
        print(f"[WARN][GLM] API error: {type(e).__name__}: {str(e)[:180]} ...")
        return ""

    # OpenAI SDK 返回对象（含 choices[0].message.content）
    try:
        content = resp.choices[0].message.content  # type: ignore[attr-defined]
        return (content or "").strip()
    except Exception:
        # 兜底：兼容 dict 结构或异常返回
        try:
            content = resp["choices"][0]["message"]["content"]  # type: ignore[index]
            return (content or "").strip()
        except Exception:
            return str(resp).strip()


# =============================================================================
# JSON 解析 + 去重哈希
# =============================================================================
def strip_code_fences(s: str) -> str:
    m = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    return "\n".join(m) if m else s


def sanitize_json_like(s: str) -> str:
    s = strip_code_fences(s)
    s = (s.replace("“", "\"")
           .replace("”", "\"")
           .replace("‘", "\"")
           .replace("’", "\"")
           .replace("：", ":"))
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r",(\s*[\]}])", r"\1", s)
    return s.strip()


def try_load_obj(text: str) -> Optional[dict]:
    s = (text or "").strip()
    if not s:
        return None
    # 尝试找最外层 {...}
    i = s.find("{")
    if i != -1:
        depth = 0
        j = -1
        for k, ch in enumerate(s[i:], i):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    j = k
                    break
        if j != -1:
            cand = s[i:j + 1]
            for v in [cand, sanitize_json_like(cand)]:
                try:
                    obj = json.loads(v)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
    # 整体尝试
    try:
        obj = json.loads(sanitize_json_like(s))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def get_ngrams(s: str, n: int) -> set:
    s = re.sub(r"\s+", "", s)
    return set(s[i:i + n] for i in range(max(len(s) - n + 1, 0)))


def jaccard7(a: str, b: str) -> float:
    A = get_ngrams(a, 7)
    B = get_ngrams(b, 7)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def stable_hash64(x: str) -> int:
    h = hashlib.md5(x.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def simhash_3gram(s: str) -> int:
    s = re.sub(r"\s+", "", s)
    feats = [s[i:i + 3] for i in range(max(len(s) - 2, 0))]
    bits = [0] * 64
    for f in feats:
        h = stable_hash64(f)
        for b in range(64):
            bits[b] += 1 if ((h >> b) & 1) else -1
    v = 0
    for b in range(64):
        if bits[b] > 0:
            v |= (1 << b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

# =============================================================================
# 对齐 char_start / char_end
# =============================================================================
DEICTIC_PREFIX = (
    "这项", "这种", "这位", "这一", "这个", "这些", "这",
    "此", "该", "该项", "该举措", "该措施", "该工程",
    "该决定", "该工作", "该项目", "上述", "上述措施", "上述行动",
)


def _normalize_head(m: str) -> str:
    s = m.strip()
    for p in DEICTIC_PREFIX:
        if s.startswith(p):
            s = s[len(p):]
            break
    return s


def find_span(sentence: str, mention: str, used_spans: List[range]) -> Optional[tuple]:
    def first_non_overlapping(hay: str, needle: str) -> Optional[tuple]:
        start = 0
        while True:
            idx = hay.find(needle, start)
            if idx == -1:
                return None
            span = range(idx, idx + len(needle))
            if all(span.stop <= u.start or span.start >= u.stop for u in used_spans):
                return (idx, idx + len(needle))
            start = idx + 1

    r = first_non_overlapping(sentence, mention)
    if r:
        return r
    norm = _normalize_head(mention)
    if norm and norm != mention:
        r = first_non_overlapping(sentence, norm)
        if r:
            return r
    return None


def _normalize_mentions_list(m_list: list) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for m in m_list:
        if not isinstance(m, dict) or "sentence_id" not in m:
            return None
        sid = int(m["sentence_id"])
        role, text = None, None
        if "role" in m and "text" in m:
            role = str(m["role"]).strip()
            text = str(m["text"]).strip()
        elif "antecedent" in m:
            role = "antecedent"
            text = str(m["antecedent"]).strip()
        elif "anaphor" in m:
            role = "anaphor"
            text = str(m["anaphor"]).strip()
        if role not in ("antecedent", "anaphor") or not text:
            return None
        out.append({"sentence_id": sid, "role": role, "text": text})
    return out




# -----------------------------
# Z.AI / BigModel compatibility helpers
# - temperature range: [0.0, 1.0]
# - top_p range: [0.01, 1.0]
# - vision models (e.g., glm-4.6v) expect OpenAI-style "content parts"
# -----------------------------
_VISION_MODEL_RE = re.compile(r"^glm-[0-9.]*v(\b|[-_].*)", re.I)

def _is_vision_model(model: str) -> bool:
    return bool(_VISION_MODEL_RE.match((model or "").strip()))

def _normalize_messages_for_model(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """兼容 vision 模型（如 glm-4.6v）时的 messages 结构。

    经验上：多数 OpenAI-compat 网关仍允许 system.content 为字符串，
    仅要求 user.content 支持 content parts（list[{"type":"text","text":...}]）。
    为了最大兼容性：只把 user 的字符串 content 包装为 parts，其它角色保持原样。
    """
    if not _is_vision_model(model):
        return messages
    out: List[Dict[str, Any]] = []
    for m in messages:
        mm = dict(m)
        role = (mm.get("role") or "").strip()
        c = mm.get("content")
        if role == "user" and isinstance(c, str):
            mm["content"] = [{"type": "text", "text": c}]
        out.append(mm)
    return out

def _clip_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v
def add_char_spans(sample: dict) -> Optional[dict]:
    try:
        sents = sample["sentences"]
        chains = sample["coreference_chains"]
        assert isinstance(sents, list) and 5 <= len(sents) <= 7
        assert isinstance(chains, list) and len(chains) >= 3
    except Exception:
        return None

    per_sent_used: Dict[int, List[range]] = {i + 1: [] for i in range(len(sents))}
    new_chains = []

    for ci, ch in enumerate(chains, start=1):
        m_list = ch.get("mentions", [])
        if not isinstance(m_list, list) or len(m_list) < 2:
            return None
        norm_ms = _normalize_mentions_list(m_list)
        if not norm_ms:
            return None

        sent_ids = [int(m["sentence_id"]) for m in norm_ms]
        start_s = int(min(sent_ids))
        end_s = int(max(sent_ids))

        mentions_new = []
        for m in norm_ms:
            sid = int(m["sentence_id"])
            if not (1 <= sid <= len(sents)):
                return None
            sent = sents[sid - 1]
            txt = m["text"]
            role = m["role"]
            used = per_sent_used[sid]
            span = find_span(sent, txt, used)
            if span is None:
                return None
            st, ed = span
            used.append(range(st, ed))
            mentions_new.append({
                "sentence_id": sid,
                "role": role,
                "text": txt,
                "char_start": st,
                "char_end": ed,
            })

        ch2 = dict(ch)
        ch2["mentions"] = mentions_new
        ch2["start_sentence"] = int(ch.get("start_sentence", start_s))
        ch2["end_sentence"] = int(ch.get("end_sentence", end_s))
        ch2["index"] = int(ch.get("index", ci))
        new_chains.append(ch2)

    out = dict(sample)
    out["coreference_chains"] = new_chains
    return out

# =============================================================================
# 轻量打分 + 验证/修复
# =============================================================================
def score_sample(obj2: dict) -> int:
    chains = obj2.get("coreference_chains", [])
    types = set()
    cross = 0
    for ch in chains:
        t = ch.get("type")
        if t:
            types.add(t)
        ss = int(ch.get("start_sentence", 0))
        es = int(ch.get("end_sentence", 0))
        if es > ss:
            cross += 1
    return len(types) * 3 + cross


YEAR_START_RE = re.compile(r"^\s*20(25|26|27|28|29)\s*年")
YEAR_PRE_2025_RE = re.compile(r"20(0[0-9]|1[0-9]|2[0-4])\s*年")


def validate_basic(obj: dict) -> Tuple[bool, str]:
    text = obj.get("text") or ""
    sents = obj.get("sentences")
    chains = obj.get("coreference_chains")

    if not isinstance(text, str) or len(text) < DOC_MIN_LEN:
        return False, "text_too_short"
    if not isinstance(sents, list) or not (5 <= len(sents) <= 7):
        return False, "bad_sentences_len"
    if not isinstance(chains, list) or len(chains) < 3:
        return False, "too_few_chains"

    # 零指代结构约束
    for ch in chains:
        if isinstance(ch, dict) and ch.get("type") == "零指代":
            ms = ch.get("mentions", [])
            if not isinstance(ms, list) or len(ms) < 2:
                return False, "zero_chain_too_short"
            ante_cnt = 0
            ana_cnt = 0
            for m in ms:
                if not isinstance(m, dict):
                    return False, "zero_chain_bad_mention"
                role = m.get("role")
                txt = str(m.get("text", "")).strip()
                try:
                    sid = int(m.get("sentence_id"))
                except Exception:
                    return False, "zero_chain_bad_sid"
                if not (1 <= sid <= len(sents)):
                    return False, "zero_chain_bad_sid"
                sent = str(sents[sid - 1])
                if role == "antecedent":
                    ante_cnt += 1
                elif role == "anaphor":
                    ana_cnt += 1
                    if txt != "Ø":
                        return False, "zero_anaphor_not_O"
                    if "Ø" not in sent:
                        return False, "zero_anaphor_not_in_sentence"
                else:
                    return False, "zero_chain_bad_role"
            if ante_cnt != 1 or ana_cnt < 1:
                return False, "zero_chain_bad_cardinality"

    # 主类型必须真实出现在链的 type 中
    primary = obj.get("category_primary")
    if isinstance(primary, str):
        chain_types = set()
        for ch in chains:
            if isinstance(ch, dict):
                t = ch.get("type")
                if t:
                    chain_types.add(t)
        if primary in COREF_TYPES and primary not in chain_types:
            return False, f"primary_type_missing:{primary}"

    # 时间约束
    first_sent = sents[0] if sents else ""
    if isinstance(first_sent, str) and YEAR_START_RE.search(first_sent):
        return False, "bad_time_anchor_year_start"
    if YEAR_PRE_2025_RE.search(text):
        return False, "contains_pre_2025_year"

    return True, "ok"


def try_repair(obj: dict, tb: str, domain: str, primary: str, reason: str) -> Optional[dict]:
    bad_json = json.dumps(obj, ensure_ascii=False)
    prompt = VERIFY_PROMPT + "\n\n下面是需要检查和修复的 JSON 文本：\n" + bad_json
    raw_fix = call_glm(prompt, temperature=0.2, top_p=0.95)
    fixed = try_load_obj(raw_fix)
    if not isinstance(fixed, dict):
        return None
    fixed.setdefault("category_primary", primary)
    fixed.setdefault("domain", domain)
    fixed.setdefault("entities", {"persons": [], "orgs": [], "locations": []})
    ok, _ = validate_basic(fixed)
    if not ok:
        return None
    return add_char_spans(fixed)

# =============================================================================
# 读写工具
# =============================================================================
def ensure_out():
    if not OUT_JSON.exists():
        OUT_JSON.write_text("[]", encoding="utf-8")
    if not OUT_TXT.exists():
        OUT_TXT.write_text("", encoding="utf-8")


def read_json_list(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_record(rec: dict):
    data = read_json_list(OUT_JSON)
    data.append(rec)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_TXT.open("a", encoding="utf-8") as f:
        f.write(rec.get("text", "").replace("\n", " ").strip() + "\n")


def count_hist_by_domain(hist: list) -> Dict[str, int]:
    c = {d: 0 for d in DOMAINS}
    for x in hist:
        if isinstance(x, dict):
            d = x.get("domain")
            if d in c:
                c[d] += 1
    return c

# =============================================================================
# 主流程
# =============================================================================
def main():
    global ZHIPU_MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="本轮总生成条数（仅当 per_domain=None 时生效）")
    ap.add_argument("--per_domain", type=int, default=10,
                    help="每个领域生成多少条；默认 10（总数=10*5=50）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖写输出文件和日志")
    ap.add_argument("--seed", type=int, default=None, help="主随机种子（留档）")
    ap.add_argument("--model", type=str, default=ZHIPU_MODEL, help="GLM 模型名")
    ap.add_argument("--model_version", type=str, default="unknown",
                    help="可选：手动记录模型版本/快照")
    args = ap.parse_args()

    ZHIPU_MODEL = args.model

    master_seed = args.seed if args.seed is not None else (int(time.time() * 1e6) & 0xFFFFFFFF)
    random.seed(master_seed)

    # 覆盖模式：清空数据 + 日志
    if args.overwrite:
        if OUT_JSON.exists():
            OUT_JSON.write_text("[]", encoding="utf-8")
        if OUT_TXT.exists():
            OUT_TXT.write_text("", encoding="utf-8")
        if LOG_DIR.exists():
            shutil.rmtree(LOG_DIR)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for p in [GEN_LOG_PATH, MANIFEST_PATH, TOPIC_MEM_PATH, GEN_COUNTER_PATH]:
            if p.exists():
                p.unlink()
        save_topic_memory([])
        save_gen_counter(0)
    else:
        ensure_out()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    hist = read_json_list(OUT_JSON)
    hist_txt = [x.get("text", "") for x in hist if isinstance(x, dict)]
    hist_sim = [simhash_3gram(t) for t in hist_txt]

    # 配额目标
    if args.per_domain is not None:
        domain_targets = {d: args.per_domain for d in DOMAINS}
        total_limit = args.per_domain * len(DOMAINS)
    else:
        total_limit = args.limit or 20
        avg = (len(hist) + total_limit) // len(DOMAINS)
        base = count_hist_by_domain(hist)
        domain_targets = {d: max(avg - base[d], 0) for d in DOMAINS}
        while sum(domain_targets.values()) < total_limit:
            for d in DOMAINS:
                if sum(domain_targets.values()) >= total_limit:
                    break
                domain_targets[d] += 1

    topic_mem = load_topic_memory()
    blocked_topics = load_topic_blocklist()
    topics = plan_topics_quota(total_limit, topic_mem, domain_targets)
    if len(topics) < total_limit:
        print(f"[WARN] 可用离散话题不足 {total_limit}，实际仅 {len(topics)}。")

    gen_counter = load_gen_counter()
    accepted_ids: List[str] = []
    ok = 0
    used_this_round: List[str] = []

    # 当前批次（以 BATCH_SIZE 条样本为一轮）内，各主类型已生成数量
    batch_type_counts: Dict[str, int] = {t: 0 for t in COREF_TYPES}

    for tb in topics:
        if topic_signature(tb) in blocked_topics:
            print(f"[SKIP] 命中内容审核黑名单：{tb}")
            continue
        # 每当当前已接受样本数达到 BATCH_SIZE 的整数倍，重置批次计数
        if ok > 0 and ok % BATCH_SIZE == 0:
            batch_type_counts = {t: 0 for t in COREF_TYPES}

        domain = tb.split("｜", 1)[0] if "｜" in tb else random.choice(DOMAINS)
        primary = pick_primary_for_batch(batch_type_counts, BATCH_SIZE)

        neg_list = used_this_round[-6:] + topic_mem[-12:] + FORBID_SEEDS
        open_style = random.choice(OPENING_STYLE_BANK)

        # 安全的 replace，不再使用 .format，避免花括号冲突
        prompt = WRITE_PROMPT
        prompt = (prompt.replace("{PRIMARY}", primary)
                         .replace("{DOMAIN}", domain)
                         .replace("{TOPIC}", tb)
                         .replace("{NEG_LIST}", "；".join(neg_list))
                         .replace("{OPEN_STYLE}", open_style))

        prompt_hash = "sha256:" + sha256_hex(prompt)

        best: Optional[dict] = None
        best_score = -1
        best_gen_id = None

        for attempt in range(MAX_TRIES):
            T = random.uniform(*TEMP_RANGE)
            P = random.uniform(*TOPP_RANGE)
            cand_seed_base = random.randint(0, 2**31 - 1)

            for cand_i in range(1, N_CANDS + 1):
                gen_counter += 1
                abbr = DOMAIN_ABBR.get(domain, "UNK")
                gen_id = f"{abbr}-{gen_counter:06d}-GLM-cand{cand_i}"

                try:
                    raw = call_glm(prompt, T, P)
                except ContentFilterError:
                    safe_prompt = build_safe_retry_prompt(prompt)
                    try:
                        raw = call_glm(safe_prompt, min(T, 0.6), min(P, 0.9))
                    except ContentFilterError:
                        sig = topic_signature(tb)
                        blocked_topics.add(sig)
                        save_topic_blocklist(blocked_topics)
                        print(f"[SKIP][FILTER] 话题触发内容审核(1301)，已加入黑名单：{tb}")
                        continue
                except BadRequestError as e:
                    # 1210 等参数错误：本候选作废，进入下一个 cand / attempt
                    print(f"[WARN][GLM] 400 BadRequest: {str(e)[:160]} ...")
                    time.sleep(SLEEP)
                    continue
                except APIError as e:
                    print(f"[WARN][GLM] APIError: {str(e)[:160]} ...")
                    time.sleep(SLEEP)
                    continue
                except Exception as e:
                    print(f"[WARN][GLM] Exception: {type(e).__name__}: {str(e)[:160]} ...")
                    time.sleep(SLEEP)
                    continue

                RAW_DIR.joinpath(f"{gen_id}.txt").write_text(raw, encoding="utf-8")

                reasons: List[str] = []
                obj2: Optional[dict] = None

                obj = try_load_obj(raw)
                if not isinstance(obj, dict):
                    reasons.append("json_parse_failed")
                else:
                    obj.setdefault("category_primary", primary)
                    obj.setdefault("domain", domain)
                    obj.setdefault("entities", {"persons": [], "orgs": [], "locations": []})

                    ok_basic, reason = validate_basic(obj)
                    if not ok_basic:
                        reasons.append(reason)
                        obj2 = try_repair(obj, tb, domain, primary, reason)
                        if obj2 is None:
                            reasons.append("repair_failed")
                    else:
                        obj2 = add_char_spans(obj)
                        if obj2 is None:
                            reasons.append("span_alignment_failed")
                            obj2 = try_repair(obj, tb, domain, primary, "span_alignment_failed")
                            if obj2 is None:
                                reasons.append("repair_failed")

                passed_filters = False
                if obj2 is not None:
                    text2 = obj2["text"]
                    if any(jaccard7(text2, ht) >= JACC_DOC_THR for ht in hist_txt):
                        reasons.append("jaccard_overlap")
                    else:
                        sh = simhash_3gram(text2)
                        if any(hamming(sh, hs) <= HAMM_THR for hs in hist_sim):
                            reasons.append("simhash_overlap")
                        else:
                            passed_filters = True

                append_genlog({
                    "gen_id": gen_id,
                    "timestamp_utc": utc_now_iso(),
                    "backend": "zhipu",
                    "model": ZHIPU_MODEL,
                    "model_version": args.model_version,
                    "master_seed": master_seed,
                    "seed": cand_seed_base,
                    "topic": tb,
                    "domain": domain,
                    "category_primary": primary,
                    "sampling": {
                        "temperature": T,
                        "top_p": P,
                        "max_tokens": MAX_TOKENS,
                        "n_cands": N_CANDS,
                        "attempt": attempt + 1,
                    },
                    "prompt_hash": prompt_hash,
                    "prompt": prompt,
                    "raw_response_text": raw,
                    "validator": {"pass": passed_filters, "reasons": reasons},
                })

                if not passed_filters:
                    continue

                sc = score_sample(obj2)
                if sc > best_score:
                    best, best_score, best_gen_id = obj2, sc, gen_id

                if best_score >= 12:
                    break

            if best is not None:
                break
            time.sleep(SLEEP)

        if not best:
            print(f"[SKIP] 话题未成功生成：{tb}")
            continue

        best["_gen_id"] = best_gen_id
        append_record(best)

        primary_final = best.get("category_primary")
        if isinstance(primary_final, str) and primary_final in COREF_TYPES:
            batch_type_counts[primary_final] = batch_type_counts.get(primary_final, 0) + 1

        hist_txt.append(best["text"])
        hist_sim.append(simhash_3gram(best["text"]))
        used_this_round.append(tb)
        topic_mem.append(topic_signature(tb))
        topic_mem = topic_mem[-TOPIC_MEM_MAX:]

        accepted_ids.append(best_gen_id)
        ok += 1
        print(f"[OK] {ok}/{len(topics)} :: {tb} —— domain={domain}, "
              f"primary={primary_final}, score={best_score}, gen_id={best_gen_id}")

    save_topic_memory(topic_mem)
    save_gen_counter(gen_counter)

    manifest = {
        "accepted_gen_ids": accepted_ids,
        "build_time_utc": utc_now_iso(),
        "script": str(Path(__file__).name),
        "backend": "zhipu",
        "model": ZHIPU_MODEL,
        "model_version": args.model_version,
        "master_seed": master_seed,
        "args": vars(args),
        "filters": {
            "doc_min_len": DOC_MIN_LEN,
            "jaccard7_thr": JACC_DOC_THR,
            "simhash_hamming_thr": HAMM_THR,
            "min_chains": 3,
            "zero_pronoun_strict": True,
            "primary_must_in_chain_types": True,
            "no_year_start_required": True,
            "forbid_pre_2025_year_in_text": True,
        },
        "outputs": {
            "dataset_json": str(OUT_JSON),
            "dataset_txt": str(OUT_TXT),
            "logs_dir": str(LOG_DIR),
            "genlog_jsonl": str(GEN_LOG_PATH),
            "manifest_json": str(MANIFEST_PATH),
            "raw_dir": str(RAW_DIR),
            "topic_memory": str(TOPIC_MEM_PATH),
            "gen_counter": str(GEN_COUNTER_PATH),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] 写入 {ok} 条 → {OUT_JSON}")
    print(f"TXT → {OUT_TXT}")
    print(f"LOGS → {LOG_DIR}")


if __name__ == "__main__":
    main()