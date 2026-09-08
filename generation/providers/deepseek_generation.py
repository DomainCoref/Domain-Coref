# -*- coding: utf-8 -*-
"""
DeepSeekAPI.py — 生成“中文指代消解”数据集（五类：人称代词、指示代词、有定描述、零指代、事件指代）

目录结构：
out/
  dataset_coref_slim05.json   # 最终数据（只放这里）
  dataset_coref_slim05.txt    # 最终纯文本（只放这里）
  logs05/                     # 其他一切日志/溯源文件都放这里
    genlog_coref_slim05.jsonl
    manifest_coref_slim05.json
    gen_counter05.txt
    topic_memory05.json
    raw05/                    # 每个候选的原始响应
       PA-000001-DeepSeek-cand1.txt
       ...

污染防护落地：
1) 时序隔离：不要求正文必须出现 2025+，但若出现年份只能是 2025 及以后；首句不得年份起首
2) provenance 日志：genlog + manifest（用 timestamp_utc 证明生成时间）
3) master seed + cand seed 留档
4) 若链 type=事件指代，必须给 trigger（触发词），脚本自动补 char spans
"""

import os
import re
import json
import time
import random
import argparse
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# =============== 基础路径（out 下 JSON/TXT；logs05 下其余文件） ===============
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT_DIR / "dataset_coref_slim15.json"
OUT_TXT = OUT_DIR / "dataset_coref_slim15.txt"

LOG_DIR = OUT_DIR / "logs15"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = LOG_DIR / "raw15"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_MEM_PATH = LOG_DIR / "topic_memory15.json"
GEN_LOG_PATH = LOG_DIR / "genlog_coref_slim15.jsonl"
MANIFEST_PATH = LOG_DIR / "manifest_coref_slim15.json"
GEN_COUNTER_PATH = LOG_DIR / "gen_counter15.txt"

# =============== 环境 & 客户端 ===============
try:
    from dotenv import load_dotenv

    load_dotenv(SCRIPT_DIR / ".env")
except Exception:
    pass

from openai import OpenAI

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("缺少 DEEPSEEK_API_KEY，请在本目录 .env 配置 DEEPSEEK_API_KEY。")

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# =============== 生成配置 ===============
TEMP_RANGE = (0.90, 1.10)
TOPP_RANGE = (0.85, 0.97)
MAX_TOKENS = 1400
N_CANDS = 2
MAX_TRIES = 18
SLEEP = 0.12

DOC_MIN_LEN = 110
JACC_DOC_THR = 0.32
HAMM_THR = 8
TOPIC_MEM_MAX = 400

# 每一轮（批次）内部目标样本数，用于在每 50 条中平衡五种主类型
BATCH_SIZE = 50


def pick_primary_for_batch(batch_type_counts: Dict[str, int], batch_size: int) -> str:
    """
    在当前批次内，根据已有计数为缺口大的类型分配更高优先级。

    batch_type_counts: 当前批次已接受样本中，各主类型的数量。
    batch_size:        本批次目标样本总数（默认 50）。
    """
    # 为了在一批中尽量均衡，设定每个类型的目标数量
    # 例：batch_size=50, 类型数=5 → 每类目标 10 条
    per_type_target = max(batch_size // 5, 1)
    type_targets = {t: per_type_target for t in COREf_TYPES}

    gaps = {
        t: max(type_targets[t] - int(batch_type_counts.get(t, 0)), 0)
        for t in COREf_TYPES
    }
    total_gap = sum(gaps.values())
    if total_gap <= 0:
        # 所有类型都已在本批次达到或超过目标，退回均匀随机
        return random.choice(COREf_TYPES)

    r = random.uniform(0, total_gap)
    acc = 0.0
    for t in COREf_TYPES:
        acc += gaps[t]
        if r <= acc:
            return t
    return COREf_TYPES[-1]


# =============== 5 个领域 ===============
DOMAINS = [
    "新闻与公共事务",  # PA
    "商业与金融",  # BF
    "科技与数字治理",  # TDG
    "教育与学术",  # EDU
    "社会民生与文体",  # SLSC
]
DOMAIN_ABBR = {
    "新闻与公共事务": "PA",
    "商业与金融": "BF",
    "科技与数字治理": "TDG",
    "教育与学术": "EDU",
    "社会民生与文体": "SLSC",
}

COREf_TYPES = ["人称代词", "指示代词", "有定描述", "零指代", "事件指代"]

# =============== 话题库（略） ===============
# =========================
# TOPIC_BANK_V49（继续一批：五域×10）
# =========================
TOPIC_BANK: Dict[str, List[str]] = {
    "新闻与公共事务": [
        "城市道路施工信息“一张图”更新滞后专项纠偏", "乡村义务教育营养餐供货质量抽检",
        "行政审批“并联审查”环节串联化回退复盘", "城市公共护栏破损缺口巡查与补装验收",
        "公共卫生突发事件舆情回应口径不一致复核", "市级应急预警短信漏发原因追溯与演练",
        "农村饮水工程冬季冻管应急预案与物资储备核验", "社区便民服务站撤点并点的服务可达性评估",
        "政府采购中标后随意涨价变更线索核查", "城市渡口/桥区水上安全巡查联动复盘"
    ],
    "商业与金融": [
        "银行对公授信“名单制优惠”执行差异整改", "平台商户“预售延期发货”赔付规则争议复核",
        "企业跨境关联交易定价合理性穿透审计", "保险理赔“自动拒赔”原因说明可解释性整改",
        "上市公司年度报告审计意见异常的整改跟踪", "消费金融“分期提前结清违约金”合规清理",
        "供应链金融电子仓单重复质押风险联查", "券商新客开户反欺诈黑名单误伤申诉闭环",
        "地方产业纾困基金投向偏离整改复盘", "私募基金托管系统估值口径与管理人不一致对账"
    ],
    "科技与数字治理": [
        "政务大模型在跨部门协同办件中的角色边界评估", "公共数据共享“用途超范围”追溯与处罚机制建设",
        "智慧交通公交客流识别对遮挡场景鲁棒性修复", "AI辅助审批对“反事实材料”误判案例复盘",
        "城市统一政务搜索对老旧文档召回不足治理", "公共算法推荐“停留时长诱导”问题监测与调参",
        "政务系统消息中台“已办仍提醒”重复推送治理", "生成式AI政务场景输出“引用段落断句错误”纠偏",
        "城市物联网终端远程运维权限滥用风险红队测试", "政务云跨可用区切换后会话丢失专项复盘"
    ],
    "教育与学术": [
        "高校研究生入学复试线上系统崩溃应急预案复盘", "本科课堂分组展示评分偏差校准机制评估",
        "实验室安全隐患挂牌督办超期整改追踪", "校企联合培养学生考勤数据与企业系统不一致对账",
        "学术论文预印本与正式发表版本差异备案规范", "高校在线教学平台作业查重阈值过严误伤复核",
        "跨学院联合导师制责任分摊与绩效考核复盘", "研究生学位论文答辩委员利益冲突回避抽查",
        "青年教师科研项目预算调整审批时效评估", "毕业论文数据/代码开放后的复现投诉处理机制"
    ],
    "社会民生与文体": [
        "社区便民停车“错时共享”后的车位占用纠纷治理", "城市公交线路首末班时间公示不一致专项清理",
        "公共医院检验预约“爽约占号”惩戒升级评估", "社区电动车充电棚接地不良隐患排查整改",
        "城市慢行道冬季结冰防滑撒布与巡查联动", "老旧小区楼顶积水排放改造工程验收复盘",
        "社区文化活动老年报名人工窗口回归试点评估", "城市公园夜间灯光秀扰民投诉与时间窗协商",
        "文体场馆赛事期间停车涨价投诉取证与处置", "文旅景区亲子设施安全隐患排查与整改跟踪"
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
    "‘不会用手机’的老年人固定桥段",
    "‘利用率不足15%’的模板数字",
    "‘一刀切’式套话反复出现",
]

OPENING_STYLE_BANK = [
    "首句不要用年份开头，直接从事件/场景起笔；若需要时间可用相对时间。",
    "首句从人物/机构动作起笔，不要先写时间；时间可以后置或不写。",
    "首句从矛盾/问题起笔，时间信息放后面或隐含即可。",
    "首句可用“近期/新一轮/本季度”等相对时间开场，但不要出现具体年份。",
]

# =============== provenance 工具 ===============


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


# =============== 话题规划 ===============
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


def topic_signature(topic_brief: str) -> str:
    t = re.sub(r"\s+", "", topic_brief.lower())
    t = re.sub(r"[：|｜，、。；;,.!！?？\\-—]+", "_", t)
    parts = sorted(set(filter(None, t.split("_"))))
    return "|".join(parts)


def make_topic_brief(domain: str) -> str:
    s = random.choice(TOPIC_BANK[domain])
    a = random.choice(ACTIONS)
    c = random.choice(CONFLICTS)
    e = random.choice(EVIDENCE)
    return f"{domain}｜{s}｜{a}｜关注：{c}｜证据：{e}"


def plan_topics_quota(
    total_limit: int, history: List[str], domain_targets: Dict[str, int]
) -> List[str]:
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

    while len(topics) < total_limit and tries < total_limit * 60:
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


# =============== Prompt（含五类概念 + 判定流程 + 零指代规范） ===============
SYSTEM_PROMPT = (
    "你是中文写作者。严格只返回一个合法 JSON 对象，不要 Markdown 代码块、不要解释。"
)

WRITE_PROMPT = """基于【话题简述】，写 6–9 句**同一主题**的段落；并**只输出一个 JSON 对象**（键用英文双引号），结构必须为：
  {{
    "category_primary": "{PRIMARY}",
    "domain": "{DOMAIN}",
    "text": "<整段文本>",
    "sentences": ["句1","句2","…"],
    "coreference_chains": [
      {{
        "type": "指示代词|人称代词|有定描述|零指代|事件指代 之一",
        "index": 1,
        "start_sentence": 1,
        "end_sentence": 3,
        "mentions": [
          {{ "sentence_id": 1, "role": "antecedent", "text": "<先行词，需逐字出现在该句>" }},
          {{ "sentence_id": 2, "role": "anaphor",    "text": "<代词/指称词/Ø，需逐字出现在该句>" }}
        ],
        "trigger": {{ "sentence_id": 2, "text": "<事件触发词，若 type=事件指代 必填；需逐字出现在该句>" }}
      }}
    ],
    "entities": {{ "persons": [], "orgs": [], "locations": [] }}
  }}

  【五类指代类型说明（非常重要，请严格区分）】
  - 人称代词：
    * 用“我、我们、你、他、她、它、他们、她们、它们、其”等形式指代前文已经出现的具体人物或机构。
    * 链中必须包含至少一个明确名词短语做先行词（如“海岳自然资源局局长谢卓锦”），后面的人称代词是 anaphor，不能只用代词自成一链。

  - 指示代词：
    * 用“这/那/此/该”等单独出现，或和名词组合成“这项措施”“该项目”“此轮检查”等，对前文刚提到的实体或事件进行指代。
    * 若短语强调的是“这项工作/该次行动本身”且与一个具体事件同指，可以视为事件指代；否则一般按“实体指代”处理。

  - 有定描述：
    * 不用人称代词，也不用“这/那/该”等指示词，而是凭借描述让读者唯一识别的名词短语。
      例如：“基金负责人”“负责整改的小组”“临江北站星河基金会”“青岚自然资源局局长”等。
    * 要求先行词和照应语是**同一个现实实体**的不同称呼，而不是“整体 vs 部分”或“机构 vs 负责人”等包含关系。
      例如“青岚自然资源局”和“局长喻尧渊”不是同一个实体，不能放在同一有定描述链中。

  - 零指代：
    * 本应出现主语/宾语等句法成分，但在上下文中被省略，只能依靠前文才能理解。
    * 用单独的字符“Ø”标出省略位置（例如“Ø将通过强化审计与现场检查实现目标”）。
    * 在 coreference_chains 中：
        - 每条零指代链必须有且只有一个 role="antecedent" 的 mention（明确出现过的名词短语），其余 mentions 全部是 role="anaphor" 且 text="Ø"。
        - sentences 中也必须在对应位置真实写出字符“Ø”，不要写成“Ø(将通过)”这类形式。
        - 零指代必须回指前文的具体实体，不能用来表达泛指或新引入的实体；也不要在整篇第一句就出现零指代。

  - 事件指代：
    * 指代的是“事件/行为/过程”，而不是实体本身。
    * 先行词通常是带动词的事件短语（如“开展融资风险排查”“完成首轮督查”“公开款项执行明细”）。
    * 照应语可以是“这次行动”“该轮检查”“上述安排”“相关结果”等，但语义上必须指向同一个事件，而不是事件产生的“数据/报告/预算”等结果。
    * 事件链必须提供 trigger 字段，trigger.text 为一个具体的动词或事件名词（如“排查”“督查”“审计”“整改”）。

  【类型判定决策流程（务必按此先判断，再选择 type）】
  1) 它有没有字面形式？
     - 没有任何词形，只是在句法上存在主语/宾语等成分缺省：
       若能明确回指前文的具体实体或事件 → 标记为“零指代”；否则不要强行标零指代。
     - 有明确词形（代词/名词短语等） → 进入步骤 2。
  2) 它指的是“实体”还是“事件”？表面是否带指示词？
     - 事件类：由动词或事件性名词触发，语义是某个行为/过程/安排（可以带“这次/该项/上述”等指示成分）→ 统一归入“事件指代”链。
     - 实体类：指向人物/机构/物品等 → 再按表面形式区分：
         * 带“这/那/该/此/上述”等 → “指示代词”链；
         * 不带指示词，但在当前语境下是唯一可识别实体 → “有定描述”链；
         * 典型人称代词（他/她/它/他们/我们/你们/其等） → “人称代词”链。

  硬性要求：
  - sentences 必须 6–9 句，且 text 为它们无损拼接；至少给出 3 条不同类型的链；每条链 ≥1 个回指（anaphor）。
  - 所有 mentions.text 必须逐字出现在对应句子里；
    对零指代来说，anaphor 的 text 一律写成 "Ø"，并且句子中也要真实出现该字符；
    不要生成 "Ø(将通过)" 等形式；不要生成 char_start/char_end（由系统计算）。
  - 若某条链 type=事件指代，必须额外给出 trigger 字段（sentence_id+text），触发词逐字出现在对应句子里；不要生成 trigger 的 char_start/char_end。
  - 段落背景必须设定在 2025 年及以后，但正文不必显式写出年份。
    如果你写到了具体年份，只能写 2025 年或之后；不要出现 2024 年及以前。
  - 首句不得以具体年份开头（例如“2025年……/2026年……”）。
  - 句式去模板化（倒装/并列/插入语/因果/让步等交替），避免套话。

  【起笔提示】：{OPEN_STYLE}
  【话题简述】：{TOPIC}
  【避免相似主题】：{NEG_LIST}
  """

VERIFY_PROMPT = """你将得到一个 JSON（可能有错误）。请**只修复 JSON**使其满足规则，然后只输出修复后的一个 JSON 对象。

  规则：
  1) sentences 6–9 句，text 为无损拼接；
  2) 至少 3 条不同 type 的 coreference_chains；
  3) 每条链 mentions 至少 2 个，且 text 必须逐字出现在对应句子里；
  4) antecedent 在 anaphor 之前；至少一条链跨句；
  5) 不要生成 char_start/char_end；
  6) 若存在 type=事件指代 的链，必须提供 trigger 字段（包含 sentence_id 和 text），触发词逐字出现在对应句子里；不要生成 trigger 的 char_start/char_end。
  7) 背景不得出现 2024 年及以前的明确年份；若出现年份只能是 2025 及以后；首句不得年份起首。
     正文允许不出现任何具体年份。
  8) 若存在 type=零指代 的链：
     - 每条链必须恰好 1 个 role="antecedent" 的 mention，其他 mentions 都是 role="anaphor" 且 text="Ø"；
     - 对应句子中也必须包含字符“Ø”，不能出现 "Ø(将…)" 等形式。

  【类型区分提醒】
  - 人称代词链：必须包含人称代词（他/她/它/他们/我们/你们/其等）和对应的实体名词短语；
  - 指示代词链：用“这/那/此/该（项/举措/项目/工作等）”指代已提到的实体或事件；
  - 有定描述链：用描述性名词短语（负责人、某局党委、某基金会等）指同一实体，避免把“机构”和“该机构负责人”放在同一实体链；
  - 零指代链：只在句法上有成分缺省、且该缺省能回指前文具体实体/事件时使用，anaphor 的 text 必须是单独的 "Ø"；
  - 事件指代链：只有在 mentions 和 trigger 都指同一事件时才使用，避免把“行动产生的结果/数据/报告”当成这个事件本身。

  【类型判定决策流程（务必遵守）】
  1) 它有没有字面形式？
     - 没有任何词形，只是句法上有主语/宾语等成分缺省：若能回指前文具体实体/事件，则标“零指代”；否则不要标零指代。
     - 有明确词形 → 进入步骤 2。
  2) 它指的是“实体”还是“事件”？表面是否带指示词？
     - 事件类：由动词或事件性名词触发，语义是某个行为/过程/安排（可带“这次/该项/上述”等）→ 统一放到“事件指代”链。
     - 实体类：指向人物/机构/物品等 →
         * 带“这/那/该/此/上述”等 → “指示代词”链；
         * 不带指示词，但在当前语境下是唯一可识别实体 → “有定描述”链；
         * 典型人称代词（他/她/它/他们/我们/你们/其等） → “人称代词”链。

  必须保留键名与整体结构。

  【话题简述】：{TOPIC}
  【期望领域 domain】：{DOMAIN}
  【期望主类型 category_primary】：{PRIMARY}
  【错误原因】：{REASONS}
  【待修复 JSON】：{BAD_JSON}
  """


def call_deepseek(prompt: str, temperature: float, top_p: float, model_name: str) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


# =============== JSON 解析/去重/对齐工具 ===============


def strip_code_fences(s: str) -> str:
    m = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    return "\n".join(m) if m else s


def sanitize_json_like(s: str) -> str:
    s = strip_code_fences(s)
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", '"')
        .replace("’", '"')
        .replace("：", ":")
    )
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r",(\s*[\]}])", r"\1", s)
    return s.strip()


def try_load_obj(text: str) -> Optional[dict]:
    s = (text or "").strip()
    if not s:
        return None
    i = s.find("{")
    if i != -1:
        depth, j = 0, -1
        for k, ch in enumerate(s[i:], i):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    j = k
                    break
        if j != -1:
            cand = s[i : j + 1]
            for v in [cand, sanitize_json_like(cand)]:
                try:
                    obj = json.loads(v)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
    try:
        obj = json.loads(sanitize_json_like(s))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def get_ngrams(s: str, n: int) -> set:
    s = re.sub(r"\s+", "", s)
    return set(s[i : i + n] for i in range(max(len(s) - n + 1, 0)))


def jaccard7(a: str, b: str) -> float:
    A, B = get_ngrams(a, 7), get_ngrams(b, 7)
    return (len(A & B) / len(A | B)) if (A and B) else 0.0


def stable_hash64(x: str) -> int:
    h = hashlib.md5(x.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def simhash_3gram(s: str) -> int:
    s = re.sub(r"\s+", "", s)
    feats = [s[i : i + 3] for i in range(max(len(s) - 2, 0))]
    bits = [0] * 64
    for f in feats:
        h = stable_hash64(f)
        for b in range(64):
            bits[b] += 1 if ((h >> b) & 1) else -1
    v = 0
    for b in range(64):
        if bits[b] > 0:
            v |= 1 << b
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


DEICTIC_PREFIX = (
    "这项",
    "这种",
    "这位",
    "这一",
    "这个",
    "这些",
    "这",
    "此",
    "该",
    "该项",
    "该举措",
    "该措施",
    "该工程",
    "该决定",
    "该工作",
    "该项目",
)


def _normalize_head(m: str) -> str:
    s = m.strip()
    for p in DEICTIC_PREFIX:
        if s.startswith(p):
            s = s[len(p) :]
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
            if all((span.stop <= u.start or span.start >= u.stop) for u in used_spans):
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


def _normalize_trigger(trig: Any) -> Optional[Tuple[int, str]]:
    if not isinstance(trig, dict):
        return None
    if "sentence_id" not in trig or "text" not in trig:
        return None
    try:
        sid = int(trig["sentence_id"])
    except Exception:
        return None
    txt = str(trig["text"]).strip()
    if not txt:
        return None
    return sid, txt


def add_char_spans(sample: dict) -> Optional[dict]:
    try:
        sents = sample["sentences"]
        chains = sample["coreference_chains"]
        assert isinstance(sents, list) and 6 <= len(sents) <= 9
        assert isinstance(chains, list) and len(chains) >= 3
    except Exception:
        return None

    per_sent_used: Dict[int, List[range]] = {i + 1: [] for i in range(len(sents))}
    new_chains: List[Dict[str, Any]] = []

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

        mentions_new: List[Dict[str, Any]] = []
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
            mentions_new.append(
                {
                    "sentence_id": sid,
                    "role": role,
                    "text": txt,
                    "char_start": st,
                    "char_end": ed,
                }
            )

        ch2 = dict(ch)
        ch2["mentions"] = mentions_new
        ch2["start_sentence"] = int(ch.get("start_sentence", start_s))
        ch2["end_sentence"] = int(ch.get("end_sentence", end_s))
        ch2["index"] = int(ch.get("index", ci))

        if ch2.get("type") == "事件指代":
            trig_norm = _normalize_trigger(ch.get("trigger"))
            if trig_norm is None:
                return None
            trig_sid, trig_txt = trig_norm
            if not (1 <= trig_sid <= len(sents)):
                return None
            used = per_sent_used[trig_sid]
            span = find_span(sents[trig_sid - 1], trig_txt, used)
            if span is None:
                return None
            st, ed = span
            used.append(range(st, ed))
            ch2["trigger"] = {
                "sentence_id": trig_sid,
                "text": trig_txt,
                "char_start": st,
                "char_end": ed,
            }

        new_chains.append(ch2)

    out = dict(sample)
    out["coreference_chains"] = new_chains
    return out


# =============== 候选轻量打分 ===============


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


# =============== 验证 & 自动修复 ===============

YEAR_START_RE = re.compile(r"^\s*20(25|26|27|28|29)\s*年")
YEAR_PRE_2025_RE = re.compile(r"20(0[0-9]|1[0-9]|2[0-4])\s*年")


def validate_basic(obj: dict) -> Tuple[bool, str]:
    text = obj.get("text") or ""
    sents = obj.get("sentences")
    chains = obj.get("coreference_chains")

    if not isinstance(text, str) or len(text) < DOC_MIN_LEN:
        return False, "text_too_short"
    if not isinstance(sents, list) or not (6 <= len(sents) <= 9):
        return False, "bad_sentences_len"
    if not isinstance(chains, list) or len(chains) < 3:
        return False, "too_few_chains"

    # 事件触发词必须存在
    for ch in chains:
        if isinstance(ch, dict) and ch.get("type") == "事件指代":
            if _normalize_trigger(ch.get("trigger")) is None:
                return False, "missing_or_bad_event_trigger"

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
                if not (isinstance(sents, list) and 1 <= sid <= len(sents)):
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

    # 主类型必须真实出现在链类型中，防止乱标 category_primary
    primary = obj.get("category_primary")
    if isinstance(primary, str):
        chain_types = set()
        for ch in chains:
            if isinstance(ch, dict):
                t = ch.get("type")
                if t:
                    chain_types.add(t)
        if primary in COREf_TYPES and primary not in chain_types:
            return False, f"primary_type_missing:{primary}"

    # 首句不得年份起首
    try:
        first_sent = sents[0] if sents else ""
        if isinstance(first_sent, str) and YEAR_START_RE.search(first_sent):
            return False, "bad_time_anchor_year_start"
    except Exception:
        return False, "bad_time_anchor_year_start"

    # 文本中不得出现 2024 及以前年份
    if YEAR_PRE_2025_RE.search(text):
        return False, "contains_pre_2025_year"

    return True, "ok"


def try_repair(
    obj: dict, tb: str, domain: str, primary: str, reason: str, model_name: str
) -> Optional[dict]:
    bad_json = json.dumps(obj, ensure_ascii=False)
    prompt = VERIFY_PROMPT.format(
        TOPIC=tb, DOMAIN=domain, PRIMARY=primary, REASONS=reason, BAD_JSON=bad_json
    )
    raw_fix = call_deepseek(
        prompt, temperature=0.2, top_p=0.95, model_name=model_name
    )
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


# =============== 读写工具 ===============


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
    OUT_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUT_TXT.write_text(
        OUT_TXT.read_text(encoding="utf-8")
        + rec.get("text", "").replace("\n", " ").strip()
        + "\n",
        encoding="utf-8",
    )


def count_hist_by_domain(hist: list) -> Dict[str, int]:
    c = {d: 0 for d in DOMAINS}
    for x in hist:
        if isinstance(x, dict):
            d = x.get("domain")
            if d in c:
                c[d] += 1
    return c


# =============== 主流程 ===============


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per_domain", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--model_version", type=str, default="unknown")
    args = ap.parse_args()

    master_seed = (
        args.seed
        if args.seed is not None
        else (int(time.time() * 1e6) & 0xFFFFFFFF)
    )
    random.seed(master_seed)

    if args.overwrite:
        OUT_JSON.write_text("[]", encoding="utf-8")
        OUT_TXT.write_text("", encoding="utf-8")
        if RAW_DIR.exists():
            shutil.rmtree(RAW_DIR)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for p in [
            GEN_LOG_PATH,
            MANIFEST_PATH,
            TOPIC_MEM_PATH,
            GEN_COUNTER_PATH,
        ]:
            if p.exists():
                p.unlink()
        save_topic_memory([])
        save_gen_counter(0)
    else:
        ensure_out()
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    hist = read_json_list(OUT_JSON)
    hist_txt = [x.get("text", "") for x in hist if isinstance(x, dict)]
    hist_sim = [simhash_3gram(t) for t in hist_txt]

    if args.per_domain is not None:
        domain_targets = {d: args.per_domain for d in DOMAINS}
        total_limit = args.per_domain * len(DOMAINS)
    else:
        total_limit = args.limit or 20
        avg = (len(hist) + total_limit) // len(DOMAINS)
        domain_targets = {
            d: max(avg - count_hist_by_domain(hist)[d], 0) for d in DOMAINS
        }
        if sum(domain_targets.values()) < total_limit:
            for d in DOMAINS:
                domain_targets[d] += (
                    total_limit - sum(domain_targets.values()) + len(DOMAINS) - 1
                ) // len(DOMAINS)
                if sum(domain_targets.values()) >= total_limit:
                    break

    topic_mem = load_topic_memory()
    topics = plan_topics_quota(total_limit, topic_mem, domain_targets)
    if len(topics) < total_limit:
        print(
            f"[WARN] 可用离散话题不足 {total_limit}，实际仅 {len(topics)}。"
        )

    gen_counter = load_gen_counter()
    accepted_ids: List[str] = []
    ok = 0
    used_this_round: List[str] = []

    # 当前批次（以 BATCH_SIZE 条样本为一轮）内，各主类型已生成的数量
    BATCH_SIZE_LOCAL = BATCH_SIZE
    batch_type_counts: Dict[str, int] = {t: 0 for t in COREf_TYPES}

    for tb in topics:
        # 每当当前批次已生成满 BATCH_SIZE_LOCAL 条样本时，重置批次计数
        if ok > 0 and ok % BATCH_SIZE_LOCAL == 0:
            batch_type_counts = {t: 0 for t in COREf_TYPES}

        domain = (
            tb.split("｜", 1)[0] if "｜" in tb else random.choice(DOMAINS)
        )
        # 按当前批次的类型缺口选择本条样本的主类型，避免某类在一批中缺失
        primary = pick_primary_for_batch(batch_type_counts, BATCH_SIZE_LOCAL)

        neg_list = used_this_round[-6:] + topic_mem[-12:] + FORBID_SEEDS
        open_style = random.choice(OPENING_STYLE_BANK)

        prompt = WRITE_PROMPT.format(
            PRIMARY=primary,
            DOMAIN=domain,
            TOPIC=tb,
            NEG_LIST="；".join(neg_list),
            OPEN_STYLE=open_style,
        )
        prompt_hash = "sha256:" + sha256_hex(prompt)

        best = None
        best_score = -1
        best_gen_id = None

        for attempt in range(MAX_TRIES):
            cand_seed_base = random.randint(0, 2**31 - 1)
            T = random.uniform(*TEMP_RANGE)
            P = random.uniform(*TOPP_RANGE)

            for cand_i in range(1, N_CANDS + 1):
                gen_counter += 1
                abbr = DOMAIN_ABBR.get(domain, "UNK")
                gen_id = f"{abbr}-{gen_counter:06d}-DeepSeek-cand{cand_i}"

                raw = call_deepseek(prompt, T, P, model_name=args.model)
                RAW_DIR.joinpath(f"{gen_id}.txt").write_text(
                    raw, encoding="utf-8"
                )

                reasons: List[str] = []
                obj2: Optional[dict] = None
                obj = try_load_obj(raw)
                if not isinstance(obj, dict):
                    reasons.append("json_parse_failed")
                else:
                    obj.setdefault("category_primary", primary)
                    obj.setdefault("domain", domain)
                    obj.setdefault(
                        "entities", {"persons": [], "orgs": [], "locations": []}
                    )

                    ok_basic, reason = validate_basic(obj)
                    if not ok_basic:
                        reasons.append(reason)
                        obj2 = try_repair(
                            obj,
                            tb,
                            domain,
                            primary,
                            reason,
                            model_name=args.model,
                        )
                        if obj2 is None:
                            reasons.append("repair_failed")
                    else:
                        obj2 = add_char_spans(obj)
                        if obj2 is None:
                            reasons.append("span_alignment_failed")
                            obj2 = try_repair(
                                obj,
                                tb,
                                domain,
                                primary,
                                "span_alignment_failed",
                                model_name=args.model,
                            )
                            if obj2 is None:
                                reasons.append("repair_failed")

                passed_filters = False
                if obj2 is not None:
                    text2 = obj2["text"]
                    if any(
                        jaccard7(text2, ht) >= JACC_DOC_THR for ht in hist_txt
                    ):
                        reasons.append("jaccard_overlap")
                    else:
                        sh = simhash_3gram(text2)
                        if any(
                            hamming(sh, hs) <= HAMM_THR for hs in hist_sim
                        ):
                            reasons.append("simhash_overlap")
                        else:
                            passed_filters = True

                append_genlog(
                    {
                        "gen_id": gen_id,
                        "timestamp_utc": utc_now_iso(),
                        "model": args.model,
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
                    }
                )

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

        # 更新当前批次内该主类型的计数（只统计真正被接受写入的数据）
        primary_final = best.get("category_primary")
        if isinstance(primary_final, str) and primary_final in COREf_TYPES:
            batch_type_counts[primary_final] = batch_type_counts.get(
                primary_final, 0
            ) + 1

        hist_txt.append(best["text"])
        hist_sim.append(simhash_3gram(best["text"]))
        used_this_round.append(tb)
        topic_mem.append(topic_signature(tb))
        topic_mem = topic_mem[-TOPIC_MEM_MAX:]

        accepted_ids.append(best_gen_id)
        ok += 1
        print(
            f"[OK] {ok}/{len(topics)} :: {tb}  —— domain={domain}, primary={primary_final}, score={best_score}, gen_id={best_gen_id}"
        )

    save_topic_memory(topic_mem)
    save_gen_counter(gen_counter)

    manifest = {
        "accepted_gen_ids": accepted_ids,
        "build_time_utc": utc_now_iso(),
        "script": str(Path(__file__).name),
        "master_seed": master_seed,
        "args": vars(args),
        "filters": {
            "doc_min_len": DOC_MIN_LEN,
            "jaccard7_thr": JACC_DOC_THR,
            "simhash_hamming_thr": HAMM_THR,
            "min_chains": 3,
            "event_trigger_required": True,
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
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[DONE] 写入 {ok} 条 → {OUT_JSON}\nTXT → {OUT_TXT}\nLOGS → {LOG_DIR}"
    )


if __name__ == "__main__":
    main()
