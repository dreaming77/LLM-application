import pandas as pd
import json
from collections import defaultdict
import jionlp as jio
import re

#系统提示模板
SYSTEM_PROMPTS = {
    '全科': "您是一位经验丰富的全科医生，请根据患者描述提供专业的医疗建议",
    '妇产科': "您是三甲医院妇产科专家，请针对孕产期健康问题提供专业指导",
    '儿科': "您是三甲医院儿科主任医师，请解答儿童健康相关问题",
    '骨科': "您是三甲医院骨科专家，请针对肌肉骨骼问题提供诊疗建议",
    '眼科': "您是三甲医院眼科主任医师，请解答视觉健康相关问题",
    '内科': "您是三甲医院内科专家，请针对内科疾病提供专业意见",
    '外科': "您是三甲医院外科专家，请解答创伤和手术治疗相关问题",
    '皮肤科': "您是三甲医院皮肤科专家，请解答皮肤健康相关问题"
}


def hybrid_department_identification(question):
    """混合科室识别方法"""
    # 预定义关键词集合（提高可维护性）
    orthopedic_keywords = ['骨折', '骨裂', '脱臼', '错位', '扭伤', '拉伤', '关节痛', '关节炎', '半月板', '韧带']
    surgical_keywords = ['伤', '肿', '出血', '伤口', '外伤', '手术', '缝合', '切除', '割伤', '刺伤', '砸伤']
    body_parts_orthopedic = ['骨', '关节', '腰', '椎', '膝盖', '髋', '股骨', '肋骨', '脊椎', '颈椎', '腰椎', '肩', '肘',
                             '腕', '踝']

    # 方法1：实体识别（优先级最高）
    try:
        entities = jio.ner.extract_medical_entities(question)
        if entities:
            entity_types = [e["type"] for e in entities]

            # 怀孕相关 - 妇产科（最高优先级）
            if "怀孕相关" in entity_types:
                return "妇产科"

            # 儿童相关 - 儿科
            if "儿童相关" in entity_types:
                return "儿科"

            # 皮肤症状 - 皮肤科
            if "皮肤症状" in entity_types:
                return "皮肤科"

            # 身体部位处理（多级判断）
            if "身体部位" in entity_types:
                body_part_entities = [e["text"] for e in entities if e["type"] == "身体部位"]

                # 骨科判断：骨骼部位或骨科症状关键词
                if any(any(kw in part for kw in body_parts_orthopedic) for part in body_part_entities) or \
                        any(kw in question for kw in orthopedic_keywords):
                    return "骨科"

                # 外科判断：创伤性症状关键词
                if any(kw in question for kw in surgical_keywords):
                    return "外科"

                # 默认内科
                return "内科"
    except Exception:
        # 实体识别失败时降级到关键词匹配
        pass

    # 方法2：关键词匹配（优先级顺序优化）
    keyword_mapping = [
        (r'怀孕|孕周|胎儿|流产|月经|分娩|孕妇|哺乳', '妇产科'),  # 妇产科特征词
        (r'宝宝|儿童|婴儿|小儿|新生儿|孩子|小孩|幼儿', '儿科'),  # 儿科特征词
        (r'腰椎|关节|骨头|骨折|腿疼|脚痛|腰疼|膝盖|脱臼|骨裂|脊柱', '骨科'),  # 骨科特征词
        (r'眼睛|视力|干涩|流泪|眼药水|角膜|近视|远视|白内障', '眼科'),  # 眼科特征词
        (r'痘痘|皮肤|皮疹|瘙痒|痤疮|疹子|红斑|脱皮', '皮肤科'),  # 皮肤科特征词
        (r'手术|伤口|缝合|外伤|切除|割伤|穿刺|创伤', '外科'),  # 外科特征词（先于内科）
        (r'心脏|胃|肠|肝|肺|肾|胰腺|胆囊', '内科'),  # 器官相关内科
        (r'头痛|恶心|发烧|感冒|腹泻|发热|头晕|呕吐|咳嗽|胸闷|心悸', '内科')  # 全身症状内科
    ]

    # 按定义顺序匹配（保证优先级）
    for pattern, department in keyword_mapping:
        if re.search(pattern, question):
            return department

    # 默认科室（当所有匹配失败时）
    return "全科"

QUESTION_CSV_PATH = 'dataset/cMedQA2/question.csv'
ANSWER_CSV_PATH = 'dataset/cMedQA2/answer.csv'
FILE_SAVE = 'dataset/medical_instruction_data.jsonl'

#读取数据
questions_df = pd.read_csv(QUESTION_CSV_PATH)
answers_df = pd.read_csv(ANSWER_CSV_PATH)

# 按问题ID分组，合并多个回答
grouped_answers = defaultdict(list)
for _, row in answers_df.iterrows():
    # 清理回答
    content = row['content'].strip()
    if content.startswith((':', '：')):
        content = content[1:].strip()
    grouped_answers[row['question_id']].append(content)

# 创建问题ID到问题内容的映射
question_map = {row['question_id']: row['content'] for _, row in questions_df.iterrows()}

# 创建转换后的数据结构
formatted_data = []
processed_questions = set()

for question_id, answers in grouped_answers.items():
    if question_id not in question_map:
        continue

    question_text = question_map[question_id]

    # 跳过重复问题
    if question_text in processed_questions:
        continue
    processed_questions.add(question_text)

    # 确定科室
    department = hybrid_department_identification(question_text)
    system_prompt = SYSTEM_PROMPTS.get(department, SYSTEM_PROMPTS['全科'])

    # 合并多个回答
    combined_response = "\n".join(
        [f"   · {answer}" for i, answer in enumerate(answers)]
    )

    # 添加医疗安全声明
    combined_response += "\n\n※ 重要提示：以上建议仅供参考，具体诊疗请咨询专业医疗机构 ※"

    # 构建对话结构
    conversation = {
        "system": system_prompt,
        "conversations": [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": combined_response}
        ]
    }
    formatted_data.append(conversation)

# 保存为JSONL文件
with open(FILE_SAVE, 'w', encoding='utf-8') as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"预处理完成！共生成 {len(formatted_data)} 条训练样本")
print(f"科室分布统计:")
for dept, count in pd.Series([item['system'] for item in formatted_data]).value_counts().items():
    print(f"- {dept.split('，')[0]}: {count}条")

