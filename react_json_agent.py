import re
import os, json

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

# 날씨 Tool 함수 (4.2와 동일)
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회합니다."""
    weather_data = {
        'Seoul': {'temp': 18.5, 'description': '맑음', 'humidity': 45},
        'Tokyo': {'temp': 15.2, 'description': '가벼운 비', 'humidity': 78},
        'New York': {'temp': 22.1, 'description': '구름 많음', 'humidity': 55},
    }
    data = weather_data.get(city.title(), {'temp': 0, 'description': '알 수 없음', 'humidity': 0})
    return json.dumps({'city': city, **data}, ensure_ascii=False)

# 계산 Tool
def calculate(expression: str) -> str:
    """수학 계산식을 계산합니다."""
    try:
        result = eval(expression, {"__builtins__": None}, {}) 
        return json.dumps({'expression': expression, 'result': result})
    except Exception as e:
        return json.dumps({'error': str(e)})


# 커스텀 Tool 스키마
TOOLS = {
    'get_weather': {
        'func': get_weather,
        'description': '도시의 기온, 날씨, 습도를 조회합니다.',
        'parameters': {'city': '영문 도시 이름 (예: Seoul, Tokyo)'},
    },
    'calculate': {
        'func': calculate,
        'description': '수학식을 계산합니다.',
        'parameters': {'expression': '계산할 수식 (예: 2+3*4)'},
    },
}

# Tool 목록을 시스템 프롬프트에 포함
def build_tool_description() -> str:
    lines = ["사용 가능한 Tool:"]
    for name, info in TOOLS.items():
        lines.append(f"- {name}: {info['description']}")
        lines.append(f"  파라미터: {info['parameters']}")
    return '\n'.join(lines)


SYSTEM_PROMPT = f"""당신은 유능한 AI 에이전트입니다. 사용자의 질문에 답하기 위해 Tool을 사용할 수 있습니다. 

{build_tool_description()} 

**응답 규칙:** 
1. 반드시 JSON 형식으로만 응답하십시오. 
2. 정보가 부족하여 Tool이 필요하다면 'action' 형식을 사용합니다. 
3. Tool 결과(Observation)를 확인한 후 답변이 가능하다면 'answer' 형식을 사용합니다.
4. 마크다운 코드 블록(```)을 절대 사용하지 마십시오. JSON 객체 하나만, 다른 텍스트 없이 출력하십시오.

**JSON 응답 형식:** 
1) Tool 호출 시: 
{{"thought": "상황 분석", "action": "Tool_이름", "action_input": {{"매개변수명": "값"}}}} 

2) 최종 답변 시: 
{{"thought": "최종 분석", "answer": "사용자에게 전달할 답변"}} """

def parse_llm_response(text: str) -> dict:
    # LLM API 의 흔한 오류 ```json ... ``` 또는 ``` ... ``` 제거
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # 본문 중 첫 번째 { ... } 블록만 추출
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"thought": "Parsing failed", "answer": text}

def run_react_json_agent(user_message: str, max_iterations: int = 5):
    print(f'\n{"="*50}')
    print(f'사용자: {user_message}')
    print('='*50)

    messages = [
        {'role': 'user', 'content': user_message},
    ]

    for i in range(max_iterations):
        # Claude 모델을 호출합니다.
        response = client.messages.create(
            model='claude-sonnet-4-6',
            system=SYSTEM_PROMPT,
            max_tokens=1024,
            messages=messages,
        )

        # 응답에서 텍스트 블록만 추출
        raw_content = ''.join(b.text for b in response.content if b.type == 'text')
        parsed = parse_llm_response(raw_content)

        thought = parsed.get('thought', '')
        if thought:
            print(f'\n[Thought] {thought}')

        # 최종 답변인 경우
        if 'answer' in parsed:
            print(f'\n[Answer] {parsed["answer"]}')
            return parsed['answer']

        # Tool 호출인 경우
        action = parsed.get('action')
        action_input = parsed.get('action_input', {})

        if action in TOOLS:
            print(f'[Action] {action}({action_input})')

            # Tool 실행
            tool_func = TOOLS[action]['func']
            try:
                result = tool_func(**action_input)
            except Exception as e:
                result = f"Error executing tool: {str(e)}"

            print(f'[Observation] {result}')

            # 대화 기록 업데이트
            messages.append({'role': 'assistant', 'content': raw_content})
            messages.append({
                'role': 'user',
                'content': f"Observation: {result}\n\n위 결과를 바탕으로 다음 단계(추가 Tool 호출 또는 최종 답변)를 진행하세요."
            })
        else:
            print(f'[Error] 알 수 없는 Tool: {action}')
            break

    print('\n[경고] 최대 반복 횟수에 도달했습니다.')

# 실행
# run_react_json_agent('서울 날씨 어때?')

run_react_json_agent('서울과 도쿄 기온 차이를 계산해줘')