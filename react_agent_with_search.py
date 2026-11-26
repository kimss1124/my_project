import os, json
from dotenv import load_dotenv # type: ignore
from anthropic import Anthropic # type: ignore

load_dotenv()
client = Anthropic()

REACT_SYSTEM_PROMPT = """
당신은 Tool을 사용할 수 있는 AI 에이전트입니다. 반드시 다음 형식을 따릅니다.
Thought: 지금까지 알아낸 것과 아직 부족한 정보를 한두 문장으로 분석합니다.
Action: 그 판단에 따라 필요한 Tool을 호출합니다.
Observation: Tool 실행 결과이며, 시스템이 제공합니다.
Tool을 호출하기 전에는 반드시 Thought를 텍스트로 먼저 출력합니다.
Observation은 직접 지어내지 말고 시스템이 준 값만 사용합니다.
Observation을 받으면 다시 Thought로 돌아가 위 과정을 반복합니다.
충분한 정보가 모이면 Thought 없이 최종 답변만 출력합니다.
"""

# 날씨 Tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회합니다."""
    # 학습용으로 더미 데이터를 반환합니다
    weather_data = {
        'Seoul': {'temp': 18.5, 'description': '맑음', 'humidity': 45},
        'Tokyo': {'temp': 15.2, 'description': '가벼운 비', 'humidity': 78},
        'New York': {'temp': 22.1, 'description': '구름 많음', 'humidity': 55},
    }
    data = weather_data.get(city, {'temp': 0, 'description': '알 수 없음', 'humidity': 0})
    return json.dumps({'city': city, **data}, ensure_ascii=False)

# 계산기 Tool
def calculate(expression: str) -> str:
    """수학 계산식을 계산합니다."""
    try:
        result = eval(expression, {'__builtins__': {}})
        return json.dumps({'expression': expression, 'result': result})
    except Exception as e:
        return json.dumps({'error': str(e)})

# 웹 검색 Tool
def search_web(query: str) -> str:
    """웹에서 검색어로 정보를 검색합니다."""
    # 학습용으로 더미 데이터를 반환합니다
    dummy_results = {
        '파이썬': [
            {'title': 'Python.org 공식 홈페이지', 'snippet': '파이썬은 배우기 쉽고 강력한 프로그래밍 언어입니다.'},
            {'title': '파이썬 입문 가이드', 'snippet': '변수, 조건문, 반복문, 함수 등 파이썬 기초 문법을 소개합니다.'},
        ],
        '에이전트': [
            {'title': 'AI 에이전트 아키텍처 개요', 'snippet': 'LLM을 기반으로 추론(Thought), 행동(Action), 관찰(Observation)을 반복하는 ReAct 패턴을 활용합니다.'},
            {'title': '하네스 엔지니어링과 에이전틱 AI', 'snippet': '도구(Tool) 연동과 메모리 시스템을 통해 자율적으로 작업을 수행하는 에이전트를 구축합니다.'},
        ],
    }
    for key, results in dummy_results.items():
        if key in query:
            return json.dumps({'query': query, 'results': results}, ensure_ascii=False)

    default_results = [
        {
            'title': f"'{query}'에 대한 웹 검색 결과 1",
            'snippet': f"{query}에 관한 최신 정보 및 핵심 개념 요약입니다.",
        },
        {
            'title': f"'{query}'에 대한 웹 검색 결과 2",
            'snippet': f"{query}의 활용 사례와 심화 가이드를 확인할 수 있습니다.",
        },
    ]
    return json.dumps({'query': query, 'results': default_results}, ensure_ascii=False)

# Tool 매핑
tool_map = {
    'get_weather': get_weather,
    'calculate': calculate,
    'search_web': search_web,
}

# Tool 스키마 (Anthropic 형식)
tools = [
    {
        'name': 'get_weather',
        'description': '도시의 현재 기온, 날씨 상태, 습도를 조회합니다.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'city': {'type': 'string',
                         'description': '영문 도시 이름 (예: Seoul, Tokyo)'}
            },
            'required': ['city'],
        },
    },
    {
        'name': 'calculate',
        'description': '수학 계산식을 평가합니다. 사칙연산, 거듭제곱을 지원합니다.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'expression': {'type': 'string',
                               'description': '계산할 수식 (예: 2+3*4)'}
            },
            'required': ['expression'],
        },
    },
    {
        'name': 'search_web',
        'description': '웹에서 최신 정보나 일반 지식을 검색어로 조회합니다.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string',
                          'description': '검색할 키워드 또는 질문'}
            },
            'required': ['query'],
        },
    },
]

def run_react_agent(user_message, max_iterations=5):
    print(f'\n{"="*50}')
    print(f'사용자: {user_message}')
    print('='*50)

    messages = [
        {'role': 'user', 'content': user_message},
    ]

    for i in range(max_iterations):
        response = client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=1024,
            system=REACT_SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )

        # Thought 출력 (LLM이 생성한 텍스트 블록)
        thought = ''.join(b.text for b in response.content if b.type == 'text')
        if thought:
            print(f'\n[Thought] {thought}')

        # Tool 호출이 없으면 최종 답변
        if response.stop_reason != 'tool_use':
            print(f'[Answer] {thought}')
            return thought

        # Action: Tool 호출 가져오기
        messages.append({'role': 'assistant', 'content': response.content})
        tool_results = []
        for block in response.content:
            if block.type != 'tool_use':
                continue
            # 선택된 Tool 이름
            name = block.name
            args = block.input
            print(f'[Action] {name}({args})')

            # Tool 실행
            result = tool_map[name](**args)
            print(f'[Observation] {result}')

            # 결과를 모은다
            tool_results.append({
                'type': 'tool_result',      # Tool 사용 결과 블록
                'tool_use_id': block.id,    # tool_use 아이디로 매핑한다
                'content': result,
            })

        # Tool 결과를 user 메시지로 전달
        messages.append({'role': 'user', 'content': tool_results})

    print('\n[경고] 최대 반복 횟수에 도달했습니다.')
    return '죄송합니다. 처리 중 문제가 발생했습니다.'

# run_react_agent('서울 날씨 어때?')
# run_react_agent('서울과 도쿄 기온 차이를 계산해줘')

run_react_agent('최근 AI 에이전트 트렌드를 검색해줘')
