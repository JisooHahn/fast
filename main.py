from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:10000",
    "http://43.201.250.218:10000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

# 문장 임베딩 모델 로드 (의미 기반 비교 가능)
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')


# ▶️ 입력 데이터 모델 정의
class FeatureItem(BaseModel):
    # id: int  # 각 문장의 고유 식별자
    bookIsbn: int
    # content: str  # 비교 대상이 되는 문장
    bookSummary: str

class RecommendRequest(BaseModel):
    target: str  # 새로 입력된 비교 기준 문장
    feature: List[FeatureItem]  # 기존 문장 리스트


# ▶️ 출력 데이터 모델 정의
class RecommendResponse(BaseModel):
    # recommendedPostId: int     # 가장 유사한 문장의 id(단일 결과용)
    # recommendedPostIds: List[int]  # ✅ 리스트로 변경
    recommendedBookIsbns: List[int]


# ▶️ 추천 API 엔드포인트
@app.post("/recommend", response_model=RecommendResponse)
def recommend(data: RecommendRequest):
    target_text = data.target
    features = data.feature

    # 예외 처리: 리스트가 비어 있으면 유사도 분석 불가
    if not features:
        raise HTTPException(status_code=400, detail="Feature list is empty.")

    # 기존 문장들에서 content만 따로 추출
    # contents = [item.content for item in features]
    contents = [item.bookSummary for item in features]

    # 전체 문장 리스트 = [target 문장] + 기존 content 리스트
    all_sentences = [target_text] + contents

    # 임베딩 생성 (문장들을 벡터로 변환)
    embeddings = model.encode(all_sentences, convert_to_tensor=True)

    # 코사인 유사도 계산: target vs 전체 문장들
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1:])[0]  # target vs features

    # 가장 유사한 문장 인덱스 (features 기준이므로 0번째가 아닌 것 주의!), Top 1 가져오기
    # most_similar_idx = int(cosine_scores.argmax())

    # 해당 인덱스의 id 반환
    # recommended_post_id = features[most_similar_idx].id

    # ✅ 유사도 높은 순 Top 3 인덱스 추출
    top_k = 9
    sorted_indices = cosine_scores.argsort(descending=True)[:top_k]

    # ✅ 인덱스를 통해 해당 id 추출
    # recommended_post_ids = [features[int(idx)].id for idx in sorted_indices]
    recommended_book_isbns = [features[int(idx)].bookIsbn for idx in sorted_indices]

    print("\n[🔍 유사도 결과]")
    for idx, score in enumerate(cosine_scores):
        # print(f"id: {features[idx].id}, content: '{features[idx].content}', score: {score.item():.4f}")
        print(f"isbn: {features[idx].bookIsbn}, summary: '{features[idx].bookSummary}', score: {score.item():.4f}")

    # return RecommendResponse(recommendedPostIds=recommended_post_ids)
    return RecommendResponse(recommendedBookIsbns=recommended_book_isbns)