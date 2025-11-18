# 공행성 쌍 개수 차이 분석

## 📊 현재 상황

- **preprocessing 모델**: 1,425개 쌍
- **ev_battery 모델**: 2,996개 쌍 (약 2.1배)

## 🔍 차이가 나는 이유

### 1. 필터링 조건의 차이

#### preprocessing 모델
```python
# 고정 임계값 사용
if best_lag is not None and abs(best_corr) >= corr_threshold:  # 0.4
    # 공행성쌍으로 채택
```

**특징**:
- 상관계수의 절댓값이 **0.4 이상**이어야만 채택
- 공급망 점수는 쌍 선택에 영향 없음 (단지 feature로만 사용)

#### ev_battery 모델
```python
# 클러스터 가중치 적용
cluster_boost = get_cluster_boost(leader_cluster, follower_cluster)
effective_corr = abs(best_corr) * cluster_boost

# 가중치가 적용된 상관계수로 판단
if best_lag is not None and effective_corr >= corr_threshold:  # 0.4
    # 공행성쌍으로 채택
```

**특징**:
- 상관계수에 **클러스터 가중치를 곱한 값**으로 판단
- 가중치가 높으면 낮은 상관계수도 채택 가능

### 2. 클러스터 가중치의 영향

| 상황 | 가중치 | 효과 |
|------|--------|------|
| battery_materials 내부 | 2.0 | 상관계수 0.2 이상도 채택 가능 |
| ev_machinery_electrical 내부 | 1.8 | 상관계수 0.22 이상도 채택 가능 |
| battery_materials → ev_machinery_electrical | 1.5 | 상관계수 0.27 이상도 채택 가능 |
| other_industries 내부 | 1.2 | 상관계수 0.33 이상도 채택 가능 |
| 기타 | 1.0 | 상관계수 0.4 이상만 채택 |

**예시**:
- `best_corr = 0.25`, `cluster_boost = 2.0` → `effective_corr = 0.5` ✅ 채택
- `best_corr = 0.25`, `cluster_boost = 1.0` → `effective_corr = 0.25` ❌ 제외

### 3. 실제 데이터 분석

ev_battery 모델의 클러스터별 분포:
```
battery_materials × battery_materials: 1,479개 (가장 많음!)
ev_machinery_electrical × ev_machinery_electrical: 123개
battery_materials × ev_machinery_electrical: 312개
other_industries × other_industries: 329개
기타 조합: 753개
```

**관찰**:
- `battery_materials` 내부 쌍이 가장 많음 (가중치 2.0)
- 같은 클러스터 내 쌍이 전체의 약 65% 차지

## 📈 수학적 분석

### preprocessing 모델
```
채택 조건: |corr| >= 0.4
```

### ev_battery 모델
```
채택 조건: |corr| * boost >= 0.4

따라서:
- boost = 2.0 → |corr| >= 0.2
- boost = 1.8 → |corr| >= 0.222
- boost = 1.5 → |corr| >= 0.267
- boost = 1.2 → |corr| >= 0.333
- boost = 1.0 → |corr| >= 0.4
```

**결과**: ev_battery 모델이 더 많은 쌍을 채택할 수 있음

## ⚠️ 주의사항

### 장점
- ✅ 도메인 지식 반영 (EV 배터리 클러스터 간 연관성)
- ✅ 더 많은 쌍 탐색 (커버리지 증가)
- ✅ 클러스터 내 연관성 강조

### 단점
- ⚠️ 낮은 상관계수 쌍 포함 (노이즈 가능성)
- ⚠️ 가중치 값이 주관적 (도메인 지식에 의존)
- ⚠️ 과적합 위험 증가

## 🔧 조정 방법

### 1. 가중치 조정
```python
# 가중치를 낮추면 더 엄격한 필터링
if leader_cluster == 'battery_materials':
    return 1.5  # 2.0 → 1.5로 낮춤
```

### 2. 이중 필터링
```python
# 가중치 적용 + 최소 상관계수 조건
if best_lag is not None and effective_corr >= 0.4 and abs(best_corr) >= 0.3:
    # 공행성쌍으로 채택
```

### 3. 가중치별 최소 상관계수
```python
# 가중치가 높을수록 최소 상관계수도 높게
min_corr_by_boost = {
    2.0: 0.25,  # 가중치 2.0일 때 최소 0.25
    1.8: 0.28,
    1.5: 0.30,
    1.2: 0.33,
    1.0: 0.40
}
min_corr = min_corr_by_boost.get(cluster_boost, 0.4)
if abs(best_corr) >= min_corr and effective_corr >= 0.4:
    # 공행성쌍으로 채택
```

## 📊 권장사항

1. **현재 설정 유지**: 도메인 특화가 중요한 경우
2. **가중치 완화**: 노이즈를 줄이고 싶은 경우
3. **이중 필터링**: 양쪽 장점을 모두 취하고 싶은 경우

## 📝 결론

**차이가 나는 핵심 이유**:
- ev_battery 모델은 클러스터 가중치로 낮은 상관계수도 채택
- preprocessing 모델은 고정 임계값으로 엄격하게 필터링

**선택 기준**:
- 더 많은 쌍이 필요하면 → ev_battery 모델
- 높은 품질의 쌍만 필요하면 → preprocessing 모델
- 균형잡힌 접근 → 가중치 조정 또는 이중 필터링

