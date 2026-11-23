# 구현 안내

이 디렉터리는 [저장소 README](../README.md)에 옮긴 논문의 ORL 정책과 비교 정책의 코드이다.
방법과 결과는 저장소 README에 있고, 이 문서는 논문의 각 부분이 어느 코드에 있는지와 실행 방법만 적는다.

## 1. 논문 ↔ 코드

경로는 `python/ontology_rgat/current_semantics/` 기준이다.

| 논문 | 내용 | 모듈 |
|---|---|---|
| 2.1 | 현재 관측 `O_t` (21채널) | `observation.py` |
| 3.1 | 의미 특징 `φ_t` (44개), 유계 함수 `B(x; s)`, 유효 조건 | `features.py`, `geometry.py` |
| 3.2–3.6 | ORL: 노드 19개, 관계 7종, 개념별 인코더, 관계 그래프 어텐션 3회 적용, 평면 가중치 공유, 거울 평균, 결정 판독, 하강 게이트 | `orl.py` |
| 3.4, 4.2 | 관계 어텐션 층, 비교 정책(MLP PPO = `raw_ppo`, untyped-graph PPO = `untyped_gnn_ppo`), 공통 critic과 정책 머리, 체크포인트 | `models.py`, `graph.py` |
| 4.1 | 보상 (14)–(15), 목표 접근 속도장, 종료 보상 | `reward.py` |
| 4.1 | PPO, 성공률 기반 커리큘럼, 학습 루프 | `ppo.py`, `pilot.py` |
| 4.1 | 시나리오: 접근 방위 × 패드 진행 방향, 제외한 구역 쌍, 패드 가속 | `scenario.py` |
| 4.2 | 로컬 플랜트와 Isaac Sim–PX4 백엔드, 관측기, 행동 변환기 | `backend.py` |
| 4.3 | 평가 셀 10개 (표 4) | `cells.py` |
| 4.4, 5.1 | 학습 속도 비교 | `speed.py` |
| 4.4, 5.2–5.3 | 같은 예산 비교, Isaac Sim–PX4 평가 | `budget.py` |
| – | 계약(관측·특징·그래프·과제·학습 설정)의 해시, 실행 폴더와 잠금 | `contracts.py`, `experiment.py` |

- 논문의 ORL은 `orl.py`의 `orl/1`(모델 이름 `orl_ppo:eb68eb947a6a49a2`)이다.
- 실행 진입점은 저장소 루트의 `run.sh` → `python/run_current_semantics_3d.py`이다.

### 설정 (`config/current_semantics/`)

| 파일 | 내용 |
|---|---|
| `spatial_view_contact_v1.yaml` | 과제, 관측, 모델, 학습 설정 |
| `evaluation_cells_v1.yaml` | 평가 셀 10개 (표 4) |

- 세 비교의 분석 계획 파일(`learning_speed_v1.yaml`, `equal_budget_v1.yaml`, `isaac_equal_budget_v1.yaml`)은 저자가 보관하며 공개 저장소에 포함하지 않는다.
  - 각 단계는 이 폴더에서 계획 파일을 읽는다. 다른 위치의 파일은 `--plan`으로 지정한다.
  - 계획 파일이 없으면 실험 단계는 실행을 거부하고, 계획 파일이 필요한 테스트는 건너뛴다.

### 시뮬레이션과 비행 스택

| 경로 | 내용 |
|---|---|
| `python/ontology_rgat/spatial/` | 3축 착륙 플랜트, 패드 추적기, 착륙 판정, Isaac 백엔드 |
| `python/ontology_rgat/controllers/` | 가속도 명령 → 자세·추력 변환 |
| `python/ontology_rgat/direct_policy/`, `landing/`, `two_axis/`, `contracts/`, `benchmarks/` | 21채널 관측 인코딩, 종료 판정 표, 플랜트 설정 형식, 실험 프로필 로딩 |
| `python/ontology_rgat/bridge.py`, `stack.py`, `live_stack.py`, `config.py` | Isaac Sim + PX4 SITL + ROS 2 게이트웨이를 띄우고 연결하는 코드 |
| `isaac_sim/` | Isaac Sim 착륙 월드, 패드 운동, 센서, 외란 |
| `ros2_ws/src/ontology_rgat_px4/` | ROS 2 ↔ PX4 게이트웨이 |
| `config/spatial-isaac-system-v12.yaml` | Isaac Sim–PX4 비행 프로필 (같은 폴더의 상위 프로필을 상속) |
| `scripts/` | PX4·ROS 2·Pegasus 설치, Isaac·게이트웨이·DDS 에이전트 실행 |

## 2. 실행

모든 명령은 저장소 루트에서 실행하며, 위의 분석 계획 파일이 있어야 한다. Isaac Sim–PX4를 띄우는 단계는 `--allow-isaac`를 줄 때만 비행한다.

```bash
C=$PWD/Ontology_RGAT_UAV_RL_ISAAC_PX4/config/current_semantics

# 학습 속도 비교 (논문 5.1)
./run.sh current-semantics-3d --stage register-speed --plan $C/learning_speed_v1.yaml
./run.sh current-semantics-3d --stage speed --plan $C/learning_speed_v1.yaml
./run.sh current-semantics-3d --stage speed-report --plan $C/learning_speed_v1.yaml

# 같은 예산 비교 (논문 5.2)
./run.sh current-semantics-3d --stage register-budget
./run.sh current-semantics-3d --stage budget
./run.sh current-semantics-3d --stage budget-report

# Isaac Sim–PX4 평가 (논문 5.3)
./run.sh current-semantics-3d --stage register-isaac-budget
./run.sh current-semantics-3d --stage isaac-budget --allow-isaac --headless
```

- `register-*` 단계는 학습이나 비행 전에 분석 계획의 해시를 기록한다. 이후 단계는 계획이나 계약이 달라지면 실행을 거부한다.
- `speed`와 `budget`은 여러 프로세스가 모델을 나눠 맡을 수 있다. 에피소드마다 시드가 정해져 있어 `--workers`는 결과를 바꾸지 않는다.
- `run.sh`는 `OMP_NUM_THREADS=1`로 실행한다. 학습은 단일 스레드에서 비트 단위로 재현된다.
- `--dry-run`은 아무것도 만들지 않고 해석된 계획만 출력한다. `./run.sh current-semantics-3d --help`가 모든 옵션을 보여 준다.

## 3. 그림

- `tools/make_doc_figures.py`: 그림 1(`architecture`), 그림 3(`overall`), 그림 4(`field`), 그림 9(`trajectory`)를 코드에서 그린다. 출력은 `results/figures/`이다.
  - 그림 9는 Isaac Sim 사례 하나를 로컬 플랜트에서 다시 비행하며, 저장된 로컬 결과와 같은 판정과 결정 수가 나와야만 그린다.
- `tools/isaac_demo_capture.py`, `tools/demo_tf_relay.py`: 그림 5의 Isaac Sim + 카메라 + RViz 화면을 녹화한다(`--allow-isaac` 필요).
- `docs/figures/`의 그림은 논문에 실린 그대로이다.

## 4. 테스트

```bash
cd Ontology_RGAT_UAV_RL_ISAAC_PX4
python3 -m pytest -q tests                    # 전체, 약 30초
python3 -m pytest -q tests/current_semantics  # ORL과 세 비교만
```

학습이나 비행은 하지 않는다. ORL의 거울 등변성·평면 공유·국소성·파라미터 수, 계약 해시, 분석 계획 검증, 판정 통계, 플랜트와 비행 스택 코드를 확인한다.

---

Copyright © 2026 Jun Suk Kim (김준석). 관리자: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>. Apache License 2.0 ([LICENSE](../LICENSE)).
