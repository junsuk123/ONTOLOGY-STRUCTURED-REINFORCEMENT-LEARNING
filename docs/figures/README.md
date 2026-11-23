# 그림 목록

논문의 그림 1–9이다. 이미지는 논문 PDF에 실린 그림을 그대로 옮겼고, 캡션은 논문의 캡션이다.
본문 설명은 [저장소 README](../../README.md)의 해당 절에 있다.

| 그림 | 파일 | 논문 캡션 | README |
|---|---|---|---|
| 1 | [fig1_system_pipeline.png](fig1_system_pipeline.png) | System pipeline for autonomous landing on a moving platform: simulation, perception and state, ontology semantics, and policy and flight control. The bottom band shows the offline training flow. | 2.2 |
| 2 | [fig2_ontology_graph.png](fig2_ontology_graph.png) | Landing ontology graph with 19 nodes and seven relation types, drawn for the x plane; the y plane has the same nodes and relations and shares all weights. Relation types are distinguished by line style and number. | 3.2 |
| 3 | [fig3_actor_structures.png](fig3_actor_structures.png) | Actor structures of the proposed ORL (left) and the MLP PPO baseline (right). Both use the same critic, policy head, and PPO settings; the 4,016 and 4,022 parameters include the common critic. | 3.4 |
| 4 | [fig4_scenario_and_field.png](fig4_scenario_and_field.png) | Approach scenario and desired velocity field: (a) scenario draw at handover; (b) desired drone-minus-pad velocity with the pad in view; (c) the field after the view is lost for 1 s (climb blended in from 0.5 s). | 4.1 |
| 5 | [fig5_isaac_px4_demo.png](fig5_isaac_px4_demo.png) | Isaac Sim–PX4 demonstration flight: simulator (left), onboard camera with detected ArUco markers (upper right), and RViz trajectory (lower right). | 4.2 |
| 6 | [fig6_learning_speed.png](fig6_learning_speed.png) | Learning speed over ten paired seeds: (a) first update T trained at difficulty 1.0; (b) decisions collected before T (post hoc). Bars: medians; open triangles: seed-45 baselines, not reaching difficulty 1.0 in 600 updates. | 5.1 |
| 7 | [fig7_equal_budget.png](fig7_equal_budget.png) | Success rate under the same training budget: (a) ten-cell success rate per seed and the mean; (b) ORL minus each baseline with 97.5% intervals (open: not tested). Seed 54: baselines stayed at the lowest difficulty. | 5.2 |
| 8 | [fig8_verdicts.png](fig8_verdicts.png) | Verdicts of the 600-update policies (50 cases per bar): (a) Isaac Sim–PX4 closed-loop flights; (b) the same 450 cases in the local plant. Comp. ID/OOD: trained/held-out directions; Comb. ID: combined condition. | 5.3 |
| 9 | [fig9_trajectory.png](fig9_trajectory.png) | One successful ORL landing replayed in the local plant (seed 59; also successful in Isaac Sim–PX4): (a) horizontal paths of drone and pad; (b) relative position and velocity; (c) height and vertical velocity. | 5.3 |

- 그림 6–8의 PPO와 Untyped GNN은 본문의 MLP PPO와 untyped-graph PPO이다.
- 그림 9는 로컬 플랜트 재생이다. Isaac Sim 비행의 시계열은 저장하지 않았다.

---

Copyright © 2026 Jun Suk Kim (김준석). 관리자: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>. Apache License 2.0 ([LICENSE](../../LICENSE)).
