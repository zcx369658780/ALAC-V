# DS-0002 Gate 1 审计

- 生成时间：2026-11-15T09:33:49.350503+00:00
- 总判定：**PASS**
- 场景/视图：train 2000/16000；validation 300/1500；test 900/4500
- 图像总量：0.97 GB

| 检查 | 结果 |
|---|---|
| counts_ok | True |
| split_isolation_ok | True |
| images_ok | True |
| eval_labels_ok | True |
| regeneration | {'samples': [{'split': 'train', 'scene': 0, 'sha_match': True, 'regen_sha': '4923c3f0371e19aadf044f04c779aff5530baea574642d277b91c7c1b59e438f', 'manifest_sha': '4923c3f0371e19aadf044f04c779aff5530baea574642d277b91c7c1b59e438f'}, {'split': 'train', 'scene': 1000, 'sha_match': True, 'regen_sha': '32bd3957d88401378d3cd9db6285a76cf69c5766324df7158ebc27e5e8175c9c', 'manifest_sha': '32bd3957d88401378d3cd9db6285a76cf69c5766324df7158ebc27e5e8175c9c'}, {'split': 'train', 'scene': 1999, 'sha_match': True, 'regen_sha': 'cdf86d0670c57a32b80daff6addeddb4856f699197b497ef478799bcd2850fe3', 'manifest_sha': 'cdf86d0670c57a32b80daff6addeddb4856f699197b497ef478799bcd2850fe3'}, {'split': 'validation', 'scene': 0, 'sha_match': True, 'regen_sha': 'a9e68562bf030438e8f80ba6bb3f2b359970fb5fa5b63ac7a4ce1b25bb8e19ca', 'manifest_sha': 'a9e68562bf030438e8f80ba6bb3f2b359970fb5fa5b63ac7a4ce1b25bb8e19ca'}, {'split': 'validation', 'scene': 150, 'sha_match': True, 'regen_sha': 'a1dd3f3c0e2d8de6cef139a94a1fdfd18057bdd1d9db2a7cacfc0ef5a1fe8ee6', 'manifest_sha': 'a1dd3f3c0e2d8de6cef139a94a1fdfd18057bdd1d9db2a7cacfc0ef5a1fe8ee6'}, {'split': 'validation', 'scene': 299, 'sha_match': True, 'regen_sha': '36e89ccb6a9c4d381d27a1c16fc3bc1ff8eb95870300ec468fc533eec8d43839', 'manifest_sha': '36e89ccb6a9c4d381d27a1c16fc3bc1ff8eb95870300ec468fc533eec8d43839'}, {'split': 'test', 'scene': 0, 'sha_match': True, 'regen_sha': '268df03cb9edded383345dc3f0566b81c272365e7be9e15e09525b650f014ccb', 'manifest_sha': '268df03cb9edded383345dc3f0566b81c272365e7be9e15e09525b650f014ccb'}, {'split': 'test', 'scene': 450, 'sha_match': True, 'regen_sha': 'd7a5a4f3b8eaeb6848de70f3714ae461afee1c11dfda2a6faec5c0047d243a99', 'manifest_sha': 'd7a5a4f3b8eaeb6848de70f3714ae461afee1c11dfda2a6faec5c0047d243a99'}, {'split': 'test', 'scene': 899, 'sha_match': True, 'regen_sha': 'b39990e0f8d723b4ab71720401720be5a6cc24bc0d986ec982c8f7154151b516', 'manifest_sha': 'b39990e0f8d723b4ab71720401720be5a6cc24bc0d986ec982c8f7154151b516'}], 'ok': True} |
| loader_allowlist | {'allowed': ['augment', 'image', 'view_id'], 'actual': ['augment', 'image', 'view_id'], 'forbidden_present': [], 'ok': True} |

明细见 `gate1-audit.json`。
