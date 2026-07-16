# lab1：验证代码能不能跑通

# lab2：baseline + factorial: 调研具体的性能问题

# lab3：Diagnostic

# lab4：Hypothesis generation

# lab5：Pairwise Factorial

## v2 正确性约束

- `dn_cutoff` 必须在所有 token-bank 下于 softmax 前真实 mask logits；
- faithful 主实验固定 `forward-gate-mode=zero`，adaptive 只作为诊断变体；
- reference diagnostic 同时记录 token 数与 64×64 grid 前景比例；
- 修复前输出不得用 `RESUME=1` 追加到 v2 结果中。
