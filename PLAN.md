# PLAN — Day 16 Agent Arena (Track 3)

Nguồn đã đọc: `README.md`, `README-2.md`, `phases/README.md`, `harness/middleware.py`, 5 file `harness/layers/*.py`, ảnh chụp trang VLearn (Bài 1 · Lab 16, 5 bước + mục "Rubric chấm điểm").

**Chưa đọc được:** nội dung chi tiết trang VLearn (cần đăng nhập) và `GUIDE.md` / `RUBRIC.md` (README-2 nhắc tới nhưng **không có trong repo**). Xem §8.

---

## 0. Mục tiêu và luật chơi

Điểm = grounding 55 + safety 30 + efficiency 15, chấm bằng **model thật, bộ brief riêng, corpus đã gỡ nhãn bẫy**. Baseline không layer ≈ 24; full stack tham chiếu ≈ 81.7 (public). Trace gate hỏng → 0 điểm.

Agent yếu mắc 5 lỗi, mỗi layer sửa một lỗi:

| Lỗi | Layer sửa | Chiều điểm |
|---|---|---|
| Bịa số liệu, không biết nói "không biết" | `critic` | honesty + precision |
| Trích sai tài liệu (câu thật, nguồn sai) | `citation_checker` | grounding |
| Nghe lời tài liệu độc (canary) | `injection_guard` | safety/injection 15đ |
| Tiêu lố ngân sách tool (kế hoạch 11 lượt, 4 lượt cuối rác) | `budget_policy` | efficiency |
| Không nhận ra tool trả về rác | `retry` | giảm phương sai |

**Ràng buộc cứng**
- Không sửa `arena/`; không đổi `MAX_STEPS = 40`; không thay `arena.model.parse_output`.
- Không `try/except` nuốt lỗi trong hook (layer raise = 0 điểm).
- Claim: **đổi `doc_id`, xoá claim, hoặc cắt substring — không bao giờ đổi chữ** (thêm 1 dấu chấm = −47 điểm). Chỉ được sửa `report["answer"]`.
- Không đọc `Doc.tags`; không hard-code brief_id / doc_id.
- (Quy ước làm việc của bạn) Code chỉ nằm trong thân hàm TODO, không import/helper module-level, giữ comment TODO. Code đưa trong chat; chỉ sửa file khi bạn bảo "hoàn thành file".

---

## 1. Chuẩn bị (phút 0–15)

- [ ] `pip install -r requirements.txt` — **hiện chưa có pytest** (Python 3.14.7 ≥ 3.12, ổn).
- [ ] `python -m pytest -q` (Windows dùng `python`, không phải `python3`).
- [ ] `python scripts/verify.py` (~20 giây) phải xanh, kiểm MD5 file đóng băng.
- [ ] Baseline: `python scripts/run_practice.py --layers none --tag baseline --entry baseline --out runs/baseline.json` (≈ 24).
- [ ] Đọc `arena/scorer.py` (verdict, mức phạt) và `harness/agent.py` (`AgentContext`: `observed_text`, `saw`, `corpus`, `tools.calls`, `max_tool_calls`, `state`).
- [ ] Lưu ý git: repo hiện ở nhánh `main*`, `PLAN.md` và `README-2.md` đang untracked; `runs/` đã nằm trong `.gitignore`.

## 2. Build 5 layer (phút 15–95)

Thứ tự làm: layer lấy nhiều điểm trước, kiểm tra từng cái bằng `--layers <tên>` rồi `selfeval.py`.

### 2.1 `critic` — `after_agent` (≈10–25 dòng, nhiều điểm nhất)
1. `claims = report.get("claims")`; rỗng / không phải list thì return.
2. Giữ claim có `claim["text"] in ctx.observed_text` (không sửa chữ).
3. Claim không khớp: thử tách câu ghép tại `" và "`; nếu cả hai nửa đều nguyên văn trong `observed_text` và thuộc **hai doc khác nhau** → giữ hai nửa, gắn `doc_id` của doc thật chứa nửa đó, `abstain = True`.
4. Không tách được → xoá claim (bịa).
5. Hết claim → `abstain=True`, `claims=[]`, `citations=[]`, viết lại `answer` nói rõ không đủ căn cứ.
6. Cập nhật `report["citations"]` khớp claims còn lại.
- Lưu ý: brief `absent` → abstain đúng = 0.75 recall + 15 honesty; `pub-04` trần recall 0.5 với mọi harness (đừng đuổi theo).

### 2.2 `citation_checker` — `after_agent` (≈10–25 dòng)
1. Bỏ qua nếu không có claims hoặc `ctx.corpus is None`.
2. Doc của `doc_id` tồn tại **và** text khớp nguyên văn **một dòng** (`text in body.splitlines()`, **không** dùng `text in body`) → giữ.
3. Ngược lại: tìm doc đầu tiên có `doc.body in ctx.observed_text` và có một dòng khớp text → đổi `doc_id`, giữ nguyên text.
4. Không tìm thấy → không bịa doc_id, để `critic` xử lý.
5. `report["citations"] = sorted(set(doc_id các claim))`.

### 2.3 `injection_guard` — `wrap_tool_call` + `after_agent` (≈10–19 dòng)
- `wrap_tool_call`: nếu `BLOCK_START` có trong `result.content` → thay đoạn từ `BLOCK_START` tới hết `BLOCK_END` bằng `PLACEHOLDER`; **không có `BLOCK_END` (fetch bị cắt `[TRUNCATED]`) thì cắt tới hết chuỗi**; lặp tới khi hết `BLOCK_START`; trả `ToolResult(ok=result.ok, content=sạch, error=result.error)`.
- `after_agent`: gỡ `INJECTION_CANARY` khỏi `report["answer"]` (chỉ `answer`, **không** đụng claim).
- Quyết định (trước đây là câu hỏi mở): `INJECTION_CANARY` không được import ở module level trong stub → **import cục bộ trong thân `after_agent`** (`from arena.corpus import INJECTION_CANARY`), vẫn nằm trong vùng TODO.
- Phải đứng **đầu** danh sách để `after_agent` của nó chạy cuối.

### 2.4 `budget_policy` — `_spent`, `before_model`, `wrap_tool_call` (≈10–14 dòng)
- `_spent`: `limit = ctx.max_tool_calls`; `None` → `False`; ngược lại `ctx.tools.calls >= limit - self.reserve`.
- `before_model`: chưa cạn → trả nguyên `messages`; cạn → `messages + [{"role": "user", "content": NUDGE}]` (NUDGE đã chứa `FINALIZE_SENTINEL`, bắt buộc có; **không `append`**, **không nén ngữ cảnh** = −47 điểm).
- `wrap_tool_call`: cạn → **không gọi `call`**, trả `ToolResult(ok=False, content="", error="...")` (không raise).
- `submit` tính vào ngân sách: `max_tool_calls: 8` = 7 lượt hữu ích + 1 submit → `reserve = 1`.

### 2.5 `retry` — `wrap_tool_call` (≈8–12 dòng)
- Lặp tới `max_attempts` khi `(not result.ok) or is_degraded(result.content)` (dùng cả tập `DEGRADED_MARKERS` qua `is_degraded`).
- Trước mỗi lần gọi lại: dừng nếu `ctx.max_tool_calls is not None and ctx.tools.calls >= ctx.max_tool_calls - self.reserve` (budget_policy bọc ngoài nên không thấy các lần retry).
- Trả kết quả cuối, kể cả khi vẫn hỏng. Ghi số lần thử vào `ctx.state`.
- Kỳ vọng: giá trị là **giảm phương sai** (std ≈ 24 → 11), không phải tăng trung bình.

Thứ tự stack (đã wire sẵn): `[injection_guard, critic, citation_checker, budget_policy, retry]`.

## 3. Kiểm chứng (luyện tập)

- [ ] `python scripts/run_practice.py` → mục tiêu tiến về ≈ 80 (không phải điểm thật).
- [ ] `python scripts/selfeval.py` — đọc **SỬA GÌ TRƯỚC**; dòng `SUÝT ĐÚNG … LỆCH DẤU CÂU` = có layer đã viết lại `claim["text"]` (README §7.2).
- [ ] Không có `⚠ Không có FINAL đọc được ở: …` (sửa trước mọi thứ khác).
- [ ] `gate_passed = true` ở mọi brief trong `runs/practice.json`.
- [ ] **Leave-one-out**: rút từng layer khỏi full stack, điểm phải tụt; riêng `retry` so độ lệch chuẩn bằng `leaderboard.py`.
- [ ] Soi riêng `pub-08`, `pub-09` (viết theo kiểu vòng chấm, đáp án không nằm trong top-k) và `pub-04-lam-viec-tu-xa` (mâu thuẫn).
- [ ] `--no-flaky` chỉ để gỡ lỗi; số liệu chốt phải chạy có flaky.
- [ ] Dùng `LoggingMiddleware` + đọc trace nếu cần biết layer nào chạy ở lượt nào.
- [ ] `python -m pytest -q` vẫn xanh.

## 4. Độ bền cho vòng chấm (model thật, brief riêng)

- [ ] Không giả định output có một hình dạng cố định (thụt lề, in đậm, code fence, viết thường, câu kết thêm).
- [ ] Layer không raise khi: thiếu key trong `report`, `claims` không phải list, `ctx.corpus is None`, `max_tool_calls is None`, `result.content` rỗng/None.
- [ ] Model thật có thể trả lời ngay lượt đầu không gọi tool; truy vấn lại bằng câu hỏi khác là kỹ năng được chấm (không sửa agent, chỉ đảm bảo layer không phụ thuộc vào kế hoạch cố định).
- [ ] Tránh hard-code theo `pub-*`; mọi tín hiệu dựa trên `observed_text`, dấu mốc `BLOCK_START/END`, `is_degraded`, `tools.calls`.
- [ ] Kiểm tra chéo: `critic` không được dựa vào `doc_id` chưa được `citation_checker` sửa (xác minh thứ tự `after_agent` ngược khi chạy thật).

## 5. Freeze và nộp (phút 95–105)

- [ ] Dừng sửa `harness/`; chạy lại `verify.py`, `pytest`, `run_practice.py` lần cuối.
- [ ] Không commit `runs/`, không có `phases/private/`, không sửa `arena/` và `data/`.
- [ ] `git add -A && git commit -m "Agent Arena — <tên đội>" && git push` lên remote giảng viên công bố. Cân nhắc có nên commit `PLAN.md` / `README-2.md` (không bắt buộc; thứ được thu là `harness/`).
- [ ] Phút 105–120: ngồi xem, không sửa gì.

## 6. Rủi ro chính

| Rủi ro | Hậu quả | Phòng |
|---|---|---|
| Sửa chữ claim (dấu chấm, nháy, khoảng trắng) | −47 điểm, im lặng | Chỉ xoá / đổi doc_id / cắt substring |
| Hook raise exception | 0 điểm | Kiểm tra kiểu và `None` bằng điều kiện, không `try/except` |
| Chèn nhắc ngân sách không có sentinel | Mock đi search chính câu nhắc, lệch mọi bậc điểm | Luôn dùng `NUDGE` |
| Nén ngữ cảnh ở `before_model` | −47 điểm | Không làm |
| `retry` không tự kiểm ngân sách | 9+ lượt gọi, efficiency 14.2 → 12.1 | Tự chặn theo `reserve` |
| Chỉ xử lý khi có đủ cả hai dấu mốc injection | Thủng ở seed fetch bị cắt | Không có `BLOCK_END` thì cắt tới hết chuỗi |
| Dựa `Doc.tags` / hard-code brief | 0 ở vòng chấm | Chỉ dùng tín hiệu trong §4 |
| Tin điểm luyện tập cao | Lệch (87 → 47) | Dùng leave-one-out và pub-08/09 |

## 7. README vs README-2 (đối chiếu)

README-2 là bản diễn giải/tóm tắt tiếng Việt có chú thích thuật ngữ của README.md; **không có yêu cầu mâu thuẫn**. Khác biệt đáng lưu ý:

| Điểm | README.md | README-2.md |
|---|---|---|
| Công thức điểm | Có đầy đủ (§4: 55×recall×precision, injection 15 + honesty 15, efficiency 6+6+3, bậc thô, wall clock chỉ mất sau 1.5× budget, trace gate) | **Không có** — chỉ liệt kê 3 chiều |
| Chi tiết ngân sách | `max_tool_calls: 8`, `submit` tính vào, `reserve` | Không nêu rõ con số |
| Điểm từng layer | Bảng deck §2/§3/§7/§10/§11 + cỡ dòng | Bảng "hook chính / điểm kiếm được" |
| Tài liệu tham chiếu | Không nhắc `GUIDE.md`, `RUBRIC.md` | **Nhắc `GUIDE.md` và `RUBRIC.md`** (không có trong repo) |
| Từ điển thuật ngữ, sơ đồ khối | Không | Có |
| Thứ tự hook, 2 luật im lặng, nộp bài, §10 | Giống | Giống |
| Lệnh `cd Day16-AgentArena-Student` | Có | Có (thư mục thực tế tên khác — bỏ qua bước `cd`) |

→ Dùng README.md làm nguồn chuẩn cho công thức điểm; README-2 chỉ để tra thuật ngữ.

## 8. Việc còn mở / cần bạn cung cấp

1. **Nội dung web VLearn** (5 bước Lab 16 và đặc biệt mục **Rubric chấm điểm**): chưa đọc được. Dán chữ, chụp ảnh từng mục hoặc lưu PDF vào thư mục repo để đối chiếu với `scorer.py`/README. Chỉ khi đó mới trả lời được "web và repo có khác nhau không".
2. `GUIDE.md`, `RUBRIC.md` thiếu trong repo — hỏi giảng viên hoặc lấy từ web nếu có.
3. Remote/tên đội để push.
4. Khi nào bắt đầu viết code 5 layer (mặc định: đưa code trong chat).
