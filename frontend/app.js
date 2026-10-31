/* Dashboard. Renders ONLY what the backend sends (events + results). All dynamic text uses
   textContent / DOM nodes, never innerHTML, because messages, memories and model output are untrusted. */
"use strict";

const STAGES = [
  ["perception", "Perception"], ["memory", "Memory"], ["attention", "Attention"], ["workspace", "Workspace"],
  ["planning", "Planning"], ["evaluation", "Evaluation"], ["selection", "Selection"], ["tool", "Tool"],
  ["language", "Language"],
];
const $ = (sel) => document.querySelector(sel);
function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return n;
}
const tag = (text, cls) => el("span", { class: "tag " + (cls || ""), text });
const fmt = (x) => (typeof x === "number" ? x.toFixed(2) : String(x));

let sid = sessionStorage.getItem("brain_sid");
if (!sid) { sid = "s_" + Math.random().toString(36).slice(2, 12); sessionStorage.setItem("brain_sid", sid); }

const S = { ws: null, config: null, memoryEnabled: false, running: false, cycleStart: {}, proposals: [], chosen: null,
            retrieved: [], dropped: {}, timer: null };

// ------------------------------------------------------------------ auth (Supabase email + password)
// Talks to the project's GoTrue REST API directly; the backend only ever sees the resulting access token.
// Tokens live in localStorage (same-origin only). The server verifies them against the project's JWKS.
const Auth = {
  cfg: null, session: null,
  get required() { return !!(this.cfg && this.cfg.mode === "supabase"); },
  init(cfg) {
    this.cfg = cfg;
    try { this.session = JSON.parse(localStorage.getItem("brain_auth") || "null"); } catch (e) { this.session = null; }
  },
  save(sess) {
    this.session = sess ? { access_token: sess.access_token, refresh_token: sess.refresh_token,
                            expires_at: Math.floor(Date.now() / 1000) + (sess.expires_in || 3600), email: sess.user && sess.user.email } : null;
    if (this.session) localStorage.setItem("brain_auth", JSON.stringify(this.session)); else localStorage.removeItem("brain_auth");
  },
  async gotrue(path, body, token) {
    const headers = { "Content-Type": "application/json", apikey: this.cfg.publishable_key };
    if (token) headers.Authorization = "Bearer " + token;
    const res = await fetch(this.cfg.supabase_url + "/auth/v1/" + path, { method: "POST", headers, body: JSON.stringify(body || {}) });
    const data = res.status === 204 ? {} : await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.msg || data.error_description || data.message || ("auth error " + res.status));
    return data;
  },
  signIn(email, password) { return this.gotrue("token?grant_type=password", { email, password }).then((d) => { this.save(d); return d; }); },
  signUp(email, password) { return this.gotrue("signup", { email, password }); },
  async signOut() {
    const t = this.session && this.session.access_token; this.save(null);
    if (t) await this.gotrue("logout", {}, t).catch(() => {});
  },
  async token() {
    if (!this.required) return null;
    if (!this.session) return null;
    if (this.session.expires_at - 60 > Date.now() / 1000) return this.session.access_token;
    try { this.save(await this.gotrue("token?grant_type=refresh_token", { refresh_token: this.session.refresh_token })); return this.session.access_token; }
    catch (e) { this.save(null); return null; }
  },
};

function showAuth(msg, ok) {
  document.body.classList.add("locked");
  $("#auth-panel").hidden = false; $("#user-chip").hidden = true; $("#sign-out").hidden = true;
  const m = $("#auth-msg"); m.textContent = msg || ""; m.className = ok ? "ok" : "err";
  setConn("signed out", "bad");
}
function hideAuth() {
  document.body.classList.remove("locked");
  $("#auth-panel").hidden = true;
  if (Auth.required) {
    $("#user-chip").textContent = Auth.session && Auth.session.email ? Auth.session.email : "signed in";
    $("#user-chip").hidden = false; $("#sign-out").hidden = false;
  }
}

// ------------------------------------------------------------------ rail
function resetRail() {
  const rail = $("#rail"); rail.replaceChildren();
  for (const [id, label] of STAGES) rail.append(el("li", { "data-stage": id }, el("b", { text: label }), el("small", { text: "" })));
}
function lightStage(id, note) {
  const li = document.querySelector(`#rail li[data-stage="${id}"]`);
  if (li) { li.classList.add("on"); if (note) li.querySelector("small").textContent = note; }
}

// ------------------------------------------------------------------ connection
async function connect() {
  const token = await Auth.token();
  if (Auth.required && !token) { showAuth(); return; }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${sid}`);
  S.ws = ws;
  ws.onopen = () => { if (token) ws.send(JSON.stringify({ type: "auth", token })); setConn("connected", "ok"); };
  ws.onclose = (e) => {
    setRunning(false);
    if (e.code === 4401) { Auth.save(null); showAuth("Your session is no longer valid. Sign in again."); return; }
    setConn("disconnected, retrying…", "bad"); setTimeout(connect, 2000);
  };
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "hello") onHello(msg);
    else if (msg.type === "event") onEvent(msg.event);
    else if (msg.type === "result") onResult(msg.result);
    else if (msg.type === "error") $("#form-error").textContent = msg.message;
  };
}
function setConn(text, cls) { const c = $("#conn-chip"); c.textContent = text; c.className = "chip " + cls; }
function setRunning(v) { S.running = v; $("#cancel").disabled = !v; $("#send").textContent = v ? "Send (replaces current)" : "Send"; }

function onHello(msg) {
  S.config = msg.config; S.memoryEnabled = msg.state.memory_enabled;
  hideAuth();
  $("#memory-toggle").checked = S.memoryEnabled;
  const mock = msg.config.model.is_mock;
  $("#mock-banner").hidden = !mock;
  const chip = $("#model-chip"); chip.textContent = "model: " + msg.config.model.name; chip.className = "chip" + (mock ? " mock" : "");
  $("#calls-chip").textContent = "calls 0/" + msg.config.settings.max_model_calls;
  renderWorkspace(msg.state.workspace);
  loadMemories();
}

// ------------------------------------------------------------------ events
function onEvent(ev) {
  const cid = ev.cycle_id;
  if (!(cid in S.cycleStart)) S.cycleStart[cid] = Date.parse(ev.timestamp);
  const rel = Date.parse(ev.timestamp) - S.cycleStart[cid];
  const p = ev.payload;
  const t = ev.event_type;
  if (t === "input.received") { newCycleUI(); setRunning(true); }
  appendTimeline(ev, rel);
  switch (t) {
    case "component.activated": lightStage(p.component, rel + " ms"); break;
    case "perception.done": renderObservation(p); break;
    case "memory.retrieved":
      S.retrieved = p.records; renderRetrieved(); lightStage("memory", p.skipped_reason ? "skipped" : p.records.length + " found");
      if (p.skipped_reason) document.querySelector('#rail li[data-stage="memory"]').classList.add("skipped"); break;
    case "attention.done": S.dropped = Object.fromEntries(p.dropped.map((d) => [d.id, d.reason])); renderRetrieved(); break;
    case "workspace.updated": renderWorkspace(p.items, p.size, p.max_items); break;
    case "model.call": $("#calls-chip").textContent = `calls ${p.calls_used}/${p.calls_limit}`; break;
    case "planning.done": S.proposals = p.proposals.map((x) => ({ ...x, eval: null })); renderProposals(p); break;
    case "evaluation.done": for (const e of p.evaluations) { const x = S.proposals.find((q) => q.proposal_id === e.proposal_id); if (x) x.eval = e; } renderProposals(); break;
    case "action.selected": S.chosen = p.proposal_id; renderProposals(); renderSelected(p); break;
    case "memory.write": renderMemoryWrite(p); if (p.status === "saved") loadMemories(); break;
    case "feedback.recorded": break;
    case "cycle.cancelled": case "cycle.timeout": case "cycle.error": case "limit.reached":
      if (t !== "limit.reached") setRunning(false); break;
  }
}

function newCycleUI() {
  resetRail(); S.proposals = []; S.chosen = null; S.retrieved = []; S.dropped = {};
  $("#timeline").replaceChildren();
  for (const id of ["#selected", "#proposals", "#retrieved"]) $(id).replaceChildren(el("p", { class: "empty", text: "Working…" }));
  $("#observation").replaceChildren();
}

function appendTimeline(ev, rel) {
  const alert = ["cycle.cancelled", "cycle.timeout", "cycle.error", "limit.reached"].includes(ev.event_type);
  const what = summarise(ev);
  const li = el("li", { class: alert ? "alert" : "", title: `event ${ev.event_id}` + (ev.parent_event_id ? ` ← ${ev.parent_event_id}` : "") },
    el("span", { class: "t", text: `+${rel}ms` }), el("span", { class: "src", text: ev.source }),
    el("span", { class: "what", text: what }));
  const tl = $("#timeline");
  tl.append(li);
  while (tl.children.length > 300) tl.firstChild.remove();
  tl.scrollTop = tl.scrollHeight;
}
function summarise(ev) {
  const p = ev.payload, t = ev.event_type;
  switch (t) {
    case "component.activated": return `activate ${p.component}: ${p.reason}`;
    case "perception.done": return `intent=${p.intent}, ${p.constraints.length} constraint(s), ${p.entities.length} entit(ies)` + (p.underspecified ? ", missing info" : "");
    case "memory.retrieved": return p.skipped_reason ? `skipped (${p.skipped_reason})` : `${p.candidates} candidate(s)`;
    case "attention.done": return `kept ${p.kept.length}, dropped ${p.dropped.length}`;
    case "workspace.updated": return `${p.size}/${p.max_items} items` + (p.evicted.length ? `, evicted ${p.evicted.length}` : "");
    case "model.call": return `${p.task} attempt ${p.attempt}: ${p.ok ? "ok" : "FAILED " + p.error_code} (${p.duration_ms} ms)`;
    case "planning.done": return `${p.proposals.length} proposal(s) via ${p.source}` + (p.rejected.length ? `, ${p.rejected.length} rejected` : "");
    case "evaluation.done": return `${p.evaluations.length} scored`;
    case "action.selected": return `${p.kind} by ${p.rule}`;
    case "memory.write": return `${p.status}` + (p.reason ? ` (${p.reason})` : "");
    case "tool.started": return `${p.tool} ${p.url}`;
    case "tool.done": return p.ok ? `${p.tool} ok: ${p.final_url} (${p.chars} chars${p.truncated || p.context_truncated ? ", truncated" : ""})` : `${p.tool} FAILED ${p.error.code}: ${p.error.message}`;
    case "language.done": return `rendered by ${p.method}` + (p.post_checks.length ? `, checks: ${p.post_checks.join(", ")}` : "");
    case "response.final": return `final ${p.action_kind} reply`;
    case "cycle.cancelled": return `cancelled: ${p.reason}`;
    case "cycle.timeout": return `time limit ${p.limit_s}s reached`;
    case "cycle.error": return p.message;
    case "limit.reached": return `limit ${p.limit} (${p.value}) reached`;
    case "feedback.recorded": return `${p.rating}: bias ${p.bias_before} → ${p.bias_after}`;
    default: return t;
  }
}

// ------------------------------------------------------------------ panels
function renderObservation(o) {
  const box = $("#observation"); box.replaceChildren();
  box.append(el("div", {}, tag("intent: " + o.intent), ...o.constraints.map((c) => tag(`${c.type}=${c.value}`)),
    ...o.entities.map((e) => tag(e.text)), o.underspecified ? tag("missing: " + o.missing.join("; "), "warn") : null));
}
function renderWorkspace(items, size, max) {
  $("#ws-count").textContent = size != null ? `${size}/${max} items` : (items.length + " items");
  const box = $("#workspace");
  if (!items.length) { box.replaceChildren(el("p", { class: "empty", text: "Empty." })); return; }
  box.replaceChildren(el("ul", { class: "ws" }, ...items.slice(-30).map((i) =>
    el("li", {}, tag(i.kind), el("span", { class: "origin", text: `${i.source} · ` }), i.preview))));
}
function renderRetrieved() {
  const box = $("#retrieved");
  if (!S.retrieved.length) { box.replaceChildren(el("p", { class: "empty", text: S.config && !S.memoryEnabled ? "Memory is off: nothing was read." : "No relevant memories found." })); return; }
  box.replaceChildren(...S.retrieved.map((m) => {
    const why = S.dropped[m.id];
    return el("div", { class: "prop" }, el("div", { text: m.text }),
      el("div", { class: "why" }, tag(m.origin), tag("score " + fmt(m.score)), tag(m.trust, m.origin === "model_interpretation" ? "warn" : ""),
        why ? tag("not used: " + why, "bad") : tag("sent to components", "ok")));
  }));
}
function renderProposals(planning) {
  const box = $("#proposals"); box.replaceChildren();
  if (!S.proposals.length) { box.append(el("p", { class: "empty", text: "No proposals." })); return; }
  for (const p of S.proposals) {
    const e = p.eval;
    const row = el("div", { class: "prop" + (p.proposal_id === S.chosen ? " chosen" : "") + (e && e.blocked ? " blocked" : "") },
      el("div", {}, tag(p.kind, p.proposal_id === S.chosen ? "sel" : ""), el("strong", { text: p.purpose })),
      el("div", { class: "why", text: `needs: ${p.required_inputs.join(", ") || "nothing"} → ${p.expected_result}` }));
    if (p.origin === "model") row.append(el("div", { class: "why", text: `model self-reported confidence ${fmt(p.model_confidence)} (uncalibrated; only a last tie-breaker)` }));
    if (e) {
      const bars = el("div", { class: "bars" });
      for (const k of ["relevance", "constraint_consistency", "support", "safety"]) {
        bars.append(el("span", { text: k.replace("_", " "), title: S.config ? S.config.criteria[k] : "" }),
          el("span", { class: "bar" }, el("i", { style: `width:${Math.round(e.scores[k] * 100)}%` })), el("span", { text: fmt(e.scores[k]) }));
      }
      row.append(bars, el("div", { class: "why" }, `total ${fmt(e.total)} · `, ...e.flags.map((f) => tag(f, e.blocked ? "bad" : "warn")),
        e.blocked ? tag("blocked", "bad") : null), el("div", { class: "why", text: e.summary }));
      if (e.model_review) row.append(el("div", { class: "why", text: `same-model review (not independent verification): ${e.model_review.note}` }));
    }
    box.append(row);
  }
  if (planning) {
    const notes = [...(planning.notes || []), ...(planning.rejected || []).map((r) => "rejected: " + r), planning.model_error ? "model error: " + planning.model_error : null].filter(Boolean);
    for (const n of notes) box.append(el("div", { class: "toast-note", text: n }));
  }
}
function renderSelected(sel) {
  $("#selected").replaceChildren(
    el("div", {}, tag(sel.kind, "sel"), tag(sel.rule), sel.proposal_id ? tag(sel.proposal_id) : tag("synthesised clarification", "warn")),
    el("p", { text: sel.reason }),
    el("div", { class: "why" }, ...sel.ranking.map((r) => tag(`${r.proposal_id} ${r.kind}: ${fmt(r.eval_total)} ${r.bias >= 0 ? "+" : ""}${fmt(r.bias)} = ${fmt(r.adjusted)}${r.blocked ? " (blocked)" : ""}`))));
}
function renderMemoryWrite(mw) {
  const box = $("#selected");
  const text = mw.status === "saved" ? `Saved memory ${mw.record.id} (${mw.record.kind}, origin ${mw.record.origin}).`
    : `Memory write: ${mw.status}${mw.reason ? " (" + mw.reason + ")" : ""}.`;
  box.append(el("div", { class: "toast-note", text }));
}

// ------------------------------------------------------------------ chat
function addMessage(role, text, extra) {
  const box = $("#messages");
  const first = box.querySelector(".empty"); if (first) first.remove();
  const m = el("div", { class: "msg " + role }, el("div", { text }));
  if (extra) m.append(extra);
  box.append(m); box.scrollTop = box.scrollHeight;
  return m;
}
function onResult(r) {
  setRunning(false);
  if (r.status === "cancelled") { addMessage("notice", `Request cancelled (${r.reason}). No reply was published for it.`); return; }
  if (!r.final) { addMessage("notice", `Cycle ended with status ${r.status}: ${r.reason || ""}`); return; }
  const meta = el("div", { class: "meta" });
  meta.append(tag(r.final.action_kind, "sel"));
  if (r.final.mock) meta.append(tag("mock model", "warn"));
  meta.append(tag(`rendered by ${r.final.method}`), tag(`${r.model_calls_used} model call(s)`));
  if (r.status !== "completed") meta.append(tag(r.status, "bad"));
  const rate = (rating) => el("button", { class: "ghost", onclick: async (ev) => {
    const res = await api("POST", `/api/sessions/${sid}/feedback`, { cycle_id: r.cycle_id, rating });
    for (const b of meta.querySelectorAll("button.fb")) b.remove();
    meta.append(tag(res.error ? res.error : `bias ${res.bias_before} → ${res.bias_after}` + (res.persisted ? " (saved)" : " (session only)"), res.error ? "bad" : "ok"));
  }, class: "ghost fb", text: rating === "useful" ? "Useful" : "Not useful" });
  meta.append(rate("useful"), rate("not_useful"));
  if (S.memoryEnabled) meta.append(el("button", { class: "link", text: "Save summary", onclick: async (ev) => {
    const res = await api("POST", `/api/sessions/${sid}/cycles/${r.cycle_id}/save-summary`);
    ev.target.replaceWith(tag(res.error ? res.error : "summary saved", res.error ? "bad" : "ok")); loadMemories();
  } }));
  addMessage("system", r.final.text, meta);
}

async function api(method, path, body) {
  try {
    const headers = body ? { "Content-Type": "application/json" } : {};
    const token = await Auth.token();
    if (token) headers.Authorization = "Bearer " + token;
    const res = await fetch(path, { method, headers, body: body ? JSON.stringify(body) : undefined });
    if (res.status === 401 && Auth.required) { Auth.save(null); showAuth("Signed out: " + ((await res.json().catch(() => ({}))).error || "sign in again")); }
    return await res.json();
  } catch (e) { return { error: "request failed" }; }
}

// ------------------------------------------------------------------ memory manager
async function loadMemories() {
  const data = await api("GET", "/api/memories");
  const list = $("#memories");
  const mems = data.memories || [];
  $("#mem-count").textContent = mems.length + " saved";
  if (!mems.length) { list.replaceChildren(el("li", { class: "empty", text: "No saved memories." })); return; }
  list.replaceChildren(...mems.map((m) => el("li", {},
    el("div", {}, el("div", { text: m.text }), el("div", { class: "origin", text: `${m.kind} · ${m.origin} · ${m.trust} · ${m.created_at.slice(0, 16).replace("T", " ")}` })),
    el("button", { class: "ghost", text: "Delete", "aria-label": "Delete memory", onclick: async () => { await api("DELETE", "/api/memories/" + m.id); loadMemories(); } }))));
}

// ------------------------------------------------------------------ wiring
$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#input").value.trim();
  $("#form-error").textContent = "";
  if (!text) { $("#form-error").textContent = "Type a message first."; return; }
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) { $("#form-error").textContent = "Not connected yet."; return; }
  addMessage("user", text);
  S.ws.send(JSON.stringify({ type: "message", text }));
  $("#input").value = "";
});
$("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); } });
$("#cancel").addEventListener("click", () => S.ws && S.ws.send(JSON.stringify({ type: "cancel" })));
$("#memory-toggle").addEventListener("change", async (e) => {
  const res = await api("PUT", `/api/sessions/${sid}/settings`, { memory_enabled: e.target.checked });
  S.memoryEnabled = !!res.memory_enabled; e.target.checked = S.memoryEnabled;
});
$("#new-session").addEventListener("click", () => { sessionStorage.removeItem("brain_sid"); location.reload(); });
$("#auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#auth-signin").disabled = true;
  try { await Auth.signIn($("#auth-email").value.trim(), $("#auth-password").value); $("#auth-msg").textContent = ""; $("#auth-password").value = ""; connect(); }
  catch (err) { showAuth(err.message); }
  finally { $("#auth-signin").disabled = false; }
});
$("#auth-signup").addEventListener("click", async () => {
  const email = $("#auth-email").value.trim(), password = $("#auth-password").value;
  if (!email || password.length < 6) { showAuth("Enter an email and a password of at least 6 characters."); return; }
  $("#auth-signup").disabled = true;
  try {
    const d = await Auth.signUp(email, password);
    if (d.access_token) { Auth.save(d); connect(); }
    else showAuth("Account created. Check " + email + " for a confirmation link, then sign in.", true);
  } catch (err) { showAuth(err.message); }
  finally { $("#auth-signup").disabled = false; }
});
$("#sign-out").addEventListener("click", async () => {
  if (S.ws) { S.ws.onclose = null; S.ws.close(); S.ws = null; }
  await Auth.signOut(); sessionStorage.removeItem("brain_sid"); location.reload();
});
$("#mem-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#mem-text").value.trim(); if (!text) return;
  const res = await api("POST", "/api/memories", { text, kind: $("#mem-kind").value });
  if (res.error) $("#form-error").textContent = res.error; else $("#mem-text").value = "";
  loadMemories();
});

async function boot() {
  resetRail();
  let cfg = null;
  try { cfg = await (await fetch("/api/config")).json(); } catch (e) { setConn("server unreachable, retrying…", "bad"); setTimeout(boot, 3000); return; }
  Auth.init(cfg.auth || { mode: "off" });
  const notice = consumeAuthFragment();
  if (notice) { showAuth(notice.message, notice.ok); if (!notice.ok) return; }
  connect();
}

// Supabase email links (confirmation, recovery, magic link) redirect back to the Site URL with the
// outcome in the URL fragment: either a session (#access_token=...&refresh_token=...) or an error
// (#error=...&error_description=...). Fragments never reach the server; handle them here, then
// scrub the URL so tokens do not linger in history.
function consumeAuthFragment() {
  if (!location.hash || location.hash.length < 2) return null;
  const h = new URLSearchParams(location.hash.slice(1));
  if (!h.has("access_token") && !h.has("error")) return null;
  history.replaceState(null, "", location.pathname + location.search);
  if (h.get("error")) {
    const code = h.get("error_code") || h.get("error");
    const msg = code === "otp_expired"
      ? "That email link was already used or has expired. If you already clicked it once, your account is confirmed: just sign in."
      : (h.get("error_description") || code).replace(/\+/g, " ");
    return { ok: false, message: msg };
  }
  if (!Auth.required) return null;
  let email = null;
  try { email = JSON.parse(atob(h.get("access_token").split(".")[1].replace(/-/g, "+").replace(/_/g, "/"))).email || null; } catch (e) { /* cosmetic only */ }
  Auth.save({ access_token: h.get("access_token"), refresh_token: h.get("refresh_token"),
              expires_in: Number(h.get("expires_in")) || 3600, user: { email } });
  return { ok: true, message: h.get("type") === "signup" ? "Email confirmed. Signing you in…" : "Signing you in…" };
}
boot();