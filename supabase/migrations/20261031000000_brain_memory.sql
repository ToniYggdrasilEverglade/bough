-- Bough memory backend. One row set per signed-in user; Postgres RLS enforces ownership so the
-- server never needs a database password or service key - it forwards the user's own JWT.

create table public.memories (
  id              text primary key,
  user_id         uuid not null default auth.uid() references auth.users(id) on delete cascade,
  kind            text not null check (kind in ('fact', 'preference', 'summary')),
  text            text not null check (char_length(text) between 1 and 500),
  text_lower      text generated always as (lower(text)) stored,
  origin          text not null check (origin in ('user_stated', 'system_summary', 'model_interpretation')),
  created_at      timestamptz not null default now(),
  source_session  text,
  source_cycle    text,
  tokens          text not null default ''
);
create index memories_user_created on public.memories (user_id, created_at desc);
create unique index memories_user_text on public.memories (user_id, text_lower);

create table public.feedback (
  id             text primary key,
  user_id        uuid not null default auth.uid() references auth.users(id) on delete cascade,
  session_id     text,
  cycle_id       text,
  proposal_id    text,
  proposal_kind  text not null,
  rating         text not null check (rating in ('useful', 'not_useful')),
  created_at     timestamptz not null default now()
);
create index feedback_user_created on public.feedback (user_id, created_at);

create table public.adaptation (
  user_id     uuid not null default auth.uid() references auth.users(id) on delete cascade,
  kind        text not null,
  bias        double precision not null,
  updated_at  timestamptz not null default now(),
  primary key (user_id, kind)
);

-- Atomic, bounded bias update. SECURITY INVOKER (default): runs as the caller, so RLS still applies.
create or replace function public.apply_bias_delta(p_kind text, p_delta double precision, p_limit double precision)
returns table (old_bias double precision, new_bias double precision)
language plpgsql
set search_path = public
as $$
declare
  v_old double precision;
  v_new double precision;
begin
  if auth.uid() is null then
    raise exception 'not signed in' using errcode = '42501';
  end if;
  select a.bias into v_old from public.adaptation a where a.user_id = auth.uid() and a.kind = p_kind for update;
  v_old := coalesce(v_old, 0.0);
  v_new := greatest(-p_limit, least(p_limit, v_old + p_delta));
  insert into public.adaptation (user_id, kind, bias, updated_at) values (auth.uid(), p_kind, v_new, now())
    on conflict (user_id, kind) do update set bias = excluded.bias, updated_at = excluded.updated_at;
  return query select v_old, v_new;
end;
$$;

-- Row-level security: a user sees and changes only their own rows. Anonymous callers get nothing.
alter table public.memories   enable row level security;
alter table public.feedback   enable row level security;
alter table public.adaptation enable row level security;

create policy "own memories"   on public.memories   for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy "own feedback"   on public.feedback   for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy "own adaptation" on public.adaptation for all to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());

revoke all on public.memories, public.feedback, public.adaptation from anon;
revoke all on function public.apply_bias_delta(text, double precision, double precision) from anon, public;
grant execute on function public.apply_bias_delta(text, double precision, double precision) to authenticated;
