-- Beat the Model: database setup for Supabase.
-- Paste this whole file into Supabase > SQL Editor > New query, and click Run. Run it once.
--
-- How it stays fair:
--   * Visitors can't touch the tables directly. They can only call two functions:
--     join_game (pick a nickname) and submit_pick (make a pick with their secret code).
--   * Every pick is stamped with the DATABASE's clock (not the visitor's), and the weekly
--     grading only counts picks stamped before that game's kickoff.
--   * Secret codes are stored hashed, so nobody can make picks for someone else.

create extension if not exists pgcrypto with schema extensions;

create table if not exists public.players (
  id          uuid primary key default gen_random_uuid(),
  nickname    text not null check (char_length(trim(nickname)) between 2 and 20),
  secret_hash text not null,
  created_at  timestamptz not null default now()
);
create unique index if not exists players_nickname_unique on public.players (lower(trim(nickname)));

create table if not exists public.picks (
  id         bigserial primary key,
  player_id  uuid not null references public.players(id) on delete cascade,
  season     int  not null check (season between 2026 and 2100),
  week       int  not null check (week between 1 and 22),
  game       text not null check (game in ('h2h', 'ou')),
  item       int  not null check (item between 0 and 49),
  choice     text not null check (choice in ('a', 'b', 'over', 'under')),
  created_at timestamptz not null default now()
);
create index if not exists picks_week on public.picks (season, week);

alter table public.players enable row level security;
alter table public.picks enable row level security;
revoke all on public.players, public.picks from anon, authenticated;

-- Join: returns your player id and a secret code (keep it to play from other devices).
create or replace function public.join_game(p_nickname text)
returns json
language plpgsql security definer set search_path = public, extensions
as $$
declare
  s text := encode(gen_random_bytes(18), 'hex');
  pid uuid;
begin
  if p_nickname !~ '^[A-Za-z0-9 _.''-]{2,20}$' then
    raise exception 'Nicknames are 2-20 letters, numbers, spaces, or _ . '' -';
  end if;
  insert into players (nickname, secret_hash) values (trim(p_nickname), crypt(s, gen_salt('bf')))
  returning id into pid;
  return json_build_object('player_id', pid, 'secret', s);
exception when unique_violation then
  raise exception 'That nickname is taken';
end $$;

-- Pick: the database stamps the time. Changing a pick just adds a newer one.
create or replace function public.submit_pick(p_player uuid, p_secret text, p_season int, p_week int,
                                              p_game text, p_item int, p_choice text)
returns timestamptz
language plpgsql security definer set search_path = public, extensions
as $$
declare t timestamptz;
begin
  if not exists (select 1 from players where id = p_player and secret_hash = crypt(p_secret, secret_hash)) then
    raise exception 'Unknown player code';
  end if;
  if (select count(*) from picks where player_id = p_player and season = p_season and week = p_week) >= 300 then
    raise exception 'Too many picks this week';
  end if;
  insert into picks (player_id, season, week, game, item, choice)
  values (p_player, p_season, p_week, p_game, p_item, p_choice)
  returning created_at into t;
  return t;
end $$;

-- Public, read-only: everyone's picks with nicknames (used for grading and the crowd split).
create or replace view public.picks_public as
  select k.player_id, p.nickname, k.season, k.week, k.game, k.item, k.choice, k.created_at
  from public.picks k join public.players p on p.id = k.player_id;

-- Public, read-only: how the crowd is splitting on each item (each player's latest pick).
create or replace view public.pick_split as
  select season, week, game, item, choice, count(*)::int as n
  from (select distinct on (player_id, season, week, game, item) season, week, game, item, choice
        from public.picks order by player_id, season, week, game, item, created_at desc) latest
  group by season, week, game, item, choice;

grant select on public.picks_public, public.pick_split to anon, authenticated;
grant execute on function public.join_game(text) to anon, authenticated;
grant execute on function public.submit_pick(uuid, text, int, int, text, int, text) to anon, authenticated;
