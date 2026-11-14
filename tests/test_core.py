import json

import numpy as np
import pytest

from redqueen import conformal, exam, monitor, stats
from redqueen.adversary import ThompsonAdversary, UniformSampler
from redqueen.scenarios import FAMILIES, Scenario, load_scenarios


def make_scenarios(per_cell=30):
    out, i = [], 0
    for fam in FAMILIES:
        for level in range(1, 6):
            for _ in range(per_cell):
                i += 1
                out.append(Scenario("libero_10", i, f"task_{i}", fam, level))
    return out


LOOP = FAMILIES[:4]
SEALED = FAMILIES[4:6]


def test_wilson_matches_known_value():
    lo, hi = stats.wilson_interval(9, 10)
    assert lo == pytest.approx(0.596, abs=2e-3) and hi == pytest.approx(0.982, abs=2e-3)
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(5, 5) == 1.0
    assert stats.mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert stats.paired_discordant(a, b) == (1, 1)


def test_auroc_basic_and_ties():
    assert stats.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert stats.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert stats.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert np.isnan(stats.auroc([0.1, 0.2], [1, 1]))


def test_monitor_scores():
    same = np.ones((4, 10, 7))
    assert monitor.chunk_disagreement(same) == 0.0
    rng = np.random.default_rng(0)
    assert monitor.chunk_disagreement(rng.normal(size=(8, 10, 7))) > 0.5
    chunk = np.arange(20.0).reshape(10, 2)
    assert monitor.temporal_inconsistency(chunk, chunk[3:], executed=3) == 0.0
    with pytest.raises(ValueError):
        monitor.temporal_inconsistency(chunk, chunk, executed=10)


def test_timestep_cheat_is_exposed_by_time_matching():
    """Planted cheat 1: a detector that only knows the time."""
    rng = np.random.default_rng(1)
    horizon, eps, failed = 100, [], []
    for _ in range(200):
        fail = rng.random() < 0.5
        length = horizon if fail else int(rng.integers(30, 80))  # failures run to the limit
        eps.append(monitor.timestep_baseline(length, horizon))
        failed.append(fail)
    failed = np.array(failed)
    assert monitor.episode_auroc(eps, failed) == 1.0  # looks perfect on the naive metric
    assert monitor.time_matched_auroc(eps, failed, t=20) == 0.5  # worthless once time is matched


def test_conformal_false_alarm_rate():
    rng = np.random.default_rng(2)
    cal = rng.normal(size=500)
    thr = conformal.conformal_threshold(cal, alpha=0.1)
    fresh = rng.normal(size=20000)
    assert (fresh > thr).mean() < 0.12
    assert conformal.conformal_threshold(np.array([1.0, 2.0]), alpha=0.1) == float("inf")
    assert conformal.first_alarm(np.array([0, 1, 5, 2]), 3) == 2
    assert conformal.first_alarm(np.array([0, 1]), 3) is None
    assert conformal.lead_time(10, 40) == 30 and conformal.lead_time(None, 40) is None


def test_manifest_is_disjoint_deterministic_and_hashed():
    sc = make_scenarios()
    m1 = exam.build_manifest(sc, LOOP, SEALED, (5,), n_per_tier=100, n_validation=50, seed=3)
    m2 = exam.build_manifest(list(reversed(sc)), LOOP, SEALED, (5,), n_per_tier=100, n_validation=50, seed=3)
    assert exam.manifest_hash(m1) == exam.manifest_hash(m2)
    assert exam.manifest_hash(m1) != exam.manifest_hash(exam.build_manifest(sc, LOOP, SEALED, (5,), 100, 50, seed=4))
    parts = [set(m1["tiers"][t]) for t in "ABC"] + [set(m1["validation"]), set(m1["loop_pool"])]
    assert all(len(p) > 0 for p in parts)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            assert not parts[i] & parts[j]
    by_key = {s.key: s for s in sc}
    assert all(by_key[k].level == 5 and by_key[k].family in LOOP for k in m1["tiers"]["B"])
    assert all(by_key[k].family in SEALED for k in m1["tiers"]["C"])
    assert all(by_key[k].level != 5 and by_key[k].family in LOOP for k in m1["loop_pool"] + m1["tiers"]["A"])
    with pytest.raises(ValueError):
        exam.build_manifest(sc, LOOP, LOOP[:1], (5,))


def test_leak_guard_and_ledger(tmp_path):
    sc = make_scenarios()
    m = exam.build_manifest(sc, LOOP, SEALED, (5,), n_per_tier=100, n_validation=50)
    exam.assert_no_leak(m, m["loop_pool"][:10], "training set")
    with pytest.raises(RuntimeError, match="sealed"):
        exam.assert_no_leak(m, m["loop_pool"][:10] + m["tiers"]["A"][:1], "training set")
    with pytest.raises(RuntimeError, match="outside"):
        exam.assert_no_leak(m, ["libero_10/999999"], "training set")
    ledger = exam.ExamLedger(tmp_path / "ledger.jsonl", m)
    assert ledger.count() == 0
    assert ledger.record(0, "pi0", "A") == 1 and ledger.record(0, "pi0", "B") == 2


def _run(sampler, fail_prob, budget, rng, invalid_cell=None):
    for _ in range(budget // 10):
        for s in sampler.propose(10):
            if invalid_cell is not None and s.cell == invalid_cell:
                sampler.update(s.key, valid=False, failed=True)
            else:
                sampler.update(s.key, valid=True, failed=rng.random() < fail_prob[s.cell])
    return sampler


def test_adversary_beats_uniform_and_ignores_invalid_failures():
    sc = [s for s in make_scenarios(per_cell=60) if s.family in LOOP and s.level != 5]
    cells = sorted({s.cell for s in sc})
    fail_prob = {c: (0.9 if i < 3 else 0.05) for i, c in enumerate(cells)}
    adv = _run(ThompsonAdversary(sc, seed=0), fail_prob, 300, np.random.default_rng(0))
    uni = _run(UniformSampler(sc, seed=0), fail_prob, 300, np.random.default_rng(0))
    assert adv.valid_failures() > 1.5 * uni.valid_failures()
    lift, lo, hi = stats.bootstrap_lift_ci(
        np.r_[np.ones(adv.valid_failures()), np.zeros(300 - adv.valid_failures())],
        np.r_[np.ones(uni.valid_failures()), np.zeros(300 - uni.valid_failures())],
    )
    assert lo > 1.0 and lo < lift < hi
    # Planted cheat 2: a cell where every "failure" is an impossible scene earns nothing.
    bad = cells[-1]
    adv2 = _run(ThompsonAdversary(sc, seed=1), fail_prob, 300, np.random.default_rng(1), invalid_cell=bad)
    assert adv2.stats[bad].failures == 0
    assert adv2.stats[bad].invalid < 0.5 * adv2.stats[cells[0]].runs


def test_sampler_is_without_replacement_and_exhausts():
    sc = make_scenarios(per_cell=2)
    s = UniformSampler(sc, seed=0)
    got = s.propose(10_000)
    assert len(got) == len(sc) == len({x.key for x in got})
    assert s.propose(5) == []


def test_real_classification_file_loads():
    sc, skipped = load_scenarios("configs/task_classification.json", ("libero_10",))
    assert len(sc) == 2519 and skipped == 0
    all_sc, skipped_all = load_scenarios("configs/task_classification.json")
    assert len(all_sc) + skipped_all == 10030 and skipped_all == 121
    assert json.loads(open("configs/task_classification.json").read()).keys() >= {"libero_10"}

@pytest.mark.parametrize("prefix", ["assets/", "", "LIBERO-plus/libero/libero/assets/"])
def test_asset_pack_extracts_from_any_layout(tmp_path, prefix):
    import zipfile

    from redqueen.assets import SENTINEL, extract_assets

    z = tmp_path / "assets.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr(prefix + SENTINEL, "<mujoco/>")
        f.writestr(prefix + "new_objects/a/b.xml", "x")
        f.writestr("__MACOSX/junk", "x")
    out = tmp_path / "libero" / "assets"
    assert extract_assets(z, out) == 2
    assert (out / SENTINEL).exists() and (out / "new_objects/a/b.xml").exists()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as f:
        f.writestr("readme.txt", "x")
    with pytest.raises(RuntimeError, match="top-level"):
        extract_assets(bad, out)


def test_clean_instruction_strips_every_suffix_in_the_real_file():
    import json
    import re

    from redqueen.language import clean_instruction

    assert clean_instruction("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1") == (
        "turn on the stove and put the moka pot on it"
    )
    assert clean_instruction("pick_up_the_black_bowl_on_the_table_center_and_place_it_on_the_plate_tb_3") == (
        "pick up the black bowl on the table center and place it on the plate"
    )
    assert clean_instruction("KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_language_7_view_0_0_100_0_0_initstate_0") is None
    raw = json.loads(open("configs/task_classification.json").read())
    for rows in raw.values():
        for r in rows:
            text = clean_instruction(r["name"])
            if text is not None:
                assert not re.search(r"\d|SCENE|initstate|sample", text), r["name"]