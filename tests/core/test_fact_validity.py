import copy
import unittest
from datetime import datetime

from tsuzuri.fact_validity import (WORLD_TZ, candidates_for_review, claim_id, project_cards,
                           project_states, review_current_turn, validate_review)


def ms(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=WORLD_TZ).timestamp()*1000)


NOW = ms('2026-09-08T19:14')


class FactValidityTests(unittest.TestCase):
    def user(self, text, **kw):
        return dict(role='user', content=text, timestamp=NOW, id='u-now', audience_scope='developer', **kw)

    def card(self, **kw):
        card = dict(character='李四', audience_scope='developer', turn_id='old',
                    user_id='u-old', assistant_id='a-old', created_at_ms=NOW-86400000,
                    fact='李四答应看北町路线。', commitments=['李四答应看北町路线。', '李四答应保管阿澈的信。'],
                    open_threads=[], corrections=[], salience=.8, confidence=.9)
        card.update(kw)
        return card

    def state(self, predicate='travel.status', evidence='我过一会就要出远门了', **kw):
        row = dict(predicate=predicate, evidence_text=evidence, value_text='即将出远门',
                   source_character='李四', source_at_ms=ms('2026-08-17T11:21'),
                   valid_until_ms=ms('2026-10-01T11:21'), temporal_status='planned', status='active')
        row.update(kw)
        return row

    def test_old_trip_exits_current_without_mutating_history(self):
        old = self.state()
        original = copy.deepcopy(old)
        result = project_states([old], '李四', NOW)
        self.assertEqual(result['active'], [])
        self.assertEqual(result['planned'], [])
        self.assertEqual(old, original)
        self.assertEqual(result['decisions'][0]['reason'], 'validity_elapsed')

    def test_today_free_does_not_last_a_week(self):
        row = self.state('availability.current', '我今天没什么安排呢',
                         source_at_ms=ms('2026-09-05T14:06'), temporal_status='active')
        self.assertFalse(project_states([row], '李四', NOW)['active'])

    def test_finished_event_not_workload(self):
        row = self.state('workload.current', '我上午刚给学弟学妹发完传单', temporal_status='active')
        self.assertEqual(project_states([row], '李四', NOW)['decisions'][0]['reason'], 'past_event_not_state')

    def test_uncertain_source_does_not_become_certain(self):
        row = self.state('workload.current', '我下周可能会非常忙', source_at_ms=NOW-1000)
        self.assertEqual(project_states([row], '李四', NOW)['decisions'][0]['reason'], 'uncertain_source')

    def test_return_updates_same_turn_and_both_fields(self):
        result = project_states([self.state()], '李四', NOW, self.user('我已经回到家了'))
        self.assertEqual({r['predicate']: r['value_text'] for r in result['active']},
                         {'travel.status': '已回到家', 'location.current': '在家'})

    def test_quote_question_third_party_and_negation_do_not_update(self):
        for text in ['你说“我已经回到家了”是什么意思？', '我朋友已经回到家了',
                     '我已经回到家了吗？', '如果我已经回到家了', '我还没有回到家']:
            with self.subTest(text=text):
                self.assertFalse(project_states([], '李四', NOW, self.user(text))['active'])

    def test_departure_time_does_not_prove_departure(self):
        row = self.state(source_at_ms=NOW-1000, valid_from_ms=NOW-500)
        result = project_states([row], '李四', NOW)
        self.assertEqual(len(result['planned']), 1)
        self.assertEqual(result['active'], [])

    def test_enduring_academic_phase_survives(self):
        row = self.state('academic.phase', '我今天开学了', temporal_status='active', value_text='in_term')
        self.assertEqual(len(project_states([row], '李四', NOW)['active']), 1)

    def test_other_character_and_future_evidence_cannot_leak(self):
        rows = [self.state(source_character='张三'), self.state(source_at_ms=NOW+1000)]
        result = project_states(rows, '李四', NOW)
        self.assertFalse(result['active'] or result['planned'])

    def action(self, card, status='cancelled', text='北町路线不用帮我看了'):
        user = self.user(text)
        raw = {'actions': [{'target_id': claim_id(card, 'commitments', card['commitments'][0]),
                            'status': status, 'evidence_span': text, 'confidence': .99}]}
        return validate_review(raw, user, candidates_for_review([card], '李四', NOW))

    def test_per_item_close_preserves_other_promise_and_source(self):
        card = self.card(fact='阿澈喜欢蓝色。')
        before = copy.deepcopy(card)
        result, decisions = project_cards([card], '李四', NOW, self.action(card))
        self.assertEqual(result[0]['commitments'], ['李四答应保管阿澈的信。'])
        self.assertEqual(result[0]['fact'], card['fact'])
        self.assertEqual(card, before)
        self.assertEqual(len(decisions), 1)

    def test_identical_fact_cannot_resurrect_closed_promise(self):
        card = self.card()
        result, _ = project_cards([card], '李四', NOW, self.action(card))
        self.assertEqual(result[0]['fact'], '')
        self.assertEqual(result[0]['near_summary'], '')
        self.assertEqual(result[0]['commitments'], ['李四答应保管阿澈的信。'])

    def test_projection_survives_workspace_sanitizer(self):
        from tsuzuri.turn_fact_ledger import select_fact_cards_for_workspace
        card = self.card()
        result, _ = project_cards([card], '李四', NOW, self.action(card), historical=True)
        selected = select_fact_cards_for_workspace([self.user('还记得那次约定吗')], result)
        self.assertEqual(selected[0]['commitments'], ['李四答应保管阿澈的信。'])
        self.assertTrue(selected[0]['_background_facts'])

    def test_resolution_survives_persistence_roundtrip(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card, sanitize_fact_cards
        old = self.card()
        user = self.user('北町路线不用帮我看了')
        user['_fact_validity'] = self.action(old)
        new = normalize_turn_fact_card({'fact': '阿澈取消北町路线约定'}, '李四', user,
                                      {'id': 'a-now', 'timestamp': NOW+1000, 'content': '好'})
        cards = sanitize_fact_cards([old, new], limit=104)
        result, _ = project_cards(cards, '李四', NOW+2000)
        self.assertEqual(result[0]['commitments'], ['李四答应保管阿澈的信。'])

    def test_model_cannot_forge_resolution_in_post_reply_card(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card
        old = self.card()
        raw = {'fact': '普通对话', 'fact_validity_actions': self.action(old)['actions']}
        new = normalize_turn_fact_card(raw, '李四', self.user('你好'), {'id': 'a', 'content': '你好', 'timestamp': NOW+1000})
        self.assertEqual(new['fact_validity_actions'], [])

    def test_aquarium_recollection_cannot_become_world_authority(self):
        from tsuzuri.character_action_ledger import normalize_character_action_updates
        entry = {'role': 'assistant', 'content': '今天去墨田水族馆，在水母回廊站了很久',
                 'timestamp': NOW, '_action_authority_snapshot': {'schedule_state': {
                     'past_steps': [{'title': '和张三看水母', 'location': '阳光城水族馆'}]}}}
        raw = [{'transition': 'complete', 'event_text': '墨田水族馆看水母',
                'evidence_span': entry['content'], 'confidence': .99}]
        self.assertEqual(normalize_character_action_updates(raw, entry), [])
        entry['content'] = entry['content'].replace('墨田', '阳光城')
        raw[0]['event_text'] = raw[0]['event_text'].replace('墨田', '阳光城')
        raw[0]['evidence_span'] = entry['content']
        self.assertEqual(len(normalize_character_action_updates(raw, entry)), 1)

    def test_cross_channel_requires_exact_occurrence(self):
        from tsuzuri.fact_validity import record_resolution
        card = self.card()
        actions = self.action(card)['actions']
        row = {'text': card['commitments'][0], 'created_at_ms': card['created_at_ms'], 'source_character': '李四'}
        self.assertEqual(record_resolution(row, actions, '李四', NOW), 'cancelled')
        for changed in ({'created_at_ms': NOW-2000}, {'source_character': '张三'}, {'text': '阿澈答应帮别人看北町路线'}):
            self.assertEqual(record_resolution({**row, **changed}, actions, '李四', NOW), '')

    def test_hypothesis_cannot_supply_partial_evidence(self):
        from tsuzuri.interlocutor_state_ledger import normalize_developer_state_updates
        user = self.user('如果我已经回到家了，就可以休息')
        self.assertEqual(normalize_developer_state_updates([{'predicate': 'travel.status', 'value': '已回家',
            'evidence_span': '我已经回到家了', 'temporal_status': 'active'}], user), [])

    def test_unrelated_not_yet_does_not_make_school_planned(self):
        from tsuzuri.interlocutor_state_ledger import normalize_developer_state_updates
        rows = normalize_developer_state_updates([], self.user('我还没吃午饭呢，刚开学'))
        self.assertEqual(rows[0]['temporal_status'], 'active')

    def test_unknown_target_cannot_close_anything(self):
        card = self.card()
        review = self.action(card)
        review['actions'][0]['target_id'] = 'unknown'
        result, decisions = project_cards([card], '李四', NOW, review)
        self.assertEqual(result[0]['commitments'], card['commitments'])
        self.assertEqual(decisions, [])

    def test_thanks_or_unfinished_cannot_fulfill(self):
        card = self.card()
        for text in ['谢谢', '晚安', '我还没完成北町路线', '我没收到路线']:
            with self.subTest(text=text):
                self.assertEqual(self.action(card, 'fulfilled', text)['actions'], [])

    def test_history_query_retains_closed_fact_as_history(self):
        card = self.card()
        result, _ = project_cards([card], '李四', NOW, self.action(card), historical=True)
        self.assertIn('已取消的约定', result[0]['_background_facts'][0])
        self.assertNotIn(card['commitments'][0], result[0]['commitments'])

    def test_background_is_turn_local_not_completion(self):
        card = self.card(fact='李四确认今天是周二。', commitments=['李四确认今天是周二。'])
        review = self.action(card, 'background', '李四确认今天是周二。')
        result, _ = project_cards([card], '李四', NOW, review)
        self.assertIn(card['commitments'][0], result[0]['_background_facts'])
        card['fact_validity_actions'] = review['actions']
        later, _ = project_cards([card], '李四', NOW+1000)
        self.assertEqual(later[0]['commitments'], card['commitments'])

    def test_failure_keeps_important_promise(self):
        card = self.card()
        def fail(prompt):
            raise TimeoutError()
        review = review_current_turn('李四', self.user('那封信还在吗'), [card], {}, [('fake', fail)])
        result, _ = project_cards([card], '李四', NOW, review)
        self.assertEqual(result[0]['commitments'], card['commitments'])

    def test_guest_never_calls_reviewer(self):
        calls = []
        user = self.user('你好')
        user['audience_scope'] = 'guest:1'
        review = review_current_turn('李四', user, [self.card()], {}, [('fake', lambda p: calls.append(p))])
        self.assertEqual(review, {})
        self.assertEqual(calls, [])

    def test_same_topic_different_day_and_character_have_distinct_ids(self):
        card = self.card()
        other = self.card(turn_id='different-day')
        peer = self.card(character='张三')
        ids = {claim_id(c, 'commitments', c['commitments'][0]) for c in (card, other, peer)}
        self.assertEqual(len(ids), 3)

    def test_elapsed_small_promise_is_recallable_but_not_current_commitment(self):
        from tsuzuri.turn_context_workspace import _card_records
        card = self.card(commitments=['李四今晚提醒阿澈喝水。'], event_time='今晚')
        rows = _card_records([card], as_of_ms=NOW)
        self.assertFalse(any(r['kind'] == 'commitment' for r in rows))
        self.assertTrue(any('是否履行未确认' in r['text'] for r in rows))
        self.assertEqual(card['commitments'], ['李四今晚提醒阿澈喝水。'])

    def test_undated_and_enduring_promises_are_not_expired(self):
        from tsuzuri.turn_context_workspace import _card_records
        for text, event_time in [('李四答应保管阿澈的信。', ''), ('李四今天答应一直保管阿澈的信。', '今天')]:
            rows = _card_records([self.card(commitments=[text], event_time=event_time)], as_of_ms=NOW)
            self.assertTrue(any(r['kind'] == 'commitment' and r['text'] == text for r in rows))

    def test_card_time_does_not_expire_a_different_promise(self):
        from tsuzuri.turn_context_workspace import _card_records
        card = self.card(commitments=['李四今晚提醒阿澈喝水。', '李四答应保管阿澈的信。'], event_time='今晚')
        rows = _card_records([card], as_of_ms=NOW)
        self.assertTrue(any(r['kind'] == 'commitment' and '保管' in r['text'] for r in rows))

    def test_invented_event_date_is_not_accepted(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card
        card = normalize_turn_fact_card({'fact': '讨论路线', 'event_time': '昨天'}, '李四',
            self.user('我明天去北町'), {'id': 'a', 'content': '小心', 'timestamp': NOW+1000})
        self.assertEqual(card['event_time'], '')

    def test_arriving_home_does_not_cancel_next_weeks_trip(self):
        import tempfile
        from pathlib import Path
        from tsuzuri.interlocutor_state_ledger import commit_turn_state_updates, load_planned_states
        future = self.state(evidence='我下周去大阪旅行', source_at_ms=NOW-1000,
                            valid_from_ms=NOW+7*86400000, valid_until_ms=NOW+14*86400000,
                            temporal_status='planned', status='planned', value_text='去大阪旅行')
        projected = project_states({'planned': [future]}, '李四', NOW, self.user('我到家啦'))
        self.assertEqual(len(projected['planned']), 1)
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td)/'state.db')
            commit_turn_state_updates('李四', 's', {'character': '李四', 'user_id': 'u-plan', 'turn_id': 't-plan',
                'audience_scope': 'developer', 'created_at_ms': NOW-1000, 'developer_state_updates': [{
                    **future, 'value': future['value_text'], 'operation': 'set', 'evidence_span': future['evidence_text']}]}, db_path=db)
            from tsuzuri.turn_fact_ledger import normalize_turn_fact_card
            returned = normalize_turn_fact_card({'fact': '阿澈到家'}, '李四', self.user('我到家啦'),
                                               {'id': 'a', 'content': '好', 'timestamp': NOW+1000})
            commit_turn_state_updates('李四', 's', returned, db_path=db)
            self.assertEqual(len(load_planned_states('李四', now_ms=NOW+2000, db_path=db)), 1)


if __name__ == '__main__':
    unittest.main()
