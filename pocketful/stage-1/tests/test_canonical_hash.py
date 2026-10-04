import pytest
from app.idempotency.canonical_hash import canonical_body_hash


class TestCanonicalBodyHash:
    def test_same_keys_different_order_produce_same_hash(self):
        body1 = {"to_handle": "bob", "amount": 1500, "note": "dinner"}
        body2 = {"amount": 1500, "note": "dinner", "to_handle": "bob"}
        assert canonical_body_hash(body1) == canonical_body_hash(body2)

    def test_different_whitespace_produce_same_hash(self):
        body1 = '{"to_handle":"bob","amount":1500}'
        body2 = '{"to_handle": "bob", "amount": 1500}'
        body3 = '''
        {
            "to_handle": "bob",
            "amount": 1500
        }
        '''
        assert canonical_body_hash(body1) == canonical_body_hash(body2)
        assert canonical_body_hash(body2) == canonical_body_hash(body3)

    def test_string_input_same_as_dict(self):
        body_str = '{"to_handle": "bob", "amount": 1500}'
        body_dict = {"to_handle": "bob", "amount": 1500}
        assert canonical_body_hash(body_str) == canonical_body_hash(body_dict)

    def test_different_bodies_produce_different_hashes(self):
        body1 = {"to_handle": "bob", "amount": 1500}
        body2 = {"to_handle": "bob", "amount": 1501}
        body3 = {"to_handle": "alice", "amount": 1500}
        assert canonical_body_hash(body1) != canonical_body_hash(body2)
        assert canonical_body_hash(body1) != canonical_body_hash(body3)

    def test_nested_objects_same_hash(self):
        body1 = {"outer": {"a": 1, "b": 2}, "c": 3}
        body2 = {"c": 3, "outer": {"b": 2, "a": 1}}
        assert canonical_body_hash(body1) == canonical_body_hash(body2)

    def test_arrays_same_hash(self):
        body1 = {"items": [1, 2, 3]}
        body2 = {"items": [1, 2, 3]}
        assert canonical_body_hash(body1) == canonical_body_hash(body2)

    def test_array_order_matters(self):
        body1 = {"items": [1, 2, 3]}
        body2 = {"items": [3, 2, 1]}
        assert canonical_body_hash(body1) != canonical_body_hash(body2)

    def test_empty_object(self):
        assert canonical_body_hash({}) == canonical_body_hash("{}")

    def test_null_values(self):
        body1 = {"a": None, "b": None}
        body2 = {"b": None, "a": None}
        assert canonical_body_hash(body1) == canonical_body_hash(body2)

    def test_boolean_and_numeric_values(self):
        body1 = {"active": True, "count": 0}
        body2 = {"count": 0, "active": True}
        assert canonical_body_hash(body1) == canonical_body_hash(body2)