from scripts.validate_output import validate_output


def make_vocab():
    return {
        "encounter_type": {"head_on": 2},
        "active_rule": {"Rule14": 6},
        "own_role": {"give_way": 2},
        "compliance_state": {"warning": 3},
        "violation_action": {"late_turn": 3},
    }


def test_valid_output():
    output = {
        "encounter_type": "head_on",
        "active_rule": "Rule14",
        "own_role": "give_way",
        "compliance_state": "warning",
        "violation_action": "late_turn",
    }
    assert validate_output(output, make_vocab(), ["Rule14"]) == []


def test_unsupported_rule():
    output = {
        "encounter_type": "head_on",
        "active_rule": "Rule14",
        "own_role": "give_way",
        "compliance_state": "warning",
        "violation_action": "late_turn",
    }
    errors = validate_output(output, make_vocab(), ["Rule7"])
    assert any(error.startswith("E05_UNSUPPORTED_RULE") for error in errors)
