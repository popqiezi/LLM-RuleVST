# XML Rule-Base Schema

The XML rule base is organized using the following hierarchy:

```text
rule_base
└── encounter
    └── rule
        ├── premise
        ├── required_action
        ├── compliance_cues
        ├── violation_conditions
        ├── parameters
        └── retrieval_tags
```

## Field definitions

- `encounter`: scenario category used to narrow retrieval.
- `rule`: primary rule identifier.
- `premise`: applicability conditions.
- `required_action`: operational action expected under the rule.
- `compliance_cues`: observable behavior consistent with the rule.
- `violation_conditions`: observable deviations used for structured labeling.
- `parameters`: input variables required by the rule unit.
- `retrieval_tags`: lexical and semantic tags used by the retriever.

The XML content is an operational abstraction for the research pipeline and
must be verified against the exact official or cited regional rule source
before public release.
