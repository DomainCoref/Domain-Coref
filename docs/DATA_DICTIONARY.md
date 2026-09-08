# Domain-Coref Data Dictionary

Document fields:
- category_primary: document-level primary anaphoric phenomenon
- domain: discourse domain
- text: complete document text
- sentences: sentence segmentation
- coreference_chains: identity-coreference chains
- entities: auxiliary entity inventory

Mention fields:
- sentence_id: 1-based sentence identifier
- text: overt mention text or annotated zero symbol
- char_start / char_end: inclusive character offsets
- role: mention role
- mention_type: personal pronoun, demonstrative, definite description,
  zero pronoun, or event mention
- referent_semantics: ENTITY or EVENT

Only identity coreference is annotated. Bridging, causal, part-whole,
temporal, and other non-identity relations are excluded.
