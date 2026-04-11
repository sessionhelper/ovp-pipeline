//! Metatalk classifier operator.
//!
//! Tags each transcript segment as in-character ("ic") or
//! out-of-character ("ooc") using rule-based keyword/pattern matching.
//! No LLM required — classification is purely heuristic.
//!
//! Ambiguous segments default to "ic" because it is better to keep
//! roleplay intact than to wrongly strip it out.

use crate::error::Result;
use crate::types::TranscriptSegment;

use super::{Operator, OperatorResult};

/// Rule-based in-character vs out-of-character classifier.
pub struct MetatalkOperator {
    /// Segments classified as IC.
    ic_count: u32,
    /// Segments classified as OOC.
    ooc_count: u32,
}

/// Strong OOC patterns — a single match is enough to tag the segment.
const STRONG_OOC_PATTERNS: &[&str] = &[
    "roll a d",
    "roll d",
    "rolled a",
    "saving throw",
    "spell slots",
    "spell slot",
    "hit points",
    "attack of opportunity",
    "bonus action",
    "the phb says",
    "according to the rules",
    "whose turn",
    "what's my ac",
    "does that hit",
    "is that in range",
    "how far is",
    "do i need to roll",
    "what do i roll",
    "nat 20",
    "nat 1",
    "natural 20",
    "natural 1",
    "short rest",
    "long rest",
    "death save",
    "death saving",
    "concentration check",
    "ability check",
    "skill check",
];

/// Weak OOC keywords — need >=2 in one segment to trigger OOC.
const WEAK_OOC_KEYWORDS: &[&str] = &[
    "roll",
    "dice",
    "d20",
    "d12",
    "d10",
    "d8",
    "d6",
    "d4",
    "modifier",
    "initiative",
    "ac",
    "hp",
    "raw",
    "technically",
    "wait",
    "hold on",
    "sorry",
    "let me think",
    "what did you say",
    "remember last session",
    "can i",
    "do i need to",
    "how far",
    "what's my",
    "advantage",
    "disadvantage",
    "proficiency",
    "damage",
    "plus",
];

/// Minimum number of weak keyword matches to classify as OOC.
const WEAK_THRESHOLD: usize = 2;

impl MetatalkOperator {
    /// Create a new metatalk classifier operator.
    pub fn new() -> Self {
        Self {
            ic_count: 0,
            ooc_count: 0,
        }
    }

    /// Classify a segment's text as "ic" or "ooc".
    fn classify(text: &str) -> &'static str {
        let lower = text.to_lowercase();

        // Strong patterns: one match is enough.
        for pattern in STRONG_OOC_PATTERNS {
            if lower.contains(pattern) {
                return "ooc";
            }
        }

        // Weak keywords: need >= WEAK_THRESHOLD distinct matches.
        let mut weak_hits = 0usize;
        for keyword in WEAK_OOC_KEYWORDS {
            if Self::keyword_match(&lower, keyword) {
                weak_hits += 1;
                if weak_hits >= WEAK_THRESHOLD {
                    return "ooc";
                }
            }
        }

        // Default: in-character.
        "ic"
    }

    /// Word-boundary-aware keyword match.
    ///
    /// Short keywords like "ac", "hp", "d6" must not fire inside longer
    /// words (e.g. "place", "shipping", "odd6"). Multi-word patterns use
    /// simple substring containment since they are already specific enough.
    fn keyword_match(haystack: &str, keyword: &str) -> bool {
        // Multi-word keywords: substring is fine.
        if keyword.contains(' ') {
            return haystack.contains(keyword);
        }

        // Single-word keywords: check word boundaries.
        for (idx, _) in haystack.match_indices(keyword) {
            let before = if idx == 0 {
                true
            } else {
                !haystack.as_bytes()[idx - 1].is_ascii_alphanumeric()
            };
            let after_idx = idx + keyword.len();
            let after = if after_idx >= haystack.len() {
                true
            } else {
                !haystack.as_bytes()[after_idx].is_ascii_alphanumeric()
            };
            if before && after {
                return true;
            }
        }
        false
    }
}

impl Default for MetatalkOperator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Operator for MetatalkOperator {
    async fn on_segment(&mut self, segment: &mut TranscriptSegment) -> OperatorResult {
        let tag = Self::classify(&segment.text);
        segment.talk_type = Some(tag.to_owned());

        match tag {
            "ooc" => self.ooc_count += 1,
            _ => self.ic_count += 1,
        }

        // Never excludes — classification only.
        OperatorResult::Pass
    }

    async fn sweep(&mut self) -> Result<u32> {
        // No retroactive analysis needed.
        Ok(0)
    }

    async fn finalize(&mut self) -> Result<()> {
        tracing::info!(
            ic = self.ic_count,
            ooc = self.ooc_count,
            "metatalk operator finalized"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_segment(text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            segment_index: 0,
            speaker_pseudo_id: "speaker_a".into(),
            start_time: 0.0,
            end_time: 1.0,
            text: text.into(),
            original_text: text.into(),
            confidence: None,
            beat_id: None,
            chunk_group: None,
            talk_type: None,
            excluded: false,
            exclude_reason: None,
        }
    }

    // --- Clear IC cases ---

    #[tokio::test]
    async fn ic_dialogue() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("I draw my sword and step toward the goblin.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    #[tokio::test]
    async fn ic_narrative() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("The tavern door creaks open and a cold wind blows in.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    #[tokio::test]
    async fn ic_character_speech() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Greetings, traveler. What brings you to our village?");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    #[tokio::test]
    async fn ic_cast_action() {
        // "I cast" alone is a character action, not enough OOC signal.
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("I cast fireball at the group of enemies.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    // --- Clear OOC cases ---

    #[tokio::test]
    async fn ooc_strong_roll_a_d20() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("I need to roll a d20 for that.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_strong_saving_throw() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Make a saving throw please.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_strong_does_that_hit() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Does that hit?");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_strong_bonus_action() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Can I use my bonus action?");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_strong_spell_slots() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("I'm out of spell slots.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_strong_whose_turn() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Wait whose turn is it?");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_weak_multiple_keywords() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Wait, can I roll for initiative?");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_weak_damage_plus() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("That's 8 damage plus my modifier.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    #[tokio::test]
    async fn ooc_rules_reference() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("The PHB says you can't do that.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ooc"));
    }

    // --- Ambiguous defaults to IC ---

    #[tokio::test]
    async fn ambiguous_defaults_to_ic() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Alright, let's keep going.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    #[tokio::test]
    async fn single_weak_keyword_stays_ic() {
        // A single weak keyword is not enough.
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Sorry about that.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    // --- Word boundary checks ---

    #[tokio::test]
    async fn ac_in_place_does_not_trigger() {
        // "ac" inside "place" should not match.
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("This place is beautiful.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    #[tokio::test]
    async fn hp_in_shipping_does_not_trigger() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("The shipping route goes north.");
        op.on_segment(&mut seg).await;
        assert_eq!(seg.talk_type.as_deref(), Some("ic"));
    }

    // --- Finalize stats ---

    #[tokio::test]
    async fn finalize_counts() {
        let mut op = MetatalkOperator::new();
        let mut ic = make_segment("The dragon roars.");
        let mut ooc = make_segment("Roll a d20 for initiative.");
        op.on_segment(&mut ic).await;
        op.on_segment(&mut ooc).await;
        assert_eq!(op.ic_count, 1);
        assert_eq!(op.ooc_count, 1);
        op.finalize().await.unwrap();
    }

    // --- Never excludes ---

    #[tokio::test]
    async fn never_excludes() {
        let mut op = MetatalkOperator::new();
        let mut seg = make_segment("Roll a d20 for that saving throw please.");
        let result = op.on_segment(&mut seg).await;
        assert!(matches!(result, OperatorResult::Pass));
        assert!(!seg.excluded);
    }
}
