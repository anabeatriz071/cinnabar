//! Stable caster admission and selective invalidation of a retained acceleration structure.

pub(crate) const MASK_WORDS: usize = 4;
pub(crate) const MAX_SECTIONS: usize = MASK_WORDS * u32::BITS as usize;
const RETENTION_MARGIN: f32 = 8.0;

/// Keeps resident casters through small distance changes, with bounded outer-radius retention.
pub(crate) fn select_casters<K: Copy + Ord>(
    candidates: impl IntoIterator<Item = (f32, K, bool)>,
    radius: f32,
    limit: usize,
) -> Vec<K> {
    let mut candidates: Vec<_> = candidates
        .into_iter()
        .filter(|&(distance, _, resident)| retain_caster(distance, resident, radius))
        .map(|(distance, key, resident)| {
            let score = if resident {
                (distance - RETENTION_MARGIN).max(0.0)
            } else {
                distance
            };
            (score, key)
        })
        .collect();
    candidates.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    candidates
        .into_iter()
        .take(limit)
        .map(|(_, key)| key)
        .collect()
}

/// A wider release radius prevents repeatedly admitting and evicting a border caster.
pub(crate) fn retain_caster(distance: f32, resident: bool, radius: f32) -> bool {
    distance
        < radius
            + if resident {
                RETENTION_MARGIN * 2.0
            } else {
                0.0
            }
}

/// Masks only withdrawn instances; the rest of the submitted scene remains usable during rebuilds.
pub(crate) fn invalid_instance_mask<K>(
    submitted: &[K],
    valid: impl Fn(&K) -> bool,
) -> [u32; MASK_WORDS] {
    assert!(submitted.len() <= MAX_SECTIONS);
    let mut mask = [0; MASK_WORDS];
    for (index, key) in submitted.iter().enumerate() {
        if !valid(key) {
            mask[index / u32::BITS as usize] |= 1 << (index % u32::BITS as usize);
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_camera_movements_keep_resident_casters_at_capacity() {
        for movement in [-0.2, 0.0, 0.2] {
            assert_eq!(
                select_casters(
                    [(20.0 + movement, 1, true), (20.0 - movement, 2, false)],
                    72.0,
                    1
                ),
                [1]
            );
        }
        assert_eq!(
            select_casters([(30.0, 1, true), (15.0, 2, false)], 72.0, 1),
            [2]
        );
    }

    #[test]
    fn radius_boundary_keeps_residents_until_the_release_margin() {
        assert_eq!(
            select_casters([(73.0, 1, true), (73.0, 2, false)], 72.0, 2),
            [1]
        );
        assert!(select_casters([(89.0, 1, true)], 72.0, 1).is_empty());
    }

    #[test]
    fn withdrawing_one_caster_keeps_other_submitted_instances_visible() {
        assert_eq!(
            invalid_instance_mask(&[10, 20, 30], |id| *id != 20),
            [2, 0, 0, 0]
        );
        let submitted: Vec<_> = (0..MAX_SECTIONS).collect();
        assert_eq!(
            invalid_instance_mask(&submitted, |id| ![31, 32, 127].contains(id)),
            [1 << 31, 1, 0, 1 << 31]
        );
        assert_eq!(invalid_instance_mask(&submitted, |_| true), [0; MASK_WORDS]);
    }
}
