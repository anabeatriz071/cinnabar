//! Additive native DX12 RT pipelines, independent of Cinnabar's existing wgpu API.
#[cfg(windows)]
pub mod native;

/// DXR shader identifier size, shader-record alignment and shader-table alignment in bytes.
pub const IDENTIFIER_BYTES: u64 = 32;
pub const RECORD_ALIGNMENT: u64 = 32;
pub const TABLE_ALIGNMENT: u64 = 64;
const MAX_RECORD_STRIDE: u64 = 4096;

/// One tightly packed shader table; absent tables have zero size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    pub offset: u64,
    pub size: u64,
    pub stride: u64,
}

/// Checked byte layout for one ray-generation record and optional miss/hit records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderTableLayout {
    pub raygen: Table,
    pub miss: Table,
    pub hit: Table,
    pub bytes: u64,
}

fn align(value: u64, alignment: u64) -> Option<u64> {
    Some(value.checked_add(alignment - 1)? & !(alignment - 1))
}

impl ShaderTableLayout {
    /// Rejects oversized local-root arguments and all arithmetic overflow before GPU allocation.
    pub fn new(local_arguments: u64, miss_count: u32, hit_count: u32) -> Option<Self> {
        let stride = align(
            IDENTIFIER_BYTES.checked_add(local_arguments)?,
            RECORD_ALIGNMENT,
        )?;
        if stride > MAX_RECORD_STRIDE {
            return None;
        }
        let raygen = Table {
            offset: 0,
            size: stride,
            stride,
        };
        let miss = Table {
            offset: align(raygen.size, TABLE_ALIGNMENT)?,
            size: stride.checked_mul(miss_count.into())?,
            stride,
        };
        let hit = Table {
            offset: align(miss.offset.checked_add(miss.size)?, TABLE_ALIGNMENT)?,
            size: stride.checked_mul(hit_count.into())?,
            stride,
        };
        let bytes = align(hit.offset.checked_add(hit.size)?, TABLE_ALIGNMENT)?;
        Some(Self {
            raygen,
            miss,
            hit,
            bytes,
        })
    }

    /// DispatchRays limits the product of width, height and depth to 2^30 rays.
    pub fn valid_dispatch(extent: [u32; 3]) -> bool {
        extent
            .into_iter()
            .try_fold(1_u64, |product, axis| {
                (axis != 0).then_some(())?;
                product.checked_mul(axis.into())
            })
            .is_some_and(|rays| rays <= (1 << 30))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dxr_tables_align_each_section_and_preserve_record_payload() {
        let layout = ShaderTableLayout::new(8, 3, 5).unwrap();
        assert_eq!(
            layout.raygen,
            Table {
                offset: 0,
                size: 64,
                stride: 64
            }
        );
        assert_eq!(
            layout.miss,
            Table {
                offset: 64,
                size: 192,
                stride: 64
            }
        );
        assert_eq!(
            layout.hit,
            Table {
                offset: 256,
                size: 320,
                stride: 64
            }
        );
        assert_eq!(layout.bytes, 576);
        let compact = ShaderTableLayout::new(0, 1, 1).unwrap();
        assert_eq!(compact.miss.offset, 64);
        assert_eq!(compact.hit.offset, 128);
        assert_eq!(compact.bytes, 192);
    }
    #[test]
    fn dxr_layout_rejects_overflow_and_oversized_records() {
        assert!(ShaderTableLayout::new(u64::MAX, 0, 0).is_none());
        assert!(ShaderTableLayout::new(4096, 0, 0).is_none());
        assert!(ShaderTableLayout::new(4096 - IDENTIFIER_BYTES, 1, 1).is_some());
    }
    #[test]
    fn dxr_dispatch_checks_all_dimensions_and_the_native_ray_limit() {
        assert!(ShaderTableLayout::valid_dispatch([1920, 1080, 1]));
        assert!(ShaderTableLayout::valid_dispatch([1 << 30, 1, 1]));
        for size in [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [u32::MAX, u32::MAX, u32::MAX],
            [1 << 30, 2, 1],
        ] {
            assert!(!ShaderTableLayout::valid_dispatch(size));
        }
    }
}
