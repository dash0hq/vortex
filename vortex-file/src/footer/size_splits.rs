// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_layout::LayoutChildType;
use vortex_layout::LayoutRef;

use crate::footer::SegmentSpec;

/// A chunk of a selected field: the rows it covers and the bytes of its segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Chunk {
    start: u64,
    end: u64,
    bytes: u64,
}

/// Row boundaries that cut `0..row_count` into splits of at most `target_bytes` each.
///
/// A split reads every chunk of the selected fields that it overlaps in full, so a split's size is
/// the bytes of those chunks. Splits end only at chunk boundaries, at the farthest one that keeps
/// the split within `target_bytes`. Where even the nearest boundary exceeds the target, the split
/// ends there anyway.
pub(crate) fn size_splits(
    root: &LayoutRef,
    segments: &[SegmentSpec],
    row_count: u64,
    field_mask: &[FieldMask],
    target_bytes: u64,
) -> VortexResult<Vec<u64>> {
    let selection: Vec<&[Field]> = field_mask
        .iter()
        .map(|mask| match mask {
            FieldMask::All => &[][..],
            FieldMask::Prefix(path) | FieldMask::Exact(path) => path.parts(),
        })
        .collect();
    let mut chunks = Vec::new();
    collect_chunks(root, 0, &selection, segments, &mut chunks)?;
    split_by_size(chunks, row_count, target_bytes)
}

fn split_by_size(
    mut chunks: Vec<Chunk>,
    row_count: u64,
    target_bytes: u64,
) -> VortexResult<Vec<u64>> {
    chunks.sort_unstable();

    let mut candidates: Vec<u64> = chunks
        .iter()
        .flat_map(|chunk| [chunk.start, chunk.end])
        .chain([row_count])
        .filter(|&row| row > 0 && row <= row_count)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();

    let mut boundaries = vec![0];
    let mut start = 0;
    while start < row_count {
        let mut bytes: u64 = chunks
            .iter()
            .take_while(|chunk| chunk.start <= start)
            .filter(|chunk| chunk.end > start)
            .map(|chunk| chunk.bytes)
            .sum();
        let mut joining = chunks.partition_point(|chunk| chunk.start <= start);
        let mut end = None;
        for &candidate in &candidates[candidates.partition_point(|&row| row <= start)..] {
            while let Some(chunk) = chunks.get(joining)
                && chunk.start < candidate
            {
                bytes += chunk.bytes;
                joining += 1;
            }
            if bytes > target_bytes && end.is_some() {
                break;
            }
            end = Some(candidate);
            if bytes > target_bytes {
                break;
            }
        }
        start = end.ok_or_else(|| vortex_err!("no split boundary after row {start}"))?;
        boundaries.push(start);
    }
    Ok(boundaries)
}

/// Collects the chunks of `layout` under the field paths in `selection`, where an empty path
/// selects the whole subtree.
fn collect_chunks(
    layout: &LayoutRef,
    row_offset: u64,
    selection: &[&[Field]],
    segments: &[SegmentSpec],
    chunks: &mut Vec<Chunk>,
) -> VortexResult<()> {
    let whole = selection.iter().any(|path| path.is_empty());
    let children = layout.children()?;
    let child_types: Vec<_> = layout.child_types().collect();
    let row_space_children = child_types
        .iter()
        .any(|child_type| !matches!(child_type, LayoutChildType::Auxiliary(_)));

    if whole && !row_space_children {
        // Leaves, and layouts such as lists whose children live in another row space: the scan
        // reads the subtree whenever it reads any of its rows.
        let bytes = subtree_bytes(layout, segments)?;
        if bytes > 0 && layout.row_count() > 0 {
            chunks.push(Chunk {
                start: row_offset,
                end: row_offset + layout.row_count(),
                bytes,
            });
        }
        return Ok(());
    }

    for (child, child_type) in children.iter().zip(child_types) {
        match child_type {
            LayoutChildType::Auxiliary(_) => {}
            LayoutChildType::Chunk((_, offset)) => {
                collect_chunks(child, row_offset + offset, selection, segments, chunks)?
            }
            LayoutChildType::Transparent(_) => {
                collect_chunks(child, row_offset, selection, segments, chunks)?
            }
            LayoutChildType::Field(name) => {
                let field = Field::from(name);
                let child_selection: Vec<&[Field]> = if whole {
                    vec![&[]]
                } else {
                    selection
                        .iter()
                        .filter_map(|path| path.split_first())
                        .filter(|(first, _)| **first == field)
                        .map(|(_, rest)| rest)
                        .collect()
                };
                if !child_selection.is_empty() {
                    collect_chunks(child, row_offset, &child_selection, segments, chunks)?;
                }
            }
        }
    }
    Ok(())
}

fn subtree_bytes(layout: &LayoutRef, segments: &[SegmentSpec]) -> VortexResult<u64> {
    let own = layout
        .segment_ids()
        .iter()
        .map(|id| {
            segments
                .get(**id as usize)
                .map(|segment| u64::from(segment.length))
                .ok_or_else(|| vortex_err!("layout references missing segment {}", **id))
        })
        .sum::<VortexResult<u64>>()?;
    layout.children()?.iter().try_fold(own, |bytes, child| {
        Ok(bytes + subtree_bytes(child, segments)?)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::FieldMask;
    use vortex_array::dtype::FieldPath;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_io::session::RuntimeSession;
    use vortex_layout::scan::split_by::SplitBy;
    use vortex_layout::session::LayoutSession;
    use vortex_session::VortexSession;

    use super::Chunk;
    use super::split_by_size;
    use crate::OpenOptionsSessionExt;
    use crate::WriteOptionsSessionExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty()
            .with::<LayoutSession>()
            .with::<RuntimeSession>();
        crate::register_default_encodings(&session);
        crate::enable_all_registered_array_encodings(&session);
        session
    });

    fn chunk(start: u64, end: u64, bytes: u64) -> Chunk {
        Chunk { start, end, bytes }
    }

    /// One 100-row chunk of 50 bytes beside ten 10-row chunks of 10 bytes.
    fn long_and_short_chunks() -> Vec<Chunk> {
        let mut chunks = vec![chunk(0, 100, 50)];
        chunks.extend((0..10).map(|i| chunk(i * 10, i * 10 + 10, 10)));
        chunks
    }

    #[test]
    fn splits_take_the_farthest_boundary_within_the_target() -> VortexResult<()> {
        // Every split holds the long chunk, so 70 bytes leaves room for two short chunks.
        assert_eq!(
            split_by_size(long_and_short_chunks(), 100, 70)?,
            [0, 20, 40, 60, 80, 100]
        );
        assert_eq!(split_by_size(long_and_short_chunks(), 100, 150)?, [0, 100]);
        Ok(())
    }

    #[test]
    fn a_target_below_one_boundary_step_splits_at_every_boundary() -> VortexResult<()> {
        assert_eq!(
            split_by_size(long_and_short_chunks(), 100, 10)?,
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        );
        Ok(())
    }

    #[test]
    fn splits_end_only_at_chunk_boundaries() -> VortexResult<()> {
        let chunks = vec![chunk(0, 64, 30), chunk(0, 40, 30), chunk(40, 100, 30)];
        assert_eq!(split_by_size(chunks.clone(), 100, 60)?, [0, 40, 100]);
        assert_eq!(split_by_size(chunks, 100, 90)?, [0, 100]);
        Ok(())
    }

    #[tokio::test]
    async fn the_walk_finds_the_chunks_the_layout_splits_on() -> VortexResult<()> {
        let rows = 120_000usize;
        let file = StructArray::from_fields(&[
            (
                "small",
                PrimitiveArray::from_iter((0..rows).map(|row| (row % 7) as u8)).into_array(),
            ),
            (
                "big",
                VarBinViewArray::from_iter_str((0..rows).map(|row| format!("{row:0100}")))
                    .into_array(),
            ),
        ])?;
        let mut buf = ByteBufferMut::empty();
        SESSION
            .write_options()
            .write(&mut buf, file.into_array().to_array_stream())
            .await?;
        let file = SESSION.open_options().open_buffer(buf)?;
        let reader = file.layout_reader()?;
        let rows = file.row_count();

        let all = [FieldMask::All];
        let small = [FieldMask::Prefix(FieldPath::from_name("small"))];
        let big = [FieldMask::Prefix(FieldPath::from_name("big"))];
        for mask in [&all[..], &small, &big] {
            // With no room in any split, every chunk boundary is a split boundary.
            assert_eq!(
                file.footer().size_splits(mask, 0)?,
                SplitBy::Layout.splits(reader.as_ref(), &(0..rows), mask)?,
                "{mask:?}"
            );
            assert_eq!(file.footer().size_splits(mask, u64::MAX)?, [0, rows]);
        }
        assert_eq!(file.footer().size_splits(&small, 0)?, [0, rows]);
        assert!(file.footer().size_splits(&big, 0)?.len() > 3);
        Ok(())
    }
}
