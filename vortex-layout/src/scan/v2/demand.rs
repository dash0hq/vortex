// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads each segment of a scan's fields once, and keeps it until every chunk that reads it has.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::LayoutChildType;
use crate::LayoutRef;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

type SharedRead = Shared<BoxFuture<'static, SharedVortexResult<BufferHandle>>>;

/// A segment stored in the rows of a scan's fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FieldSegment {
    pub(super) id: SegmentId,
    /// The rows of the file the segment stores.
    pub(super) rows: Range<u64>,
    /// The chunks of the scan that overlap `rows`, each of which reads the segment once.
    pub(super) chunks: u32,
}

struct Demand {
    /// The segment's read, once requested.
    read: Option<SharedRead>,
    /// The chunks that have not yet taken the segment.
    remaining: u32,
}

/// Serves each segment of a scan's fields from one read, shared by the chunks that overlap it.
///
/// A V2 scan reads through one IO session per chunk, so a segment longer than a chunk is requested
/// by every chunk it overlaps, often after the previous chunk's read has finished. This source
/// keeps the read until the last of those chunks has taken it.
///
/// With `register`, every segment is requested up front in row order. The file's segment source
/// queues those requests without reading them, so each read it coalesces takes in neighbouring
/// segments of the scan rather than bytes no chunk asked for. Only scans that read every chunk
/// should register: a filter or selection may skip chunks, whose segments would then be read for
/// nothing. Segments of a skipped chunk stay held until the scan is dropped.
///
/// Requests for other segments, such as zone maps and dictionary values, go straight to `inner`.
pub(super) struct DemandSegmentSource {
    inner: Arc<dyn SegmentSource>,
    demand: Mutex<FxHashMap<SegmentId, Demand>>,
}

impl DemandSegmentSource {
    pub(super) fn new(
        inner: Arc<dyn SegmentSource>,
        mut segments: Vec<FieldSegment>,
        register: bool,
    ) -> Self {
        segments.sort_by_key(|segment| (segment.rows.start, segment.id));
        let demand = segments
            .into_iter()
            .filter(|segment| segment.chunks > 0)
            .map(|segment| {
                let demand = Demand {
                    read: register.then(|| read(&inner, segment.id)),
                    remaining: segment.chunks,
                };
                (segment.id, demand)
            })
            .collect();
        Self {
            inner,
            demand: Mutex::new(demand),
        }
    }
}

impl SegmentSource for DemandSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let mut demand = self.demand.lock();
        let Some(entry) = demand.get_mut(&id) else {
            drop(demand);
            return self.inner.request(id);
        };
        let shared = entry
            .read
            .get_or_insert_with(|| read(&self.inner, id))
            .clone();
        entry.remaining -= 1;
        if entry.remaining == 0 {
            demand.remove(&id);
        }
        shared.map_err(VortexError::from).boxed()
    }
}

fn read(inner: &Arc<dyn SegmentSource>, id: SegmentId) -> SharedRead {
    inner.request(id).map_err(Arc::new).boxed().shared()
}

/// The segments stored in the rows of the fields `masks` select, under `root`, with how many of
/// `chunks` overlap each. `chunks` must be sorted and disjoint.
///
/// A layout whose children all live in other row domains, such as a list, is read whole by any
/// chunk that reads its rows, so its whole subtree counts as stored in its rows. Auxiliary
/// children, such as zone maps and dictionary values, are not stored in the rows.
pub(super) fn field_segments(
    root: &LayoutRef,
    masks: &[FieldMask],
    chunks: &[Range<u64>],
) -> VortexResult<Vec<FieldSegment>> {
    let selection: Vec<&[Field]> = masks
        .iter()
        .map(|mask| match mask {
            FieldMask::All => &[][..],
            FieldMask::Prefix(path) | FieldMask::Exact(path) => path.parts(),
        })
        .collect();
    let mut segments = Vec::new();
    collect(root, 0, &selection, &mut segments)?;
    Ok(segments
        .into_iter()
        .map(|(id, rows)| {
            let first = chunks.partition_point(|chunk| chunk.end <= rows.start);
            let end = chunks.partition_point(|chunk| chunk.start < rows.end);
            let chunks = u32::try_from(end.saturating_sub(first)).unwrap_or(u32::MAX);
            FieldSegment { id, rows, chunks }
        })
        .collect())
}

fn collect(
    layout: &LayoutRef,
    row_offset: u64,
    selection: &[&[Field]],
    segments: &mut Vec<(SegmentId, Range<u64>)>,
) -> VortexResult<()> {
    let whole = selection.iter().any(|path| path.is_empty());
    let children = layout.children()?;
    let child_types: Vec<_> = layout.child_types().collect();
    let row_space_children = child_types
        .iter()
        .any(|child_type| !matches!(child_type, LayoutChildType::Auxiliary(_)));
    if whole && !row_space_children {
        let rows = row_offset..row_offset + layout.row_count();
        return subtree_segments(layout, &rows, segments);
    }
    for (child, child_type) in children.iter().zip(child_types) {
        match child_type {
            LayoutChildType::Auxiliary(_) => {}
            LayoutChildType::Chunk((_, offset)) => {
                collect(child, row_offset + offset, selection, segments)?
            }
            LayoutChildType::Transparent(_) => collect(child, row_offset, selection, segments)?,
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
                    collect(child, row_offset, &child_selection, segments)?;
                }
            }
        }
    }
    Ok(())
}

fn subtree_segments(
    layout: &LayoutRef,
    rows: &Range<u64>,
    segments: &mut Vec<(SegmentId, Range<u64>)>,
) -> VortexResult<()> {
    segments.extend(
        layout
            .segment_ids()
            .into_iter()
            .map(|id| (id, rows.clone())),
    );
    for child in layout.children()? {
        subtree_segments(&child, rows, segments)?;
    }
    Ok(())
}
