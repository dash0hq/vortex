// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Masking run-end encoded arrays.
//!
//! The output stays run-end encoded. Runs split where the mask changes, and a split run that the
//! mask hides takes a null value index, so the values array is only gathered, never decoded:
//!
//! ```text
//! Input rows:   A A A | B B
//! Mask:         1 0 1 | 1 1
//! Output ends:  1 2 | 5
//! Value index:  0 null | 0 1   (adjacent hidden runs merge)
//! ```
//!
//! This costs one pass over the mask's true ranges and the runs, independent of the row count.

use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::runend::iter::slice_relative_ends;
use crate::builtins::ArrayBuiltins;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::mask::MaskKernel;
use crate::validity::Validity;

impl MaskKernel for RunEnd {
    fn mask(
        array: ArrayView<'_, Self>,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let nullable = array.dtype().as_nullable();
        let keep = Validity::Array(mask.clone()).execute_mask(array.len(), ctx)?;
        let true_ranges = match keep.slices() {
            AllOr::All => {
                // SAFETY: only the values' nullability changes.
                return Ok(Some(
                    unsafe {
                        RunEnd::new_unchecked(
                            array.ends().clone(),
                            array.values().cast(nullable)?,
                            array.offset(),
                            array.len(),
                        )
                    }
                    .into_array(),
                ));
            }
            AllOr::None => {
                return Ok(Some(
                    ConstantArray::new(Scalar::null(nullable), array.len()).into_array(),
                ));
            }
            AllOr::Some(ranges) => ranges,
        };

        let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
        let (new_ends, value_indices, kept) =
            match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
                split_runs(
                    slice_relative_ends(ends.as_slice::<E>(), array.offset(), array.len()),
                    true_ranges,
                    array.len(),
                )
            });

        let indices = PrimitiveArray::new(value_indices.freeze(), Validity::from_iter(kept));
        let values = array.values().take(indices.into_array())?;
        // SAFETY: the new ends are strictly increasing, the last is `len`, and there is one value
        // per end.
        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    PrimitiveArray::new(new_ends.freeze(), Validity::NonNullable).into_array(),
                    values,
                    0,
                    array.len(),
                )
            }
            .into_array(),
        ))
    }
}

/// Splits runs at the boundaries of `true_ranges`, returning the new run ends, the value index of
/// each new run, and whether the mask keeps it. Adjacent hidden runs merge into one.
fn split_runs(
    relative_ends: impl Iterator<Item = usize>,
    true_ranges: &[(usize, usize)],
    len: usize,
) -> (BufferMut<u64>, BufferMut<u64>, Vec<bool>) {
    let mut new_ends = BufferMut::<u64>::empty();
    let mut value_indices = BufferMut::<u64>::empty();
    let mut kept = Vec::new();
    let mut ranges = true_ranges.iter().peekable();
    let mut pos = 0;

    for (run, run_end) in relative_ends.enumerate() {
        while pos < run_end {
            while ranges.next_if(|&&(_, end)| end <= pos).is_some() {}
            let (segment_end, keep) = match ranges.peek() {
                Some(&&(start, end)) if start <= pos => (end.min(run_end), true),
                Some(&&(start, _)) => (start.min(run_end), false),
                None => (run_end, false),
            };
            if !keep
                && kept.last() == Some(&false)
                && let Some(last_end) = new_ends.last_mut()
            {
                *last_end = segment_end as u64;
            } else {
                new_ends.push(segment_end as u64);
                value_indices.push(run as u64);
                kept.push(keep);
            }
            pos = segment_end;
        }
        if pos == len {
            break;
        }
    }
    (new_ends, value_indices, kept)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::runend::RunEnd;
    use crate::arrays::runend::array::RunEndArraySlotsExt;
    use crate::arrays::runend::tests::SESSION;
    use crate::assert_arrays_eq;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::scalar_fn::fns::mask::Mask as MaskExpr;

    fn sliced_runend(values: ArrayRef) -> ArrayRef {
        // Logical rows after the offset of 1: v0 v0 v1 v2 v2 v3 v3.
        RunEnd::try_new_offset_length(
            buffer![3u32, 4, 6, 9].into_array(),
            values,
            1,
            7,
            &mut SESSION.create_execution_ctx(),
        )
        .unwrap()
        .into_array()
    }

    #[rstest]
    #[case::primitive(sliced_runend(buffer![1i32, 2, 3, 4].into_array()))]
    #[case::nullable(sliced_runend(
        PrimitiveArray::from_option_iter([Some(1i64), None, Some(3), Some(4)]).into_array()
    ))]
    #[case::dict(sliced_runend(
        DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinViewArray::from_iter_nullable_str([Some("a"), Some("b"), None]).into_array(),
        )
        .unwrap()
        .into_array()
    ))]
    fn test_mask_runend_conformance(#[case] array: ArrayRef) {
        test_mask_conformance(&array, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn mask_splits_runs_and_merges_hidden_ones() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = sliced_runend(buffer![1i32, 2, 3, 4].into_array());
        let mask = BoolArray::from_iter([true, false, false, false, true, true, true]);
        let masked = MaskExpr::try_new(array, mask.into_array())?
            .into_array()
            .execute::<ArrayRef>(&mut ctx)?;

        let masked = masked.as_::<RunEnd>();
        assert_arrays_eq!(
            masked.ends().clone(),
            PrimitiveArray::from_iter([1u64, 4, 5, 7]),
            &mut ctx
        );
        assert_arrays_eq!(
            masked.values().clone(),
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn mask_of_a_slice_stays_run_end() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = sliced_runend(buffer![1i32, 2, 3, 4].into_array()).slice(1..6)?;
        let mask = BoolArray::from_iter([true, true, false, true, true]);
        let masked = MaskExpr::try_new(array, mask.into_array())?
            .into_array()
            .execute_until::<RunEnd>(&mut ctx)?;

        assert!(masked.is::<RunEnd>());
        assert_arrays_eq!(
            masked,
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3), Some(4)]),
            &mut ctx
        );
        Ok(())
    }
}
