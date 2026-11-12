// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Masking run-end encoded arrays.
//!
//! A run-end array masks as the dictionary it converts to: the mask applies to the run indices,
//! and the values stay untouched, so scalar functions above the mask still reach them.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::runend::RunEnd;
use crate::builtins::ArrayBuiltins;
use crate::scalar_fn::fns::mask::MaskKernel;

impl MaskKernel for RunEnd {
    fn mask(
        array: ArrayView<'_, Self>,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let dict = array.into_owned().to_dict(ctx)?;
        Ok(Some(dict.into_array().mask(mask.clone())?))
    }
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
    use crate::arrays::Dict;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::dict::DictArraySlotsExt;
    use crate::arrays::runend::RunEnd;
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
    fn mask_keeps_run_values_and_masks_codes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = sliced_runend(buffer![1i32, 2, 3, 4].into_array());
        let mask = BoolArray::from_iter([true, false, false, false, true, true, true]);
        let masked = MaskExpr::try_new(array, mask.into_array())?
            .into_array()
            .execute_until::<Dict>(&mut ctx)?;

        let masked = masked.as_::<Dict>();
        assert_arrays_eq!(
            masked.codes().clone(),
            PrimitiveArray::from_option_iter([
                Some(0u64),
                None,
                None,
                None,
                Some(2),
                Some(3),
                Some(3)
            ]),
            &mut ctx
        );
        assert_arrays_eq!(
            masked.values().clone(),
            PrimitiveArray::from_iter([1i32, 2, 3, 4]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn mask_of_a_slice_keeps_run_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = sliced_runend(buffer![1i32, 2, 3, 4].into_array()).slice(1..6)?;
        let mask = BoolArray::from_iter([true, true, false, true, true]);
        let masked = MaskExpr::try_new(array, mask.into_array())?
            .into_array()
            .execute_until::<Dict>(&mut ctx)?;

        assert!(masked.is::<Dict>());
        assert_arrays_eq!(
            masked,
            PrimitiveArray::from_option_iter([Some(1i32), Some(2), None, Some(3), Some(4)]),
            &mut ctx
        );
        Ok(())
    }
}
