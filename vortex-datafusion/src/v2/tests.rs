// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use anyhow::anyhow;
use arrow_array::ArrayRef;
use arrow_array::Int32Array;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::extension::Json as ArrowJson;
use arrow_schema::extension::Uuid as ArrowUuid;
use datafusion::assert_batches_eq;
use datafusion::prelude::SessionContext;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::compute::concat_batches;
use datafusion_common::config::ConfigOptions;
use datafusion_datasource::source::DataSource;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex::buffer::ByteBufferMut;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;

use super::VortexDataSource;

async fn vortex_source() -> anyhow::Result<VortexDataSource> {
    let batch =
        arrow_array::record_batch!(("a", Int32, vec![1, 2, 3]), ("b", Int32, vec![30, 10, 20]))?;
    vortex_source_from_batch(batch).await
}

async fn vortex_source_from_batch(batch: RecordBatch) -> anyhow::Result<VortexDataSource> {
    vortex_source_from_batch_with_session(batch, &VortexSession::default()).await
}

async fn vortex_source_from_batch_with_session(
    batch: RecordBatch,
    session: &VortexSession,
) -> anyhow::Result<VortexDataSource> {
    let schema = batch.schema();
    let array = session.arrow().from_arrow_record_batch(batch, &schema)?;
    let mut buffer = ByteBufferMut::empty();
    session
        .write_options()
        .write(&mut buffer, array.to_array_stream())
        .await?;
    let file = session.open_options().open_buffer(buffer)?;
    Ok(
        VortexDataSource::builder(file.data_source()?, session.clone())
            .with_arrow_schema(schema)
            .build()
            .await?,
    )
}

#[tokio::test]
async fn projection_filters_use_scan_schema() -> anyhow::Result<()> {
    let source = vortex_source().await?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("b", 1)), "b"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                Operator::Modulo,
                Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
            )),
            "odd",
        ),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let supported: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("b", 0)),
        Operator::Gt,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(15)))),
    ));
    let unsupported: Arc<dyn PhysicalExpr> = Arc::new(df_expr::BinaryExpr::new(
        Arc::new(df_expr::Column::new("odd", 1)),
        Operator::Eq,
        Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(1)))),
    ));

    let pushed =
        projected.try_pushdown_filters(vec![supported, unsupported], &ConfigOptions::new())?;
    assert!(matches!(
        pushed.filters.as_slice(),
        [PushedDown::Yes, PushedDown::No]
    ));
    let updated = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not added to the source"))?;
    let plan = Arc::new(DataSourceExec::new(updated)) as Arc<dyn ExecutionPlan>;
    let ctx = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;

    assert_batches_eq!(
        [
            "+----+-----+",
            "| b  | odd |",
            "+----+-----+",
            "| 30 | 1   |",
            "| 20 | 1   |",
            "+----+-----+",
        ],
        &batches
    );
    Ok(())
}

#[rstest]
#[tokio::test]
async fn projection_preserves_arrow_types(
    #[values(false, true)] fallback: bool,
) -> anyhow::Result<()> {
    let batch = arrow_array::record_batch!(
        ("a", Int32, vec![1, 2, 3]),
        ("s", Utf8, vec![Some("one"), None, Some("three")])
    )?;
    let source = vortex_source_from_batch(batch).await?;
    let projection = ProjectionExprs::from(vec![
        ProjectionExpr::new(Arc::new(df_expr::Column::new("s", 1)), "text"),
        ProjectionExpr::new(
            Arc::new(df_expr::BinaryExpr::new(
                Arc::new(df_expr::Column::new("a", 0)),
                if fallback {
                    Operator::Modulo
                } else {
                    Operator::Plus
                },
                Arc::new(df_expr::Literal::new(ScalarValue::Int32(Some(2)))),
            )),
            "n",
        ),
    ]);
    let projected = source
        .try_swapping_with_projection(&projection)?
        .ok_or_else(|| anyhow!("projection was not swapped"))?;
    let plan = Arc::new(DataSourceExec::new(projected)) as Arc<dyn ExecutionPlan>;
    let expected_schema = plan.schema();
    let ctx = SessionContext::new();
    let batches = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    for batch in &batches {
        assert_eq!(batch.schema(), expected_schema);
    }
    let expected = if fallback {
        arrow_array::record_batch!(
            ("text", Utf8, vec![Some("one"), None, Some("three")]),
            ("n", Int32, vec![1, 0, 1])
        )?
    } else {
        arrow_array::record_batch!(
            ("text", Utf8, vec![Some("one"), None, Some("three")]),
            ("n", Int32, vec![3, 4, 5])
        )?
    };
    let actual = concat_batches(&expected_schema, &batches)?;
    assert_eq!(actual.columns(), expected.columns());
    Ok(())
}

/// Extension dtypes covered by filter tests.
enum ExtensionFilterType {
    Uuid,
    Json,
}

/// Input values and Arrow metadata shared by the extension filter tests.
struct ExtensionFilterFixture {
    session: VortexSession,
    batch: RecordBatch,
    /// Storage-typed literals in batch order, with a null in the final row.
    values: Vec<ScalarValue>,
}

impl ExtensionFilterFixture {
    fn new(extension_type: ExtensionFilterType) -> anyhow::Result<Self> {
        let session = VortexSession::default();
        vortex_json::initialize(&session);

        let (field, values) = match extension_type {
            ExtensionFilterType::Uuid => {
                let mut field = Field::new("value", DataType::FixedSizeBinary(16), true);
                field.try_with_extension_type(ArrowUuid)?;

                (
                    field,
                    vec![
                        ScalarValue::FixedSizeBinary(16, Some(b"0123456789abcde1".to_vec())), //
                        ScalarValue::FixedSizeBinary(16, Some(b"0123456789abcde2".to_vec())), //
                        ScalarValue::FixedSizeBinary(16, Some(b"0123456789abcde3".to_vec())), //
                        ScalarValue::FixedSizeBinary(16, None),                               //
                    ],
                )
            }
            ExtensionFilterType::Json => {
                let mut field = Field::new("value", DataType::Utf8, true);
                field.try_with_extension_type(ArrowJson::default())?;

                (
                    field,
                    vec![
                        ScalarValue::Utf8(Some(r#"{"x":1}"#.into())), //
                        ScalarValue::Utf8(Some(r#"{"x":2}"#.into())), //
                        ScalarValue::Utf8(Some(r#"{"x":3}"#.into())), //
                        ScalarValue::Utf8(None),                      //
                    ],
                )
            }
        };

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                field,
            ])),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
                ScalarValue::iter_to_array(values.iter().cloned())?,
            ],
        )?;

        Ok(Self {
            session,
            batch,
            values,
        })
    }

    fn column(&self) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Column::new(self.batch.schema().field(1).name(), 1))
    }

    fn literal(&self, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(df_expr::Literal::new(self.values[index].clone()))
    }

    fn membership(&self) -> anyhow::Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(df_expr::InListExpr::try_new(
            self.column(),
            vec![self.literal(1), self.literal(1)],
            false,
            &self.batch.schema(),
        )?))
    }

    async fn source(&self) -> anyhow::Result<VortexDataSource> {
        vortex_source_from_batch_with_session(self.batch.clone(), &self.session).await
    }
}

async fn scan_ids(plan: Arc<dyn ExecutionPlan>) -> anyhow::Result<Vec<i32>> {
    let batches = datafusion_physical_plan::collect(plan, SessionContext::new().task_ctx()).await?;

    Ok(batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
        })
        .collect())
}

#[rstest]
#[case::uuid(ExtensionFilterType::Uuid)]
#[case::json(ExtensionFilterType::Json)]
#[tokio::test]
async fn test_extension_comparison_literals_are_pushed_down(
    #[case] extension_type: ExtensionFilterType,
    #[values(false, true)] reverse_operands: bool,
) -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(extension_type)?;
    let (left, right) = if reverse_operands {
        (fixture.literal(1), fixture.column())
    } else {
        (fixture.column(), fixture.literal(1))
    };

    let filter = Arc::new(df_expr::BinaryExpr::new(left, Operator::Eq, right));
    let pushed = fixture
        .source()
        .await?
        .try_pushdown_filters(vec![filter], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));
    let source = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not pushed"))?;

    assert_eq!(scan_ids(Arc::new(DataSourceExec::new(source))).await?, [2]);

    Ok(())
}

#[rstest]
#[case::uuid(ExtensionFilterType::Uuid)]
#[case::json(ExtensionFilterType::Json)]
#[tokio::test]
async fn test_extension_membership_literals_are_pushed_down(
    #[case] extension_type: ExtensionFilterType,
) -> anyhow::Result<()> {
    let fixture = ExtensionFilterFixture::new(extension_type)?;
    let pushed = fixture
        .source()
        .await?
        .try_pushdown_filters(vec![fixture.membership()?], &ConfigOptions::new())?;
    assert!(matches!(pushed.filters.as_slice(), [PushedDown::Yes]));
    let source = pushed
        .updated_node
        .ok_or_else(|| anyhow!("filter was not pushed"))?;

    assert_eq!(scan_ids(Arc::new(DataSourceExec::new(source))).await?, [2]);

    Ok(())
}
