from datetime import timedelta
from feast import FileSource

# The Prudential dataset does not have a real timestamp column.
# In a real-world scenario, this would be the application submission time.
# We'll use a placeholder for now, but Feast requires it to be defined.
applicant_source = FileSource(
    name="applicant_data_source",
    path="feature_repo/data/train.parquet", # We will create this parquet file in the next step
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="A table of historical applicant data for risk assessment",
)

