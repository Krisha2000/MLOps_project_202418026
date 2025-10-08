from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from data_sources import applicant_source

# Define an entity for the applicant. An entity is the object that features are attached to.
applicant = Entity(
    name="applicant_id",
    join_keys=["Id"],
    description="The unique ID of an insurance applicant",
)

# This FeatureView defines a group of related features using the modern 'schema' syntax.
applicant_features_view = FeatureView(
    name="applicant_features",
    entities=[applicant],
    ttl=timedelta(days=365 * 10),  # How long to keep features in the online store
    schema=[
        Field(name="Product_Info_4", dtype=Float32),
        Field(name="Ins_Age", dtype=Float32),
        Field(name="Ht", dtype=Float32),
        Field(name="Wt", dtype=Float32),
        Field(name="BMI", dtype=Float32),
        Field(name="Employment_Info_1", dtype=Float32),
        Field(name="Employment_Info_4", dtype=Float32),
        Field(name="Employment_Info_6", dtype=Float32),
        Field(name="Insurance_History_5", dtype=Float32),
        Field(name="Family_Hist_2", dtype=Float32),
        Field(name="Family_Hist_3", dtype=Float32),
        Field(name="Family_Hist_4", dtype=Float32),
        Field(name="Family_Hist_5", dtype=Float32),
        Field(name="Medical_History_1", dtype=Int64),
        Field(name="Medical_History_10", dtype=Int64),
        Field(name="Medical_History_15", dtype=Int64),
        Field(name="Medical_History_24", dtype=Int64),
        Field(name="Medical_History_32", dtype=Int64),
        # This will be our target variable, but we include it here for training data retrieval
        Field(name="Response", dtype=Int64),
    ],
    source=applicant_source,
    online=True,
    tags={},
)