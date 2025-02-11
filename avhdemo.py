from avh.data_generation import DataGenerationPipeline, NormalNumericColumn, BetaNumericColumn

from avh.data_issues import IncreasedNulls
from avh.auto_validate_by_history import AVH

# To begin with, we nned to collect a history of data in a form of list of dataframes.

# It's pretty easy to model virtual data tables with our provided classes like so:
#   * columns - what colums and what type should make up the table?
#   * issues - should the natural data have any quirks?
pipeline = DataGenerationPipeline(
    columns=[
        NormalNumericColumn("money", 1000, 50),
        NormalNumericColumn("height", 180, 10),
    ],
    issues=[
        ("money", [IncreasedNulls(0.05)]),
    ],
    random_state=42
)

# In this case, we'll generate 30 "data pipeline" executions of size ~ N(10000, 30)
H = [pipeline.generate_normal(10000, 30) for i in range(30)]

# Finally, let's see what data quality constraints does the AVH generate
#   for the 'money' column of our data:
PS = AVH(columns=["money"], random_state=42).generate(H, fpr_target=0.05)

ps_money = PS["money"]
print("Contraint Rules:")
print(ps_money)

# As you can see, the AVH algorithm correctly identifies a statistical invariate
#   which in our case was data completeness ratio of 95% (as specified in our data generation).
#
#   To the algorithm, any deviation outside this narrow metric interval would appear as an
#   anomaly and thus would trigger the generated data quality constraint.
new_data = pipeline.generate_normal(1000, 30)
new_data_w_issues = IncreasedNulls(p=0.5).fit_transform(new_data)

ps_money.predict(new_data["money"])
ps_money.predict(new_data_w_issues["money"])

# Naturally, in a real-world scenario you'd like to have as complete data as possible.
# The library allows the user to make necessary adjustments if one wishes to do so:
ps_money.constraints[0].u_upper_ = 1.0
ps_money
