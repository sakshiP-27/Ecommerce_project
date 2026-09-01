from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_engineered_features(df):
    # total pages viewed
    df["TotalPages"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )

    # Avg time per page
    total_duration = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )
    df["AvgTimePerPage"] = total_duration / df["TotalPages"].replace(0, 1)

    # Product engagement ratio
    df["ProductEngagementRatio"] = (
        df["ProductRelated"] / df["TotalPages"].replace(0, 1)
    )

    # Returning visitor flag
    df["IsReturningVisitor"] = df["VisitorType"].apply(
        lambda x: 1 if x == "Returning_Visitor" else 0
    )

    logger.info("Engineered features added to the DataFrame.")
    return df
