"""
NYC Taxi Data Processing Pipeline
This script processes NYC taxi trip data for fare prediction modeling using DuckDB.
"""

import duckdb
import polars as pl


# Configuration
RAW_DATA_PATH = "data/raw/*.csv"  # or "data/raw/Yellow_Taxi_Trip_Data.csv"
DB_PATH = "taxi_data.db"
OUTPUT_PARQUET = 'data/processed/cleaned_data.parquet'


def main():
    """Main pipeline execution"""

    # Initialize DuckDB connection
    con = duckdb.connect(DB_PATH)

    # ============================================================================
    # STEP 1: Load raw data into DuckDB
    # ============================================================================


    print("STEP 1: Load raw data into DuckDB")
    
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw_data AS
        SELECT
            *
        FROM
            read_csv ('{RAW_DATA_PATH}')
        """
    )


    # ============================================================================
    # STEP 2: Analyze missing data
    # ============================================================================


    print("STEP 2: Analyze missing data")

    con.execute(
        """
        CREATE OR REPLACE TABLE valid_transactions AS
        SELECT
            *
        FROM
            raw_data
        WHERE
            payment_type != 0;

        drop table raw_data;
        """
    )



    # ============================================================================
    # STEP 3: Remove invalid transactions and outliers
    # ============================================================================


    print("STEP 3: Remove invalid transactions and outliers")

    con.execute(
        """
        CREATE OR REPLACE TABLE cleaned_transactions AS
        select
            *
        from
            valid_transactions
        where
            total_amount > 0
            and fare_amount > 0
            and trip_distance > 0
            and payment_type <> 3
            and year (tpep_pickup_datetime) = 2023;

        drop table valid_transactions;
        """
    )

    # ============================================================================
    # STEP 4: Filter by RatecodeID (keep standard metered trips only)
    # ============================================================================

    print('STEP 4: Filter by RatecodeID (keep standard metered trips only)')

    con.execute(
        """
        CREATE OR REPLACE TABLE filtered_transactions AS
        select
            row_number() over (
                order by
                    tpep_pickup_datetime
            ) as id,
            *
        from
            cleaned_transactions
        where
            RatecodeID = 1;

        drop table cleaned_transactions;
        """
    )


    # ============================================================================
    # STEP 5: Fix MTA tax
    # ============================================================================


    print('STEP 5: Fix MTA tax')


    con.execute(
        """
        UPDATE filtered_transactions
        SET
            mta_tax = 0.50,
            total_amount = total_amount + 0.50
        WHERE
            DOLocationID NOT IN (1, 265)
            AND mta_tax = 0;
        """
    )

    # ============================================================================
    # STEP 6: Fix improvement surcharge
    # ============================================================================


    print('STEP 6: Fix improvement surcharge')

    con.execute(
        """
        UPDATE filtered_transactions
        SET
            improvement_surcharge = 1.00,
            total_amount = total_amount + 0.70
        WHERE
            improvement_surcharge = 0.30;
        """
    )

    # ============================================================================
    # STEP 7: Calculate congestion surcharge
    # ============================================================================


    print("STEP 7: Calculate congestion surcharge")

    con.execute(
        """
        update filtered_transactions
        set
            improvement_surcharge = 2.5,
            total_amount = total_amount + 2.5 - improvement_surcharge
        where
            improvement_surcharge <> 2.5
            and DOLocationID in (
                select
                    LocationID
                from
                    zone_lookup
                where
                    service_zone = 'Yellow Zone'
            )
        """
    )

    con.execute(
        """
        update filtered_transactions
        set
            improvement_surcharge = 2.5,
            total_amount = total_amount + 2.5 - improvement_surcharge
        where
            improvement_surcharge <> 2.5
            and PULocationID in (
                select
                    LocationID
                from
                    zone_lookup
                where
                    service_zone = 'Yellow Zone'
            )
        """
    )

    # ============================================================================
    # STEP 8: Calculate airport fee
    # ============================================================================

    print("STEP 8: Calculate airport fee")

    con.execute(
        """
        update filtered_transactions as ft
        set
            airport_fee = case
                when tpep_pickup_datetime < '2023-04-05' then 1.25
                else 1.75
            end
        from
            zone_lookup as zn
        where
            zn.LocationID = ft.PULocationID 
            and airport_fee not in (1.25, 1.75)
            and zn.service_zone = 'Airports';
        """
    )

    # ============================================================================
    # STEP 9: Extract temporal features
    # ============================================================================


    print("STEP 9: Extract temporal features")

    con.execute(
        """
        CREATE OR REPLACE TABLE cleaned_data AS
        select
            *,
            year (tpep_pickup_datetime) as travel_year,
            month (tpep_pickup_datetime) as travel_month,
            day (tpep_pickup_datetime) as travel_day,
            hour (tpep_pickup_datetime) as travel_hour,
            dayofweek (tpep_pickup_datetime) as travel_weekday,
            case
                when dayofweek (tpep_pickup_datetime) in (0, 6) then 1
                else 0
            end as is_weekend,
            case
                when hour (tpep_pickup_datetime) in (7, 8, 9, 17, 18, 19) then 1
                else 0
            end as is_rush_hour
        from
            filtered_transactions;

        drop table filtered_transactions;
        """
    )


    # ============================================================================
    # STEP 10: Export cleaned data to parquet
    # ============================================================================
    # TODO: Create function to export cleaned data to single parquet file
    con.execute(f"COPY cleaned_data TO '{OUTPUT_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute('DROP TABLE cleaned_data;')
    con.close()


    # ============================================================================
    # STEP 11: Generate data quality report
    # ============================================================================
    
    print("STEP 11: Generate data quality report")

    df = pl.read_parquet(OUTPUT_PARQUET)
    df.describe().write_csv('data_quality_report.csv')



    


# ============================================================================
# Target Variable Strategy (for ML modeling)
# ============================================================================
# Model should predict: fare_amount only (metered base fare based on distance/time)
# Post-prediction calculation:
#   predicted_total = predicted_fare_amount + mta_tax + improvement_surcharge +
#                     congestion_surcharge + airport_fee + extra
# Reason: Separates learnable patterns (fare) from deterministic rules (fixed fees)


if __name__ == "__main__":
    main()
