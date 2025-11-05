"""
NYC Taxi Data Processing Pipeline
This script processes NYC taxi trip data for fare prediction modeling using DuckDB.
"""

import duckdb


# Configuration
RAW_DATA_PATH = "data/raw/*.csv"  # or "data/raw/Yellow_Taxi_Trip_Data.csv"
DB_PATH = "taxi_data.db"


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
        CREATE TABLE
            IF NOT EXISTS raw_data AS
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
        CREATE TABLE
            IF NOT EXISTS valid_transactions AS
        SELECT
            *
        FROM
            raw_data
        WHERE
            payment_type != 0
        """
    )



    # ============================================================================
    # STEP 3: Remove invalid transactions and outliers
    # ============================================================================


    print("STEP 3: Remove invalid transactions and outliers")

    con.execute(
        """
        create table 
            if not exists cleaned_transactions as
        select
            *
        from
            valid_transactions
        where
            total_amount > 0
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
        create table
            if not exists filtered_transactions as
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

    print(con.execute(
        """
        select count(*) from filtered_transactions
        where mta_tax = 0 and DOLocationID not in (1, 265);
        """
    ).fetchall())

    con.execute(
        """
        UPDATE filtered_transactions
        SET mta_tax = 0.50, total_amount = total_amount + 0.50
        WHERE DOLocationID NOT IN (1, 265)
        AND mta_tax = 0;
        """
    )

    print(con.execute(
        """
        select count(*) from filtered_transactions
        where mta_tax = 0 and DOLocationID not in (1, 265);
        """
    ).fetchall())


    # ============================================================================
    # STEP 6: Fix improvement surcharge
    # ============================================================================


    print('STEP 6: Fix improvement surcharge')

    print(con.execute(
        """
        select count(*) from filtered_transactions
        where improvement_surcharge = 0.30;
        """
    ).fetchall())

    con.execute(
        """
        UPDATE filtered_transactions
        SET improvement_surcharge = 1.00, total_amount = total_amount + 0.70
        WHERE improvement_surcharge = 0.30;
        """
    )

    print(con.execute(
        """
        select count(*) from filtered_transactions
        where improvement_surcharge = 0.30;
        """
    ).fetchall())

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
    # TODO: Create function to extract datetime features using DuckDB date functions
    # Features to extract from tpep_pickup_datetime:
    #   - year (YEAR())
    #   - month (MONTH())
    #   - day_of_week (DAYOFWEEK() or EXTRACT(DOW))
    #   - hour (HOUR())
    #   - is_weekend (CASE WHEN day_of_week IN (6, 7) THEN 1 ELSE 0)
    #   - is_rush_hour (CASE WHEN hour IN (7,8,9,17,18,19) THEN 1 ELSE 0)






    # ============================================================================
    # STEP 11: Calculate/validate rush hour extra charge
    # ============================================================================
    # TODO: Create function to validate or calculate extra charge based on time
    # Note: This is based on pickup time and rush hours
    # Extra charges: $0.50 rush hour, $1.00 overnight (8pm-6am weekdays, 9pm-6am weekends)


    # ============================================================================
    # STEP 12: Create cleaned dataset
    # ============================================================================
    # TODO: Create function to create final cleaned table
    # Select relevant columns, exclude unpredictable ones (tolls_amount, tip_amount)
    # Keep: VendorID, pickup/dropoff datetime, locations, passenger_count, trip_distance,
    #       fare_amount, extra, mta_tax, improvement_surcharge, congestion_surcharge,
    #       airport_fee, total_amount, payment_type, temporal features
    # Exclude: tolls_amount, tip_amount (not predictable from pickup information)


    # ============================================================================
    # STEP 13: Export cleaned data to parquet
    # ============================================================================
    # TODO: Create function to export cleaned data to single parquet file
    # con.execute(f"COPY cleaned_data TO '{OUTPUT_PARQUET}' (FORMAT PARQUET, COMPRESSION ZSTD)")


    # ============================================================================
    # STEP 14: Generate data quality report
    # ============================================================================
    # TODO: Create function to generate summary statistics
    # - Row counts (before/after cleaning)
    # - Missing value counts
    # - Fare statistics (min, max, avg, median)
    # - Trip distance statistics
    # Save report for documentation


    con.close()


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
