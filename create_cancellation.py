import pandas as pd

input_file = "flights.csv"
output_file = "flight-cancellation.csv"

first = True

for chunk in pd.read_csv(input_file, chunksize=100000):

    data = pd.DataFrame({
        "date": (
            chunk["YEAR"].astype(str)
            + "-"
            + chunk["MONTH"].astype(str).str.zfill(2)
            + "-"
            + chunk["DAY"].astype(str).str.zfill(2)
        ),
        "flight_number": chunk["FLIGHT_NUMBER"].astype(str),
        "flight_identifier": chunk["AIRLINE"].astype(str),
        "cancel": chunk["CANCELLED"].astype(str),
        "reason": chunk["CANCELLATION_REASON"].astype(str)
    })

    data.to_csv(
        output_file,
        mode="w" if first else "a",
        header=first,
        index=False
    )

    first = False

print("flight-cancellation.csv created successfully!")