import pandas as pd
# passangerId : A unique identifier assigned to each passanger.
# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display PassengerId column
print(df["PassengerId"])

# Check if PassengerId is unique
print("Is PassengerId unique?", df["PassengerId"].is_unique)

# Total number of passengers
print("Total Passengers:", df["PassengerId"].count())

# Check for duplicate PassengerIds
print("Duplicate PassengerIds:", df["PassengerId"].duplicated().sum())

# Survived: Survival status (0 = No, 1 = Yes).

# Display Survived column
print(df["Survived"])

# Count passengers by survival status
print(df["Survived"].value_counts())

# Count with labels
print(df["Survived"].value_counts().rename(index={0: "Not Survived", 1: "Survived"}))

# Survival rate
survival_rate = (df["Survived"].mean()) * 100
print(f"Survival Rate: {survival_rate:.2f}%")

# Number of survivors and non-survivors
survived = df[df["Survived"] == 1].shape[0]
not_survived = df[df["Survived"] == 0].shape[0]

print("Survived:", survived)
print("Not Survived:", not_survived)

# Pclass: Passenger’s ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)

# Display Pclass column
print(df["Pclass"])

# Count passengers in each class
print(df["Pclass"].value_counts().sort_index())

# Percentage of passengers in each class
print((df["Pclass"].value_counts(normalize=True) * 100).sort_index())

# Survival count by passenger class
print(pd.crosstab(df["Pclass"], df["Survived"]))

# Average fare for each passenger class
print(df.groupby("Pclass")["Fare"].mean())

# Name: The full name of the passenger.

# Display Name column
print(df["Name"])

# Total number of passenger names
print("Total Passengers:", df["Name"].count())

# Check for duplicate names
print("Duplicate Names:", df["Name"].duplicated().sum())

# Display first 5 passenger names
print(df["Name"].head())

# Age: The passenger's age in years. It is fractional if less than 1 and may contain estimated values (often ending in .5).
# Display the Age column
print(df["Age"])

# Check data type
print("\nData Type:")
print(df["Age"].dtype)

# Total passengers with recorded age
print("\nPassengers with Age Available:")
print(df["Age"].count())

# Missing Age values
print("\nMissing Age Values:")
print(df["Age"].isnull().sum())

# Basic statistics
print("\nAge Statistics:")
print(df["Age"].describe())

# Youngest and oldest passenger
print("\nYoungest Passenger:", df["Age"].min())
print("Oldest Passenger:", df["Age"].max())

# Average and Median Age
print("\nAverage Age:", round(df["Age"].mean(), 2))
print("Median Age:", df["Age"].median())

# Number of children (Age < 18)
children = df[df["Age"] < 18].shape[0]
print("\nChildren (Age < 18):", children)

# Number of adults (Age >= 18)
adults = df[df["Age"] >= 18].shape[0]
print("Adults (Age >= 18):", adults)

# Age grouped by survival
print("\nAverage Age by Survival:")
print(df.groupby("Survived")["Age"].mean())

# Age grouped by passenger class
print("\nAverage Age by Passenger Class:")
print(df.groupby("Pclass")["Age"].mean())

# SibSp: The number of siblings and spouses the passenger had aboard the Titanic.

# Display the SibSp column
print(df["SibSp"])

# Check data type
print("\nData Type:")
print(df["SibSp"].dtype)

# Count passengers by SibSp value
print("\nPassengers by SibSp:")
print(df["SibSp"].value_counts().sort_index())

# Basic statistics
print("\nSibSp Statistics:")
print(df["SibSp"].describe())

# Passengers traveling alone (SibSp = 0)
alone = df[df["SibSp"] == 0].shape[0]
print("\nPassengers Traveling Alone:", alone)

# Passengers traveling with siblings/spouse
with_family = df[df["SibSp"] > 0].shape[0]
print("Passengers Traveling with Siblings/Spouse:", with_family)

# Average SibSp by survival
print("\nAverage SibSp by Survival:")
print(df.groupby("Survived")["SibSp"].mean())

# Survival count by SibSp
print("\nSurvival Count by SibSp:")
print(pd.crosstab(df["SibSp"], df["Survived"]))

# Parch: The number of parents and children the passenger had aboard the Titanic.

# Display the Parch column
print(df["Parch"])

# Check data type
print("\nData Type:")
print(df["Parch"].dtype)

# Count passengers by Parch value
print("\nPassengers by Parch:")
print(df["Parch"].value_counts().sort_index())

# Basic statistics
print("\nParch Statistics:")
print(df["Parch"].describe())

# Passengers traveling without parents/children
no_family = df[df["Parch"] == 0].shape[0]
print("\nPassengers with No Parents/Children:", no_family)

# Passengers traveling with parents/children
with_family = df[df["Parch"] > 0].shape[0]
print("Passengers Traveling with Parents/Children:", with_family)

# Average Parch by survival status
print("\nAverage Parch by Survival:")
print(df.groupby("Survived")["Parch"].mean())

# Survival count by Parch
print("\nSurvival Count by Parch:")
print(pd.crosstab(df["Parch"], df["Survived"]))

# Average Parch by passenger class
print("\nAverage Parch by Passenger Class:")
print(df.groupby("Pclass")["Parch"].mean())

# Ticket: The ticket number assigned to the passenger.
# Display the Ticket column
print(df["Ticket"])

# Check data type
print("\nData Type:")
print(df["Ticket"].dtype)

# Total number of ticket records
print("\nTotal Ticket Records:")
print(df["Ticket"].count())

# Number of unique ticket numbers
print("\nUnique Ticket Numbers:")
print(df["Ticket"].nunique())

# Check duplicate ticket numbers
print("\nDuplicate Ticket Numbers:")
print(df["Ticket"].duplicated().sum())

# Display duplicate tickets (shared by multiple passengers)
print("\nPassengers Sharing the Same Ticket:")
print(df[df.duplicated("Ticket", keep=False)]
      .sort_values("Ticket")[["PassengerId", "Name", "Ticket"]])

# Display first 10 ticket numbers
print("\nFirst 10 Ticket Numbers:")
print(df["Ticket"].head(10))

#  Fare: The amount of money paid for the ticket.

# Display the Fare column
print(df["Fare"])

# Check data type
print("\nData Type:")
print(df["Fare"].dtype)

# Basic statistics
print("\nFare Statistics:")
print(df["Fare"].describe())

# Minimum and Maximum Fare
print("\nMinimum Fare:", df["Fare"].min())
print("Maximum Fare:", df["Fare"].max())

# Average and Median Fare
print("\nAverage Fare:", round(df["Fare"].mean(), 2))
print("Median Fare:", df["Fare"].median())

# Number of passengers with zero fare
zero_fare = df[df["Fare"] == 0].shape[0]
print("\nPassengers with Zero Fare:", zero_fare)

# Average fare by passenger class
print("\nAverage Fare by Passenger Class:")
print(df.groupby("Pclass")["Fare"].mean())

# Average fare by survival status
print("\nAverage Fare by Survival:")
print(df.groupby("Survived")["Fare"].mean())

# Top 10 highest fares
print("\nTop 10 Highest Fares:")
print(df.nlargest(10, "Fare")[["PassengerId", "Name", "Pclass", "Fare"]])

#  Cabin: The cabin number occupied by the passenger (contains many missing values).
# Display the Cabin column
print(df["Cabin"])

# Check data type
print("\nData Type:")
print(df["Cabin"].dtype)

# Total number of records
print("\nTotal Passengers:", len(df))

# Number of missing cabin values
print("\nMissing Cabin Values:")
print(df["Cabin"].isnull().sum())

# Number of available cabin values
print("\nAvailable Cabin Values:")
print(df["Cabin"].notnull().sum())

# Percentage of missing values
missing_percent = (df["Cabin"].isnull().sum() / len(df)) * 100
print(f"\nMissing Percentage: {missing_percent:.2f}%")

# Number of unique cabin numbers
print("\nUnique Cabin Numbers:")
print(df["Cabin"].nunique())

# Display first 10 non-missing cabin values
print("\nSample Cabin Numbers:")
print(df["Cabin"].dropna().head(10))

# Survival based on Cabin availability
df["Cabin_Available"] = df["Cabin"].notnull()

print("\nSurvival Based on Cabin Availability:")
print(pd.crosstab(df["Cabin_Available"], df["Survived"]))

#  Embarked: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton). 

# Display the Embarked column
print(df["Embarked"])

# Check data type
print("\nData Type:")
print(df["Embarked"].dtype)

# Count passengers from each port
print("\nPassengers by Embarkation Port:")
print(df["Embarked"].value_counts())

# Count missing values
print("\nMissing Values:")
print(df["Embarked"].isnull().sum())

# Percentage of passengers from each port
print("\nPercentage by Port:")
print((df["Embarked"].value_counts(normalize=True) * 100).round(2))

# Survival count by embarkation port
print("\nSurvival Count by Embarkation Port:")
print(pd.crosstab(df["Embarked"], df["Survived"]))

# Average fare by embarkation port
print("\nAverage Fare by Embarkation Port:")
print(df.groupby("Embarked")["Fare"].mean())

# Average age by embarkation port
print("\nAverage Age by Embarkation Port:")
print(df.groupby("Embarked")["Age"].mean())