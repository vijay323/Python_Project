import pandas as pd  # is for data loding and manipulation.
import numpy as np  # handles numerical operarions and arra.
import matplotlib.pylab as plt   # is for potting and visualization.
from sklearn.linear_model import LinearRegression    # is for building regressio models.
from sklearn.model_selection import train_test_split  # helps gplit data for machine learning.
from sklearn.metrics import r2_score          # evalutes regression model performance.

df=None # Global variable to store loded dataset

def Load_dataset():
    global df
    path = input("Enter csv file name: ")

    try:
        df = pd.read_csv(path)  # Assign loaded dataframe to df
        print("\n Dataset Loaded Successfully")
        print("Total row:", df.shape[0])
        print("Total column:", df.shape[1])
        print("Column Names:", list(df.columns))
        return df

    except FileNotFoundError:
        print("File not found. Please try again.")
        return None

    except Exception as e:
        print("Error while loading dataset:", e)
        return None
    
def check_df_loaded(df):
     if df is None:
        print(" No dataset loaded! Please choose option 1 first.")
        return False
     return True

def  show_dataset_info(df):
    if not check_df_loaded(df):
        return

    print("\n=====  DATASET INFORMATION =====")

    print("\n🔹 First 5 Rows:")
    print(df.head())

    print("\n🔹 Dataset Shape (Rows, Columns):", df.shape)

    print("\n🔹 Column Names and Data Types:")
    print(df.dtypes)

    print("\n🔹 Missing Values in Each Column:")
    print(df.isnull().sum())

def clean_data(df):
    if not check_df_loaded(df):
        return df

    print("\n===== 🧹 DATA CLEANING =====")

    # Show current shape and first rows
    print("Before cleaning -> Shape:", df.shape)
    print("Before cleaning -> Head:")
    print(df.head())

    # ------ Remove Duplicates ------
    duplicates = df.duplicated().sum()
    print("Found duplicates count:", duplicates)
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"✔ Removed {duplicates} duplicate rows!")
    else:
        print("✔ No duplicate rows found.")

    # ------ Handle Missing Values ------
    missing_each = df.isnull().sum()
    missing_total = missing_each.sum()
    print("Missing values per column BEFORE cleaning:\n", missing_each)
    if missing_total > 0:
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print(f"✔ Filled {missing_total} missing numeric values with column mean.")
    else:
        print("✔ No missing values found.")

    print("After cleaning -> Shape:", df.shape)
    print("After cleaning -> Head:")
    print(df.head())

    print("📌 Data Cleaning Completed Successfully!")
    return df


def statistical_summary(df):
    if not check_df_loaded(df):
        return

    print("\n===== 📈 STATISTICAL SUMMARY =====")

    numeric_df = df.select_dtypes(include=['number'])

    print("\n🔹 Numeric Columns:")
    print(list(numeric_df.columns))

    print("\n🔹 Pandas Describe Summary:")
    print(numeric_df.describe())

    print("\n🔹 Custom Statistics (NumPy):")
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        print(f"\nColumn: {col}")
        print("   Mean:", np.mean(col_data))
        print("   Median:", np.median(col_data))
        print("   Std Dev:", np.std(col_data))
        print("   Min:", np.min(col_data))
        print("   Max:", np.max(col_data))

def visualize_data(df):
    if not check_df_loaded(df):
        return df

    while True:
        print("\n===== 📊 VISUALIZATION MENU =====")
        print("1. Histogram (numeric column)")
        print("2. Bar chart (categorical counts OR numeric mean by category)")
        print("3. Pie chart (categorical column)")
        print("4. Line chart (numeric column over index or a numeric 'x' column)")
        print("5. Correlation heatmap (numeric columns)")
        print("6. Back to main menu")

        choice = input("Choose visualization (1-6): ").strip()
        if choice == '1':
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                print("No numeric columns available for histogram.")
                continue
            print("Numeric columns:", numeric_cols)
            col = input("Enter numeric column for histogram: ").strip()
            if col not in numeric_cols:
                print("Invalid column. Try again.")
                continue
            plt.figure()
            plt.hist(df[col].dropna())
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            save = input("Save plot as PNG? (y/n): ").strip().lower()
            if save == 'y':
                fname = input("Enter filename (example: hist_math.png): ").strip()
                plt.savefig(fname, bbox_inches='tight')
                print(f"Saved plot to {fname}")
            else:
                plt.show()

        elif choice == '2':
            print("Columns:", list(df.columns))
            col_cat = input("Enter column for categories (e.g., name / class / any categorical): ").strip()
            if col_cat not in df.columns:
                print("Invalid column. Try again.")
                continue

            if pd.api.types.is_numeric_dtype(df[col_cat]):
                vc = df[col_cat].value_counts().sort_index()
                plt.figure()
                plt.bar(vc.index.astype(str), vc.values)
                plt.title(f"Value counts of {col_cat}")
                plt.xlabel(col_cat)
                plt.ylabel("Counts")
                plt.xticks(rotation=45)
                save = input("Save plot as PNG? (y/n): ").strip().lower()
                if save == 'y':
                    fname = input("Enter filename (example: bar_values.png): ").strip()
                    plt.savefig(fname, bbox_inches='tight')
                    print(f"Saved plot to {fname}")
                else:
                    plt.show()
            else:
                mode = input("Type 'counts' for counts or 'mean' to compute mean of a numeric column grouped by this category: ").strip().lower()
                if mode == 'counts':
                    vc = df[col_cat].value_counts()
                    plt.figure()
                    plt.bar(vc.index.astype(str), vc.values)
                    plt.title(f"Counts of {col_cat}")
                    plt.xlabel(col_cat)
                    plt.ylabel("Counts")
                    plt.xticks(rotation=45)
                    save = input("Save plot as PNG? (y/n): ").strip().lower()
                    if save == 'y':
                        fname = input("Enter filename (example: bar_counts.png): ").strip()
                        plt.savefig(fname, bbox_inches='tight')
                        print(f"Saved plot to {fname}")
                    else:
                        plt.show()
                elif mode == 'mean':
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if not numeric_cols:
                        print("No numeric columns available to compute mean.")
                        continue
                    print("Numeric columns:", numeric_cols)
                    numcol = input("Enter numeric column to compute mean by category: ").strip()
                    if numcol not in numeric_cols:
                        print("Invalid numeric column.")
                        continue
                    grp = df.groupby(col_cat)[numcol].mean().sort_values(ascending=False)
                    plt.figure()
                    plt.bar(grp.index.astype(str), grp.values)
                    plt.title(f"Mean of {numcol} by {col_cat}")
                    plt.xlabel(col_cat)
                    plt.ylabel(f"Mean {numcol}")
                    plt.xticks(rotation=45)
                    save = input("Save plot as PNG? (y/n): ").strip().lower()
                    if save == 'y':
                        fname = input("Enter filename (example: mean_by_cat.png): ").strip()
                        plt.savefig(fname, bbox_inches='tight')
                        print(f"Saved plot to {fname}")
                    else:
                        plt.show()
                else:
                    print("Invalid mode.")

        elif choice == '3':
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not cat_cols:
                print("No categorical columns available for pie chart.")
                continue
            print("Categorical columns:", cat_cols)
            col = input("Enter categorical column for pie chart: ").strip()
            if col not in cat_cols:
                print("Invalid column. Try again.")
                continue
            vc = df[col].value_counts()
            plt.figure()
            plt.pie(vc.values, labels=vc.index.astype(str), autopct='%1.1f%%')
            plt.title(f"Pie chart of {col}")
            save = input("Save plot as PNG? (y/n): ").strip().lower()
            if save == 'y':
                fname = input("Enter filename (example: pie_col.png): ").strip()
                plt.savefig(fname, bbox_inches='tight')
                print(f"Saved plot to {fname}")
            else:
                plt.show()

        elif choice == '4':
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                print("No numeric columns available for line chart.")
                continue
            print("Numeric columns:", numeric_cols)
            x_choice = input("Enter x column (leave blank to use dataframe index): ").strip()
            if x_choice == "":
                x = df.index
            else:
                if x_choice not in df.columns:
                    print("Invalid x column.")
                    continue
                x = df[x_choice]
            y_choice = input("Enter y (numeric) column for line chart: ").strip()
            if y_choice not in numeric_cols:
                print("Invalid y column.")
                continue
            plt.figure()
            plt.plot(x, df[y_choice])
            plt.title(f"Line chart: {y_choice} vs {x_choice or 'index'}")
            plt.xlabel(x_choice or 'index')
            plt.ylabel(y_choice)
            save = input("Save plot as PNG? (y/n): ").strip().lower()
            if save == 'y':
                fname = input("Enter filename (example: line_plot.png): ").strip()
                plt.savefig(fname, bbox_inches='tight')
                print(f"Saved plot to {fname}")
            else:
                plt.show()

        elif choice == '5':
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] < 2:
                print("Need at least 2 numeric columns for correlation heatmap.")
                continue
            corr = numeric_df.corr()
            plt.figure()
            plt.imshow(corr, interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title("Correlation Heatmap")
            save = input("Save plot as PNG? (y/n): ").strip().lower()
            if save == 'y':
                fname = input("Enter filename (example: corr_heatmap.png): ").strip()
                plt.savefig(fname, bbox_inches='tight')
                print(f"Saved plot to {fname}")
            else:
                plt.show()

        elif choice == '6':
            break
        else:
            print("Invalid option. Choose 1-6.")
    return df

def run_linear_regression(df):
    """
    Simple Linear Regression (one or multiple numeric features).
    Prints coefficients, intercept, R2 score and sample actual vs predicted.
    """
    if not check_df_loaded(df):
        return df

    print("\n===== 🤖 BASIC PREDICTION: LINEAR REGRESSION =====")

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        print("Need at least 2 numeric columns (features + target).")
        return df

    print("Numeric columns available:", numeric_cols)
    # Get features
    feat_input = input("Enter feature column names (comma-separated) e.g. math,science : ").strip()
    features = [c.strip() for c in feat_input.split(",") if c.strip()]
    if not features:
        print("No features selected. Aborting.")
        return df
    for f in features:
        if f not in numeric_cols:
            print(f"Feature '{f}' is not a numeric column or doesn't exist. Aborting.")
            return df

    # Get target
    target = input("Enter target column (numeric) e.g. english : ").strip()
    if target == "" or target not in numeric_cols:
        print("Invalid target column. Aborting.")
        return df
    if target in features:
        print("Target cannot be one of the features. Aborting.")
        return df

    # Prepare X, y and drop rows with NaN in selected columns
    selected_cols = features + [target]
    sub = df[selected_cols].dropna()
    if sub.shape[0] < 5:
        print("Not enough rows after dropping missing values (need >=5). Aborting.")
        return df

    X = sub[features].values
    y = sub[target].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print("\n✅ Model trained successfully!")
    # Coeffs
    if len(features) == 1:
        print(f"Coefficient for {features[0]}: {model.coef_[0]:.4f}")
    else:
        print("Coefficients:")
        for feat, coef in zip(features, model.coef_):
            print(f"  {feat}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"R² score on test set: {r2:.4f}")

    # Show few predictions vs actual
    print("\n🔎 Sample Actual vs Predicted (test set):")
    compare = []
    for actual, pred in zip(y_test, y_pred):
        compare.append((actual, pred))
    # print first 5
    for i, (a, p) in enumerate(compare[:5], start=1):
        print(f" {i}. Actual: {a:.3f}  |  Predicted: {p:.3f}")

    # Optional: show scatter plot of actual vs predicted or actual vs single feature
    if len(features) == 1:
        try:
            plt.figure()
            plt.scatter(X_test.flatten(), y_test, label="Actual")
            plt.scatter(X_test.flatten(), y_pred, label="Predicted", marker='x')
            plt.xlabel(features[0])
            plt.ylabel(target)
            plt.title(f"{target} vs {features[0]} (Actual vs Predicted)")
            plt.legend()
            plt.show()
        except Exception as e:
            print("Plot error (maybe headless env). You can save the plot instead.")

    # Optional: save model
    save_choice = input("\nSave trained model to file using joblib? (y/n): ").strip().lower()
    if save_choice == 'y':
        try:
            import joblib
            fname = input("Enter filename (example: linear_model.joblib): ").strip()
            joblib.dump(model, fname)
            print(f"Model saved to {fname}")
        except Exception as e:
            print("Could not save model. Make sure joblib is installed. Error:", e)

    return df


def print_menu():
    print("=== DATA ANALYTICS DASHBOARD===")
    print("1. Load Dataset")
    print("2. Show Dataset information")
    print("3. Data Cleaning")
    print("4. Statistical summary")
    print("5. Visualization")
    print("6. Basic prediction(Linear Regression)")

def main():
    global df
    while True:
      print_menu()
      choice=input("Enter your choice (1-7):-")
      if choice== '1':
         df=Load_dataset()
   
      elif choice=='2':
         show_dataset_info(df)

      elif choice == '3':
         df = clean_data(df)

      elif choice == '4':
        statistical_summary(df)
         
      elif choice=='5':
         df = visualize_data(df)
         
      elif choice=='6':
         df = run_linear_regression(df)
         
      elif choice=='7':
         print("Exiting... thankyou")
         break
      else:
         print("Invalid number ! Try again")


if __name__== "__main__":
          main()