from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "deployment_data" / "processed"
MODELS = ROOT / "models"


class DataService:
    def __init__(self):
        self._test = None
        self._static = None
        self._predictions = None
        self._anomalies = None
        self._scenarios = None
        self._scenario_summary = None

    @property
    def test(self):
        if self._test is None:
            self._test = pd.read_csv(
                DATA / "loan_monthly_performance_test.csv",
                low_memory=False,
            )
            self._test["loan_id"] = self._test["loan_id"].astype(str)
        return self._test

    @property
    def static(self):
        if self._static is None:
            self._static = pd.read_csv(
                DATA / "loan_static_attributes.csv",
                low_memory=False,
            )
            self._static["loan_id"] = self._static["loan_id"].astype(str)
        return self._static

    @property
    def predictions(self):
        if self._predictions is None:
            self._predictions = pd.read_csv(
                DATA / "ml_predictions_test.csv",
                low_memory=False,
            )
            self._predictions["loan_id"] = self._predictions["loan_id"].astype(str)
        return self._predictions

    @property
    def anomalies(self):
        if self._anomalies is None:
            self._anomalies = pd.read_csv(
                DATA / "servicer_anomaly_results.csv",
                low_memory=False,
            )
            self._anomalies["loan_id"] = self._anomalies["loan_id"].astype(str)
        return self._anomalies

    @property
    def scenarios(self):
        if self._scenarios is None:
            self._scenarios = pd.read_csv(
                DATA / "scenario_risk_results.csv",
                low_memory=False,
            )
            self._scenarios["loan_id"] = self._scenarios["loan_id"].astype(str)
        return self._scenarios

    @property
    def scenario_summary(self):
        if self._scenario_summary is None:
            self._scenario_summary = pd.read_csv(
                DATA / "scenario_risk_summary.csv",
                low_memory=False,
            )
        return self._scenario_summary

    def find_loans(self, query: str = "", limit: int = 20):
        df = self.static

        if query:
            mask = df["loan_id"].str.contains(
                str(query),
                case=False,
                na=False,
            )
            result = df.loc[mask].head(limit)
        else:
            result = df.head(limit)

        return result

    def get_loan_rows(self, loan_id: str):
        loan_id = str(loan_id)
        return self.test[self.test["loan_id"] == loan_id].copy()

    def get_static(self, loan_id: str):
        loan_id = str(loan_id)
        result = self.static[self.static["loan_id"] == loan_id]

        if result.empty:
            return None

        return result.iloc[0]

    def get_predictions(self, loan_id: str):
        loan_id = str(loan_id)
        return self.predictions[
            self.predictions["loan_id"] == loan_id
        ].copy()

    def get_anomalies(self, loan_id: str):
        loan_id = str(loan_id)
        return self.anomalies[
            self.anomalies["loan_id"] == loan_id
        ].copy()

    def get_scenarios(self, loan_id: str):
        loan_id = str(loan_id)
        return self.scenarios[
            self.scenarios["loan_id"] == loan_id
        ].copy()

    def get_model_metadata(self, model_name: str):
        path = MODELS / model_name / "metadata.json"

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
