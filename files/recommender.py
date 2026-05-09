"""
Car Recommendation Engine with intelligent filtering and scoring.
"""

import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class RecommendationScore:
    """Score breakdown for a car recommendation."""
    car_id: int
    match_score: float
    value_score: float
    overall_score: float
    matched_preferences: List[str]


class CarRecommender:
    """Intelligent car recommendation engine."""

    def __init__(self, dataset: pd.DataFrame):

        self.df = dataset.copy()
        self.df_original = dataset.copy()

        required_cols = [
            'Brand',
            'Model',
            'Year',
            'Price',
            'Fuel_Type',
            'Transmission',
            'Mileage',
            'Engine_CC',
            'Seating_Capacity'
        ]

        missing = [
            col for col in required_cols
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def recommend(
        self,
        preferences: Dict,
        num_recommendations: int = 5
    ) -> List[Dict]:

        filtered_df = self._apply_filters(preferences)

        if len(filtered_df) == 0:
            return self._fallback_recommendations(
                preferences,
                num_recommendations
            )

        scores = self._score_cars(filtered_df, preferences)

        scores_sorted = sorted(
            scores,
            key=lambda x: x.overall_score,
            reverse=True
        )

        recommendations = self._diversify_recommendations(
            scores_sorted,
            filtered_df,
            num_recommendations
        )

        return recommendations

    def _apply_filters(self, preferences: Dict) -> pd.DataFrame:

        df = self.df_original.copy()

        applied_filters = []


        # =========================
        # BRAND FILTER
        # =========================
        if "brand" in preferences and preferences["brand"]:

            brand_value = preferences["brand"].strip().lower()

            df = df[
                df["Brand"]
                .astype(str)
                .str.strip()
                .str.lower()
                == brand_value
        ]

        applied_filters.append(
            f"Brand: {preferences['brand']}"
        )

        # =========================
        # MODEL FILTER
        # =========================
        if "model" in preferences and preferences["model"]:

            model_value = preferences["model"].strip().lower()

            df = df[
                df["Model"]
                .astype(str)
                .str.strip()
                .str.lower()
                == model_value
            ]
            
            applied_filters.append(
                f"Model: {preferences['model']}"
    )
        # =========================
        # BUDGET
        # =========================
        if ('budget_max' in preferences and preferences['budget_max']):

            df = df[
                df['Price'] <= preferences['budget_max']
            ]

            applied_filters.append(
                f"Budget ≤ ₹{preferences['budget_max']:,.0f}"
            )

        # =========================
        # FUEL TYPE
        # =========================
        if (
            'fuel_type' in preferences
            and preferences['fuel_type']
        ):

            df = df[
                df['Fuel_Type']
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                preferences['fuel_type'].lower()
            ]

            applied_filters.append(
                f"Fuel: {preferences['fuel_type']}"
            )

        # =========================
        # TRANSMISSION
        # =========================
        if (
            'transmission' in preferences
            and preferences['transmission']
        ):

            df = df[
                df['Transmission']
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                preferences['transmission'].lower()
            ]

            applied_filters.append(
                f"Transmission: {preferences['transmission']}"
            )

        # =========================
        # MILEAGE
        # =========================
        if (
            'min_mileage' in preferences
            and preferences['min_mileage']
        ):

            df = df[
                df['Mileage'] >= preferences['min_mileage']
            ]

            applied_filters.append(
                f"Mileage ≥ {preferences['min_mileage']} km/l"
            )

        # =========================
        # SEATING
        # =========================
        if (
            'min_seating' in preferences
            and preferences['min_seating']
        ):

            df = df[
                df['Seating_Capacity']
                >= preferences['min_seating']
            ]

            applied_filters.append(
                f"Seats ≥ {preferences['min_seating']}"
            )

        # =========================
        # AGE
        # =========================
        if (
            'max_age' in preferences
            and preferences['max_age']
            and 'Age' in df.columns
        ):

            df = df[
                df['Age'] <= preferences['max_age']
            ]

            applied_filters.append(
                f"Age ≤ {preferences['max_age']} years"
            )

        print(
            f"Applied filters: "
            f"{', '.join(applied_filters) if applied_filters else 'None'}"
        )

        print(f"Matching cars: {len(df)}")

        # IMPORTANT FIX
        df = df.reset_index(drop=True)

        return df

    def _score_cars(
        self,
        filtered_df: pd.DataFrame,
        preferences: Dict
    ) -> List[RecommendationScore]:

        scores = []

        filtered_df = filtered_df.reset_index(drop=True)

        for idx, row in filtered_df.iterrows():

            matched = []

            match_score = 0

            weights = {}

            # =========================
            # BUDGET SCORE
            # =========================
            if (
                'budget_max' in preferences
                and preferences['budget_max']
            ):

                budget_ratio = (
                    row['Price']
                    / preferences['budget_max']
                )

                budget_score = max(
                    0,
                    100 * (1 - budget_ratio * 0.3)
                )

                match_score += budget_score * 0.2

                weights['budget'] = 0.2

                matched.append("Budget-friendly")

            # =========================
            # MILEAGE SCORE
            # =========================
            if (
                'min_mileage' in preferences
                and preferences['min_mileage']
            ):

                mileage_ratio = (
                    row['Mileage']
                    / preferences['min_mileage']
                )

                mileage_score = min(
                    100,
                    mileage_ratio * 100
                )

                match_score += mileage_score * 0.25

                weights['mileage'] = 0.25

                matched.append(
                    f"Good mileage ({row['Mileage']} km/l)"
                )

            # =========================
            # SEATING SCORE
            # =========================
            if (
                'min_seating' in preferences
                and preferences['min_seating']
            ):

                seating_score = min(
                    100,
                    (
                        row['Seating_Capacity']
                        /
                        preferences['min_seating']
                    ) * 100
                )

                match_score += seating_score * 0.15

                weights['seating'] = 0.15

                matched.append(
                    f"{int(row['Seating_Capacity'])} seats"
                )

            # =========================
            # PRIORITY SCORE
            # =========================
            if (
                'prioritize' in preferences
                and preferences['prioritize']
            ):

                priority_score = 0

                for priority in preferences['prioritize']:

                    if priority.lower() == 'comfort':

                        comfort = (
                            (row['Engine_CC'] / 2000) * 40
                            +
                            (row['Mileage'] / 25) * 30
                            +
                            (row['Seating_Capacity'] / 7) * 30
                        )

                        priority_score = max(
                            priority_score,
                            min(100, comfort)
                        )

                    elif priority.lower() == 'performance':

                        age_value = (
                            row['Age']
                            if 'Age' in row
                            else 5
                        )

                        performance = (
                            (row['Engine_CC'] / 2500) * 70
                            +
                            (10 - age_value) * 5
                        )

                        priority_score = max(
                            priority_score,
                            min(100, performance)
                        )

                    elif priority.lower() == 'economy':

                        economy = (
                            (row['Mileage'] / 30) * 60
                            +
                            (
                                1 -
                                (
                                    row['Price']
                                    /
                                    self.df['Price'].max()
                                )
                            ) * 40
                        )

                        priority_score = max(
                            priority_score,
                            min(100, economy)
                        )

                    elif priority.lower() == 'luxury':

                        luxury = (
                            (
                                row['Price']
                                /
                                self.df['Price'].max()
                            ) * 50
                            +
                            (row['Year'] / 2024) * 50
                        )

                        priority_score = max(
                            priority_score,
                            min(100, luxury)
                        )

                if priority_score > 0:

                    match_score += priority_score * 0.25

                    weights['priority'] = 0.25

            # =========================
            # NORMALIZATION
            # =========================
            if sum(weights.values()) > 0:

                match_score = (
                    match_score
                    /
                    sum(weights.values())
                )

            else:
                match_score = 50

            # =========================
            # VALUE SCORE
            # =========================
            price_score = (
                1 -
                (
                    row['Price']
                    /
                    self.df['Price'].max()
                )
            ) * 100

            mileage_score = (
                row['Mileage']
                /
                self.df['Mileage'].max()
            ) * 100

            value_score = (
                price_score * 0.5
                +
                mileage_score * 0.5
            )

            # =========================
            # OVERALL SCORE
            # =========================
            overall_score = (
                match_score * 0.7
                +
                value_score * 0.3
            )

            scores.append(
                RecommendationScore(
                    car_id=idx,
                    match_score=match_score,
                    value_score=value_score,
                    overall_score=overall_score,
                    matched_preferences=matched
                )
            )

        return scores

    def _diversify_recommendations(
        self,
        scores: List[RecommendationScore],
        filtered_df: pd.DataFrame,
        num_recommendations: int
    ) -> List[Dict]:

        selected = []

        selected_brands = set()

        filtered_df = filtered_df.reset_index(drop=True)

        for idx, score in enumerate(scores):

            if len(selected) >= num_recommendations:
                break

            if idx >= len(filtered_df):
                continue

            car_row = filtered_df.iloc[idx]

            brand = car_row['Brand']

            if (
                len(selected) < num_recommendations // 2
                or brand not in selected_brands
            ):

                selected_brands.add(brand)

                rec_dict = self._create_recommendation_dict(
                    car_row,
                    score
                )

                selected.append(rec_dict)

        return selected

    def _create_recommendation_dict(
        self,
        car_row: pd.Series,
        score: RecommendationScore
    ) -> Dict:

        return {
            'Brand': car_row['Brand'],
            'Model': car_row['Model'],
            'Year': int(car_row['Year']),
            'Price': float(car_row['Price']),
            'Fuel_Type': car_row['Fuel_Type'],
            'Transmission': car_row['Transmission'],
            'Mileage': float(car_row['Mileage']),
            'Engine_CC': int(car_row['Engine_CC']),
            'Seating_Capacity': int(car_row['Seating_Capacity']),
            'Service_Cost': (
                float(car_row['Service_Cost'])
                if 'Service_Cost' in car_row
                else None
            ),
            'Overall_Score': round(score.overall_score, 2),
            'Match_Score': round(score.match_score, 2),
            'Value_Score': round(score.value_score, 2),
            'Matched_Preferences': score.matched_preferences
        }

    def _fallback_recommendations(
        self,
        preferences: Dict,
        num_recommendations: int
    ) -> List[Dict]:

        print("⚠️ No exact matches. Relaxing constraints...")

        relaxed_prefs = preferences.copy()

        filtered = pd.DataFrame()

        for key in [
            'max_age',
            'min_mileage',
            'min_seating',
            'fuel_type',
            'transmission'
        ]:

            if key in relaxed_prefs:

                del relaxed_prefs[key]

                filtered = self._apply_filters(relaxed_prefs)

                if len(filtered) > 0:
                    break

        # FINAL FALLBACK
        if len(filtered) == 0:

            if (
                'budget_max' in preferences
                and preferences['budget_max']
            ):

                filtered = self.df[
                    self.df['Price']
                    <= preferences['budget_max']
                ]

            else:
                filtered = self.df.copy()

        # SAFETY FIX
        filtered = filtered.reset_index(drop=True)

        if len(filtered) == 0:
            return []

        scores = self._score_cars(
            filtered,
            relaxed_prefs
        )

        scores_sorted = sorted(
            scores,
            key=lambda x: x.overall_score,
            reverse=True
        )

        recommendations = []

        for idx, score in enumerate(
            scores_sorted[:num_recommendations]
        ):

            if idx >= len(filtered):
                continue

            car_row = filtered.iloc[idx]

            recommendations.append(
                self._create_recommendation_dict(
                    car_row,
                    score
                )
            )

        return recommendations