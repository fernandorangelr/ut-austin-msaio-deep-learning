from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        return torch.min(self.data, dim=1).values, torch.max(self.data, dim=1).values

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        return torch.min(torch.diff(torch.mean(self.data, dim=1)))

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        _, indices = torch.max(torch.abs(self.data - torch.mean(self.data, dim=1, keepdim=True)), dim=1)
        return self.data[torch.arange(self.data.size(0)), indices]


    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        num_days = self.data.size(0)

        # Ensure k does not exceed the number of available days
        k = min(k, num_days)

        # Extract the last k days of data
        last_k_days = self.data[-k:]

        # Compute the maximum temperature for each day
        max_temps_per_day = last_k_days.max(dim=1).values  # Max across each row (each day's max temp)

        return max_temps_per_day

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        num_days = self.data.size(0)

        # Ensure k does not exceed the number of available days
        k = min(k, num_days)

        # Extract the last k days of data
        last_k_days = self.data[-k:]

        # Compute the average temperature over the last k days
        predicted_temp = last_k_days.mean()

        return predicted_temp

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        # Compute the sum of absolute differences (SAD) between `t` and each row (day) in the dataset
        sum_differences = torch.abs(self.data - t).sum(dim=1)

        # Find the index of the day with the minimum SAD
        closest_day_index = torch.argmin(sum_differences)

        return closest_day_index.long()
