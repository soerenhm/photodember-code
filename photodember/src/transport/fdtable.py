from __future__ import annotations
from dataclasses import dataclass
import json
import pandas as pd
import pathlib
from .core import ArrayF64


class DataSchema:
    REDUCED_TEMPERATURE = "Reduced temperature"
    CHEMICAL_POTENTIAL = "Chemical potential"
    TEMPERATURE = "Temperature"
    REDUCED_CHEMICAL_POTENTIAL = "Reduced chemical potential"
    NUMBER_DENSITY = "Number density"
    ENERGY_DENSITY = "Energy density"
    KINETIC_INTEGRAL_0 = "I0"
    KINETIC_INTEGRAL_1 = "I1"
    KINETIC_INTEGRAL_2 = "I2"


class MetaSchema:
    BANDGAP = "bandgap"
    RELATIVE_MASS = "relativeMass"
    NON_PARABOLICITY = "nonParabolicity"
    RELAXATION_TIME = "relaxationTime"


@dataclass
class FermiDiracTable:
    _data: pd.DataFrame
    _datafile: str

    @staticmethod
    def load_data(csvfile: str) -> FermiDiracTable:
        schema = DataSchema
        df = pd.read_csv(csvfile, delimiter="\t")
        columns = df.columns
        df = df.set_axis(
            [col.split(" [")[0] for col in columns], axis="columns"
        ).sort_values([schema.TEMPERATURE, schema.REDUCED_CHEMICAL_POTENTIAL])
        return FermiDiracTable(df, str(pathlib.Path(csvfile).absolute()))

    @staticmethod
    def load_meta(jsonfile: str) -> dict:
        with open(jsonfile, "rb") as io:
            out = json.load(io)
        return out

    def scale_mobility(self, mu_scale: float) -> FermiDiracTable:
        data = self._data.copy()
        data[DataSchema.KINETIC_INTEGRAL_0] *= mu_scale
        data[DataSchema.KINETIC_INTEGRAL_1] *= mu_scale
        data[DataSchema.KINETIC_INTEGRAL_2] *= mu_scale
        return FermiDiracTable(data, self._datafile)

    @property
    def reduced_temperature(self) -> ArrayF64:
        return self._data[DataSchema.REDUCED_TEMPERATURE].unique()

    @property
    def temperature(self) -> ArrayF64:
        return self._data[DataSchema.TEMPERATURE].unique()

    @property
    def chemical_potential(self) -> ArrayF64:
        return self._data[DataSchema.CHEMICAL_POTENTIAL].unique()

    @property
    def reduced_chemical_potential(self) -> ArrayF64:
        return self._data[DataSchema.REDUCED_CHEMICAL_POTENTIAL].unique()

    @property
    def origin(self) -> str:
        return self._datafile

    def as_array2d(self, key: str) -> ArrayF64:
        M = len(self.temperature)
        N = len(self.reduced_chemical_potential)
        return self._data[key].to_numpy().reshape((M, N))
