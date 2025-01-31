import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class UnstructuredMesh:
    def __init__(self, vtk_file):
        """
        Инициализация класса, чтение данных из VTK-файла.
        """
        self.vtk_file = vtk_file
        self.points = None  # Координаты узлов
        self.cells = None  # Индексы узлов в ячейках
        self.cell_types = None  # Типы ячеек (например, тетраэдры, гексаэдры)
        self.porosity = None  # Пористость в ячейках
        self.permeability = None  # Проницаемость в ячейках

        # Загружаем данные из файла
        self._load_vtk()

    def _load_vtk(self):
        """
        Загрузка данных сетки из VTK-файла.
        """
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(self.vtk_file)
        reader.Update()

        data = reader.GetOutput()

        # Получаем координаты узлов
        self.points = vtk_to_numpy(data.GetPoints().GetData())

        # Получаем ячейки и их типы
        self.cells = vtk_to_numpy(data.GetCells().GetData())
        self.cell_types = vtk_to_numpy(data.GetCellTypesArray())

        # Получаем скалярные данные (например, пористость и проницаемость)
        if data.GetCellData().HasArray("porosity"):
            self.porosity = vtk_to_numpy(data.GetCellData().GetArray("porosity"))
        else:
            raise ValueError("Пористость (porosity) не найдена в VTK-файле.")

        if data.GetCellData().HasArray("permeability"):
            self.permeability = vtk_to_numpy(data.GetCellData().GetArray("permeability"))
        else:
            raise ValueError("Проницаемость (permeability) не найдена в VTK-файле.")

    def calculate_transmissibility(self):
        """
        Вычисление трансмиссивности для каждой грани между соседними ячейками.
        Пока реализуем только структуру.
        """
        pass  # Здесь будет реализация вычислений трансмиссивности.

    def summary(self):
        """
        Вывод основных свойств сетки.
        """
        print(f"Сетка из файла: {self.vtk_file}")
        print(f"Количество узлов: {len(self.points)}")
        print(f"Количество ячеек: {len(self.porosity)}")
        print(f"Типы ячеек (уникальные): {np.unique(self.cell_types)}")
