`solver.py` - решатель для уравнения давления  ∇⋅(k∇p)=q

`generate_pressure.py` - сгенерирует датасет, перебор всех возможных комбинаций расположения нагнетальной и 
добывающей скважин на сетке. датасет загружается в папку pressure_dataset

`gnn_model.py` - графовая нейросеть, датасет для обучения загружается из папки `pressure_dataset`, выборка делится на train, val.
после обучения сохраняем модель в файл `pressure_gnn_model.pth`

`eval_gnn.py` - оцениваем получившуюся модель из файла `pressure_gnn_model.pth`, считаем MSE, визуализируем валидационный набор данных: правильное и спрогнозированные значения

`visualize_dataset.py` - визуализируем любой файл с распределением давления из датасета `pressure_dataset`