# Deploying-ML-Model-to-Cloud-Platform-with-FastAPI


```python .\train.py

Precision: 0.7074047447879224
Recall: 0.6465177398160316
F-beta: 0.6755921730175077
Metrics for feature: workclass
              slice  precision    recall     fbeta  number_of_samples
0      Self-emp-inc   0.720930  0.808696  0.762295                223
1         Local-gov   0.716814  0.669421  0.692308                432
2           Private   0.703911  0.640895  0.670927               4507
3         State-gov   0.796875  0.680000  0.733813                247
4  Self-emp-not-inc   0.641667  0.506579  0.566176                535
5       Without-pay   1.000000  1.000000  1.000000                  5
6       Federal-gov   0.742857  0.684211  0.712329                190
Metrics for feature: marital-status
                   slice  precision    recall     fbeta  number_of_samples
0          Never-married   0.701754  0.449438  0.547945               1986
1     Married-civ-spouse   0.705744  0.690000  0.697783               2873
2               Divorced   0.733333  0.354839  0.478261                852
3  Married-spouse-absent   0.600000  0.500000  0.545455                 73
4                Widowed   1.000000  0.285714  0.444444                157
5              Separated   1.000000  0.416667  0.588235                195
6      Married-AF-spouse   0.000000  0.000000  1.000000                  3
Metrics for feature: occupation
                slice  precision    recall     fbeta  number_of_samples
0      Prof-specialty   0.790368  0.752022  0.770718                836
1    Transport-moving   0.511111  0.359375  0.422018                323
2        Adm-clerical   0.655172  0.612903  0.633333                742
3   Machine-op-inspct   0.588235  0.384615  0.465116                406
4     Exec-managerial   0.759709  0.794416  0.776675                836
5       Other-service   0.500000  0.137931  0.216216                656
6        Craft-repair   0.588608  0.497326  0.539130                864
7               Sales   0.670157  0.621359  0.644836                696
8   Handlers-cleaners   0.400000  0.117647  0.181818                260
9     Farming-fishing   0.461538  0.300000  0.363636                201
10       Tech-support   0.622642  0.673469  0.647059                166
11    Protective-serv   0.812500  0.650000  0.722222                127
12    Priv-house-serv   1.000000  1.000000  1.000000                 24
13       Armed-Forces   1.000000  1.000000  1.000000                  2
Metrics for feature: relationship
            slice  precision    recall     fbeta  number_of_samples
0   Not-in-family   0.701031  0.414634  0.521073               1544
1         Husband   0.706560  0.686477  0.696374               2552
2       Own-child   1.000000  0.333333  0.500000                926
3            Wife   0.690141  0.748092  0.717949                283
4       Unmarried   0.875000  0.311111  0.459016                663
5  Other-relative   0.666667  0.333333  0.444444                171
Metrics for feature: race
                slice  precision    recall     fbeta  number_of_samples
0               White   0.708366  0.656046  0.681203               5266
1               Black   0.728571  0.607143  0.662338                598
2  Asian-Pac-Islander   0.647059  0.458333  0.536585                169
3               Other   0.000000  0.000000  1.000000                 47
4  Amer-Indian-Eskimo   0.714286  0.714286  0.714286                 59
Metrics for feature: sex
    slice  precision    recall     fbeta  number_of_samples
0  Female   0.715084  0.603774  0.654731               1958
1    Male   0.706271  0.653435  0.678826               4181
```

```pytest .\test_model.py
=============================================== test session starts ================================================ 
platform win32 -- Python 3.8.20, pytest-8.3.3, pluggy-1.5.0
rootdir: C:\Users\samue\Downloads\Udacity\ML DevOps\Deploying-ML-Model-to-Cloud-Platform-with-FastAPI
plugins: anyio-4.5.0, hydra-core-1.3.2, typeguard-2.13.3
collected 4 items

test_model.py ....                                                                                            [100%] 

================================================= warnings summary ================================================= 
tests/test_model.py::test_process_data
tests/test_model.py::test_train_model
tests/test_model.py::test_inference
  C:\Users\samue\miniconda3\envs\deploy-ml-model\lib\site-packages\sklearn\preprocessing\_encoders.py:975: FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================================== 4 passed, 3 warnings in 15.42s ========================================== 
```


```python .\sanitycheck.py
This script will perform a sanity test to ensure your code meets the criteria in the rubric.

Please enter the path to the file that contains your test cases for the GET() and POST() methods
The path should be something like abc/def/test_xyz.py
> ./tests/test_main.py
C:\Users\samue\miniconda3\envs\deploy-ml-model\lib\site-packages\pydantic\_internal\_config.py:341: UserWarning: Valid config keys have changed in V2:
* 'schema_extra' has been renamed to 'json_schema_extra'
  warnings.warn(message, UserWarning)

============= Sanity Check Report ===========
Your test cases look good!
This is a heuristic based sanity testing and cannot guarantee the correctness of your code.
You should still check your work against the rubric to ensure you meet the criteria.
```

``` pytest .\test_main.py
=============================================== test session starts ================================================
platform win32 -- Python 3.8.20, pytest-8.3.3, pluggy-1.5.0
rootdir: C:\Users\samue\Downloads\Udacity\ML DevOps\Deploying-ML-Model-to-Cloud-Platform-with-FastAPI
plugins: anyio-4.5.0, hydra-core-1.3.2, typeguard-2.13.3
collected 3 items

test_main.py ...                                                                                              [100%] 

================================================= warnings summary ================================================= 
..\..\..\..\..\miniconda3\envs\deploy-ml-model\lib\site-packages\pydantic\_internal\_config.py:291
  C:\Users\samue\miniconda3\envs\deploy-ml-model\lib\site-packages\pydantic\_internal\_config.py:291: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.8/migration/
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning)

..\..\..\..\..\miniconda3\envs\deploy-ml-model\lib\site-packages\pydantic\_internal\_config.py:341
  C:\Users\samue\miniconda3\envs\deploy-ml-model\lib\site-packages\pydantic\_internal\_config.py:341: UserWarning: Valid config keys have changed in V2:
  * 'schema_extra' has been renamed to 'json_schema_extra'
    warnings.warn(message, UserWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================================== 3 passed, 2 warnings in 7.19s ===========================================
```

```python .\inference.py
Status Code: 200
Prediction: {'prediction': '<=50K'}
```

