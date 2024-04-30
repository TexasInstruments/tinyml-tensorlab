# TI's tiny models for Microcontrollers

### The following models are supported in tinyml-modelmaker:


| Model Name                 | Suited for                         | Availability           | Model Parameters | Model MACs (M) | Params Size (MB) | Estimated Total Size (MB) |
|----------------------------|------------------------------------|------------------------|------------------|----------------|------------------|---------------------------|
| TimeSeries_Generic_3k      | Generic Time series tasks          | GUI, tinyml-modelmaker | 3050             | 0.11           | 0.01             | 0.1                       |
| TimeSeries_Generic_AF_3k   | Arc Fault Classification           | GUI, tinyml-modelmaker | 3050             | 0.11           | 0.01             | 0.1                       |
| TimeSeries_Generic_MF_3k   | Motor Bearing Fault Classification | GUI, tinyml-modelmaker | 3050             | 0.11           | 0.01             | 0.1                       |
| TimeSeries_Generic_7k      | Generic Time series tasks          | GUI, tinyml-modelmaker | 7210             | 0.22           | 0.03             | 0.16                      |
| TimeSeries_Generic_AF_7k   | MArc Fault Classification          | GUI, tinyml-modelmaker | 7210             | 0.22           | 0.03             | 0.16                      |
| TimeSeries_Generic_MF_7k   | Motor Bearing Fault Classification | GUI, tinyml-modelmaker | 7210             | 0.22           | 0.03             | 0.16                      |
| TimeSeries_Generic_3k_t    | Generic Time series tasks          | GUI, tinyml-modelmaker | 3052             | 0.11           | 0.01             | 0.11                      |
| TimeSeries_Generic_AF_3k_t | Arc Fault Classification           | GUI, tinyml-modelmaker | 3052             | 0.11           | 0.01             | 0.11                      |
| TimeSeries_Generic_MF_3k_t | Motor Bearing Fault Classification | GUI, tinyml-modelmaker | 3052             | 0.11           | 0.01             | 0.11                      |
| TimeSeries_Generic_7k_t    | Generic Time series tasks          | GUI, tinyml-modelmaker | 7212             | 0.22           | 0.03             | 0.16                      |
| TimeSeries_Generic_AF_7k_t | Arc Fault Classification           | GUI, tinyml-modelmaker | 7212             | 0.22           | 0.03             | 0.16                      |
| TimeSeries_Generic_MF_7k_t | Motor Bearing Fault Classification | GUI, tinyml-modelmaker | 7212             | 0.22           | 0.03             | 0.16                      |
| ArcFault_model_200         | Arc Fault Classification           | GUI                    | 294              | 0.01           | <0.01            | 0.02                      |
| ArcFault_model_300         | Arc Fault Classification           | GUI                    | 386              | 0.02           | <0.01            | 0.05                      |
| ArcFault_model_700         | Arc Fault Classification           | GUI                    | 842              | 0.03           | <0.01            | 0.05                      |
| ArcFault_model_200_t       | Arc Fault Classification           | GUI                    | 296              | 0.01           | <0.01            | 0.03                      |
| ArcFault_model_300_t       | Arc Fault Classification           | GUI                    | 388              | 0.02           | <0.01            | 0.05                      |
| ArcFault_model_700_t       | Arc Fault Classification           | GUI                    | 844              | 0.03           | <0.01            | 0.05                      |
| MotorFault_model_1         | Motor Bearing Fault Classification | GUI                    | 586              | 0.01           | <0.01            | 0.02                      |
| MotorFault_model_2         | Motor Bearing Fault Classification | GUI                    | 4030             | 0.47           | 0.02             | 0.30                      |
| MotorFault_model_3         | Motor Bearing Fault Classification | GUI                    | 3698             | 0.59           | 0.01             | 0.35                      |
| MotorFault_model_1_t       | Motor Bearing Fault Classification | GUI                    | 588              | 0.01           | <0.01            | 0.02                      |
| MotorFault_model_2_t       | Motor Bearing Fault Classification | GUI                    | 4032             | 0.47           | 0.02             | 0.31                      |
| MotorFault_model_3_t       | Motor Bearing Fault Classification | GUI                    | 3700             | 0.59           | 0.01             | 0.35                      |

---
#### Note:
* The numbers in the table have all been measured for an input dimension of N,C,H,W of (1,1,512,1)
* 'Suited for' means it was originally created for that specific application. However, there is no stopping you from using it for other applications.
* Certain models above are available on the GUI version only, They are TI proprietary models.
  * The other models are called generic models.
  * They are available on tinyml-modelmaker too, and have their model definition exposed, meaning it can be tweaked by the user. 
* The models with '_t' in their names are slightly tweaked for TI MCUs with Hardware NPU (Eg: F28P55).
  * It is compulsory to use '_t' version models for F28P55 
  * On other devices, there is no significant advantage if you use '_t' models or not.
  * If you are confused, please use the '_t' variants.
* TimeSeries_Generic_*_3k are just exact copies of TimeSeries_Generic_3k created for GUI purposes. No difference in performance or parameters
  * Same applies for TimeSeries_Generic_*_7k models
---