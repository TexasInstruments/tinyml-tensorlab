# TI's tiny models for Microcontrollers

### The following models are supported in tinyml-modelmaker:


| Model Name              | Suited for                         | Availability      | No. of Parameters | Model MACs (M) |
|-------------------------|------------------------------------|-------------------|-------------------|----------------|
| TimeSeries_Generic_3k   | Generic Time series tasks          | tinyml-modelmaker | 3050              | 0.11           |
| TimeSeries_Generic_7k   | Generic Time series tasks          | tinyml-modelmaker | 7210              | 0.22           |
| TimeSeries_Generic_3k_t | Generic Time series tasks          | tinyml-modelmaker | 3052              | 0.11           |
| TimeSeries_Generic_7k_t | Generic Time series tasks          | tinyml-modelmaker | 7212              | 0.22           |
| ArcFault_model_200_t    | Arc Fault Classification           | GUI               | 296               | 0.01           |
| ArcFault_model_300_t    | Arc Fault Classification           | GUI               | 388               | 0.02           |
| ArcFault_model_700_t    | Arc Fault Classification           | GUI               | 844               | 0.03           |
| MotorFault_model_1_t    | Motor Bearing Fault Classification | GUI               | 588               | 0.01           |
| MotorFault_model_2_t    | Motor Bearing Fault Classification | GUI               | 4032              | 0.47           |

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
