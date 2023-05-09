# D3HackathonE2EInventory
D3 onsite (May 11-12 2023) hackathon on replicating the implementation of the paper "A Practical End-to-End Inventory Management Model with Deep Learning". The paper is available at https://pubsonline.informs.org/doi/full/10.1287/mnsc.2022.4564. 


**Log files using original data**

1. “original_E2E_section4.pdf”: output of running “E2E_Section4.ipynb” using original data. This is the reproduce of Figure 2.

2. “original_field_experiment.pdf”: output of running “propensity_score_matching.ipynb” and part of “field_experiment_summary.ipynb”: reproduce Figure 3 and Table 2 and 3.

- Remarks: we separate “propensity_score_matching.ipynb” in the puiblished version is because we want to avoid sharing the sales data of all products in candidate control group. One only needs average demand and average VLT data to do propensity score matching.

3. “original_field_experiment_summary.ipynb”: remaining part of “field_experiment_summary.ipynb”: reproduce Table 4.



**Data description (experiment)**
1.	Dataset "df_test.csv": 
•	Dataset description: a subset of data (3000 SKUs) that used for testing the various inventory replenishment model performance in the offline experiment.
•	Dataset dictionary: 
	“SKU”: disguised SKU ID
	“demand_mean”: average demand of the SKU
	“demand_std”: standard deviation of demand of the SKU
	“review_period”: item review period 
	“vendor_vlt_mean”: average vendor lead time 
	“vendor_vlt_std”: standard deviation of vendor lead time 
	“initial_stock”: initial inventory level
	“demand_hist”: historical demand 
	“OPT_pred”: offline optimal solution given the realized demand and vlt in the test period
	“E2E_RNN_pred”: replenishment quantity predicted by the proposed E2E model
	“gbm_pred_pred”: replenishment quantity predicted by the LightGBM model
	“sf_rnn”: demand forecasting series outputted by the proposed E2E model
	“vlt”: vlt forecast outputted by the proposed E2E model
•	The original dataset has the same dictionaries.
•	Disguise process: desensitization of the SKU ID. We also randomly sample a subset of 3000 SKU from the original dataset because the original dataset is too large to disclose. 
2.	Datasets “order_e2e_post_500.csv”, “order_e2e_post_500.csv”, “order_e2e_post_500.csv” and “order_e2e_post_500.csv”.
•	Dataset description: “order_e2e_post_500.csv” and “order_e2e_post_500.csv” include post- and pre- experiment results, respectively, for a subset of the treatment group (500 SKU-DC pairs). “order_e2e_post_500.csv” and “order_e2e_post_500.csv” include post- and pre- experiment results, respectively, for a subset (500 SKU-DC pairs) of the control group. Note that the control group has been selected from all candidate SKU-DC pairs in the same category by propensity score matching.
•	Data dictionary: 
	“sku_dc_pair_index”: desensitized index for each of the (SKU, DC) pair
	“complete_tm”: order complete time
	“order_amount”: order amount (replenishment quantities)
	“test_inv”: inventory curve within test period
	“test_demand”: demand observations within test period
	“ave_demand”: average demand
	“ave_inv”: average inventory level
	“vlt”: vendor lead time
	“vlt_num”: extracted value of vendor lead time
•	Dataset dictionaries of original datasets:
	Two columns “SKU_id”:  SKU ID and “DC”: distribution center ID instead of “sku_dc_pair_index”.
	Other columns are the same as the disguised dataset
•	Disguise process: The sku_dc_pair_index has been desensitized to disguise the original SKU name and DC ID. We randomly sampled 500 (SKU,DC) pairs for the treatment group because the original dataset is too large to be disclosed.
3.	Dataset “control_demand_vlt.csv”：
•	Dataset discerption:  These (SKU, DC) pairs are all candidate pairs that can be selected as the control group. The propensity score matching are performed on this dataset to select a control group to reduce the potential selection bias.
•	Dataset dictionary:
	Index: (SKU, DC) pair index
	“ave_demand”: average demand
	“vlt_num”: vendor lead time value 
•	Dataset dictionaries of original datasets:
	Two additional columns “SKU_id”:  SKU ID and “DC”: distribution center ID 
	Other columns are the same as the disguised dataset
•	Disguise process: we desensitized the original SKU name and DC ID and give each (SKU, DC) pair an index. 
Data description (model)
1.	Dataset "rdc_sales_1320.csv": 
•	Dataset description: a disguised example data set (5 SKU-DC pairs) for daily sales. 
•	Dataset dictionary: 
	“row”: disguised SKU-DC ID (format: SKU_ID#DC_ID)
	Other than “row”, the following columns represents daily sales data. The index of each column was converted to date-time format in “data_parser.py”.
•	The original dataset has the same dictionaries.
•	Disguise process: desensitization of the SKU ID and DC ID. We also randomly sample a subset of 5 SKU from the original dataset to give an example. 
2.	Dataset “stock.csv”:
•	Dataset description: “stock.csv” is a disguised example data set for stock levels in the warehouse. 
•	Data dictionary: 
	“item_sku_id”: disguised SKU-DC ID (format: SKU_ID#DC_ID)
	Other than “row”, the following columns represents daily stock-level data. Each column is indexed with date. 
•	The original dataset includes longer record of stock level. The provided data set is trimmed to one month. 
•	Disguise process: desensitization of the SKU ID and DC ID. 2 SKUs are randomly picked from the original dataset and is trimmed to be one-month long.
3.	Dataset “vlt_2019_0305.csv”：
•	Dataset discerption: a disguised example dataset for vendor lead time. 
•	Dataset dictionary:
	“item_sku_id”: disguised SKU- ID 
	“int_org_num”: disguised DC-ID
	“pur_bill_id”: purchase bill ID.
	“create_tm” and “complete_tm”: create time and complete time.
	“vlt_actual”: actual vendor lead time (equals to complete time – create time).
	“actual_pur_qtty”: actual purchase quantity.
	“item_first_cate_cd”, “item_second_cate_cd”, “item_third_cate_cd”: item first, second, and third category ID.
	“brand_code”: brand code.
	“create_day_of_week”: create day of the week.
	Disguised volumn names for three groups of features: item dimension information (e.g. weight, height, length), item-wise and vendor-wise historical VLT statistics (e.g. averaged VLT over certain time window, standard deviation of VLT ), statistics of item historical purchase quantities (e.g. averaged purchase quantity over certain time windows).
•	Dataset dictionaries of original datasets:
	More column corresponding to item dimension information, item-wise and vendor-wise historical VLT statistics. The column names should be consistent with “VLT_FEA” defined in config/config.py
	Other columns are the same as the disguised dataset
•	Disguise process: we desensitized the original SKU name and DC ID and remove confidential column names. The data set contains on SKU-DC pair as an example. 
Code (experiment)
1.	“E2E_Section4.ipynb”: reproduce Figure 2
•	Same code applies on the disguised and undisguised data. 
•	Description: Offline comparisons of seven different inventory replenishment models. As only a subset of the entire test dataset is allowed to share due to business reasons. The reproduced results have different numerical value compared to the results in Section 4.2 that are obtained on the entire test dataset. However, all the observed trend, relative performance comparison and conclusion remain the same.
2.	“propensity_score_matching.ipynb”: reproduce Figure 3
•	Same code applies on the disguised and undisguised data.
•	Description: Demonstration of using propensity score matching to select the control group out of all candidate SKU-DC pairs.  As we only provide a subset of the treatment group, the propensity score density plots are different than plots in the paper. But it still has a perfect matching. 
3.	“field_experiment_summary.ipynb”: reproduce Table 2,3 and 4
•	Same code applies on the disguised and undisguised data
•	Description: evaluates five inventory performance metrics for both the treatment and control groups for the pre-experiment periods and post-experiment periods. This notebook also demonstrates the result of the t-test, linear regression and Difference-in-Difference estimation. As we only can share a subset of the dataset, the numerical value is different from the results in Section 5 but the conclusion remains the same.
Code (model)
The folder “E2E-model-code” includes code and pseudo-code for constructing and training the End-to-End model with RNN.
Due to confidentiality, certain parts of the source code are replaced with pseudo-code or detailed description.
Disguised parts:
1.	“config/config.py”: Details of feature names and RNN parameters are disguised. 
2.	“data_loader/data_parser.py”: Disguised input and output paths. Disguised value of some parameters. Disguised data cleaning and preprocessing details in functions “add_target” and “dummy_cut”.
3.	“models/loss.py”: Disguised weight parameters in training loss functions. Disguised details of choice of loss function. 
4.	“models/model.py”: Disguised initialization of weights. Disguised feature dimensions. 
5.	“utils/demand_pkg.py”: Window lists used in features are disguised.  

![image](https://github.com/FelixGSchmidt/D3HackathonE2EInventory/assets/51131718/ea3f90a6-4135-4c41-8a99-f1ecc28ab520)
