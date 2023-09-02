

# Commented out IPython magic to ensure Python compatibility.
# Load libraries
import numpy as np
import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set(style="ticks")
# %matplotlib inline

from scipy.stats import norm
from scipy import stats

import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)

data=pd.read_csv("Prepaid.csv")
#data

telecom = data
telecom.head()

telecom.info(verbose=True, null_counts=True)

telecom.shape

telecom.describe()

highvalue = telecom

highvalue= highvalue.rename(columns={"is_churn" : "churn_flag"})

highvalue['churn_flag'].value_counts()

highvalue['churn_flag'].value_counts()* 100/highvalue.shape[0]

highvalue.shape

highvalue.isna().sum().sort_values(ascending=False)

round((highvalue.isna().sum()*100/highvalue.shape[0]),2).sort_values(ascending=False)

unique_stats = pd.DataFrame(highvalue.nunique()).reset_index().rename(columns = {'index': 'feature', 0: 'nunique'})
print(unique_stats[unique_stats['nunique'] == 1])

print('%d features with a single unique value.\n' % len(unique_stats[unique_stats['nunique'] == 1]))

highvalue.shape

highvalue = highvalue.drop(columns = list(unique_stats[unique_stats['nunique'] == 1]['feature']))
highvalue.head()

highvalue.shape

highvalue.fillna(0,inplace=True)

highvalue.isna().values.any()

highvalue[highvalue.duplicated(keep=False)]

highvalueo=highvalue.copy()
highvalueo.drop_duplicates(subset=None, inplace=True)
highvalueo.shape

del highvalueo
highvalue.shape

round((highvalue['churn_flag'].value_counts()*100 / highvalue.shape[0]),2)

plt.figure(figsize=(10,5))
ax=highvalue['churn_flag'].value_counts().plot(kind = 'bar')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.xticks(rotation = 45)
plt.ylabel('Count')
plt.xlabel('Churn Status')
plt.title('Churn Status Distribution',fontsize=14)
plt.show()

highvalue['accs_mthd_id'].value_counts().sort_values(ascending = False).head()

sns.pairplot(data=highvalue[['arpu_w01','arpu_w02','arpu_w03','arpu_w04','arpu_w05', 'arpu_w06', 'arpu_w07',
                             'arpu_w08', 'arpu_w09', 'arpu_w10','arpu_w11', 'arpu_w12', 'churn_flag']],hue='churn_flag',diag_kind='None')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12, 4))
ax = sns.distplot(highvalue[highvalue['churn_flag']==1].arpu_w01, bins = 30, ax = axes[0], kde = False)
ax.set_title('Churn')
ax.grid()
ax = sns.distplot(highvalue[highvalue['churn_flag']==0].arpu_w01, bins = 30, ax = axes[1], kde = False)
ax.set_title('Non-Churn')
ax.grid()
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(12,6))
sns.boxplot(x='churn_flag', y='arpu_w01', data=highvalue)
sns.stripplot(x='churn_flag', y='arpu_w01', data=highvalue, jitter=True, edgecolor="gray")
plt.show()

highvalue.info(verbose=True, null_counts=True)

highvalue['accs_mthd_id'] = highvalue['accs_mthd_id'].astype(str)
highvalue.info(verbose=True, null_counts=True)

highvalue.describe()

joincorr= highvalue.corr()
highvalue_corr = joincorr.stack().reset_index().sort_values(by = 0, ascending = False)
highvalue_corr[((highvalue_corr[0] < 1) & (highvalue_corr[0] >= 0.4)) |
               ((highvalue_corr[0] <= -0.4) & (highvalue_corr[0] > -1))]

highvalue['age_on_net_weeks'] = highvalue[['age_on_net_weeks_w01','age_on_net_weeks_w02','age_on_net_weeks_w03','age_on_net_weeks_w04','age_on_net_weeks_w05','age_on_net_weeks_w06','age_on_net_weeks_w07','age_on_net_weeks_w08','age_on_net_weeks_w09','age_on_net_weeks_w10','age_on_net_weeks_w11','age_on_net_weeks_w12']].mean(axis=1)
highvalue.head()

highvalue['rgtrn_provnc_id'] = highvalue[['rgtrn_provnc_id_w01','rgtrn_provnc_id_w02','rgtrn_provnc_id_w03','rgtrn_provnc_id_w04','rgtrn_provnc_id_w05','rgtrn_provnc_id_w06','rgtrn_provnc_id_w07','rgtrn_provnc_id_w08','rgtrn_provnc_id_w09','rgtrn_provnc_id_w10','rgtrn_provnc_id_w11','age_on_net_weeks_w12']].mean(axis=1)
highvalue.head()

highvalue['AVG_arpu_1_to_11'] = highvalue[['arpu_w01','arpu_w02','arpu_w03','arpu_w04','arpu_w05','arpu_w06','arpu_w07','arpu_w08','arpu_w09','arpu_w10','arpu_w11']].mean(axis=1)
highvalue['is_arpu_flag'] = np.where((highvalue['arpu_w12'] > highvalue['AVG_arpu_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_onnet_cl_cnt_1_to_11'] = highvalue[['onnet_cl_cnt_w01','onnet_cl_cnt_w02','onnet_cl_cnt_w03','onnet_cl_cnt_w04','onnet_cl_cnt_w05','onnet_cl_cnt_w06','onnet_cl_cnt_w07','onnet_cl_cnt_w08','onnet_cl_cnt_w09','onnet_cl_cnt_w10','onnet_cl_cnt_w11']].mean(axis=1)
highvalue['is_onnet_cnt_flag'] = np.where((highvalue['onnet_cl_cnt_w12'] > highvalue['AVG_onnet_cl_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_onnet_dur_1_to_11'] = highvalue[['onnet_actl_dur_w01','onnet_actl_dur_w02','onnet_actl_dur_w03','onnet_actl_dur_w04','onnet_actl_dur_w05','onnet_actl_dur_w06','onnet_actl_dur_w07','onnet_actl_dur_w08','onnet_actl_dur_w09','onnet_actl_dur_w10','onnet_actl_dur_w11']].mean(axis=1)
highvalue['is_onnet_dur_flag'] = np.where((highvalue['onnet_actl_dur_w12'] > highvalue['AVG_onnet_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_offnet_cl_cnt_1_to_11'] = highvalue[['offnet_cl_cnt_w01','offnet_cl_cnt_w02','offnet_cl_cnt_w03','offnet_cl_cnt_w04','offnet_cl_cnt_w05','offnet_cl_cnt_w06','offnet_cl_cnt_w07','offnet_cl_cnt_w08','offnet_cl_cnt_w09','offnet_cl_cnt_w10','offnet_cl_cnt_w11']].mean(axis=1)
highvalue['is_offnet_cnt_flag'] = np.where((highvalue['offnet_cl_cnt_w12'] > highvalue['AVG_offnet_cl_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_offnet_dur_1_to_11'] = highvalue[['offnet_actl_dur_w01','offnet_actl_dur_w02','offnet_actl_dur_w03','offnet_actl_dur_w04','offnet_actl_dur_w05','offnet_actl_dur_w06','offnet_actl_dur_w07','offnet_actl_dur_w08','offnet_actl_dur_w09','offnet_actl_dur_w10','offnet_actl_dur_w11']].mean(axis=1)
highvalue['is_offnet_dur_flag'] = np.where((highvalue['offnet_actl_dur_w12'] > highvalue['AVG_offnet_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_roam_ic_1_to_11'] = highvalue[['roma_voice_mt_w01','roma_voice_mt_w02','roma_voice_mt_w03','roma_voice_mt_w04','roma_voice_mt_w05','roma_voice_mt_w06','roma_voice_mt_w07','roma_voice_mt_w08','roma_voice_mt_w09','roma_voice_mt_w10','roma_voice_mt_w11']].mean(axis=1)
highvalue['is_roam_ic_flag'] = np.where((highvalue['roma_voice_mt_w12'] > highvalue['AVG_roam_ic_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_roam_og_1_to_11'] = highvalue[['roma_voice_mo_w01','roma_voice_mo_w02','roma_voice_mo_w03','roma_voice_mo_w04','roma_voice_mo_w05','roma_voice_mo_w06','roma_voice_mo_w07','roma_voice_mo_w08','roma_voice_mo_w09','roma_voice_mo_w10','roma_voice_mo_w11']].mean(axis=1)
highvalue['is_roam_og_flag'] = np.where((highvalue['roma_voice_mo_w12'] > highvalue['AVG_roam_og_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_fix_line_dur_1_to_11'] = highvalue[['fix_line_voi_dur_w01','fix_line_voi_dur_w02','fix_line_voi_dur_w03','fix_line_voi_dur_w04','fix_line_voi_dur_w05','fix_line_voi_dur_w06','fix_line_voi_dur_w07','fix_line_voi_dur_w08','fix_line_voi_dur_w09','fix_line_voi_dur_w10','fix_line_voi_dur_w11']].mean(axis=1)
highvalue['is_fix_line_dur_flag'] = np.where((highvalue['fix_line_voi_dur_w12'] > highvalue['AVG_fix_line_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_call_center_cnt_1_to_11'] = highvalue[['call_center_cnt_w01','call_center_cnt_w02','call_center_cnt_w03','call_center_cnt_w04','call_center_cnt_w05','call_center_cnt_w06','call_center_cnt_w07','call_center_cnt_w08','call_center_cnt_w09','call_center_cnt_w10','call_center_cnt_w11']].mean(axis=1)
highvalue['is_call_center_cnt_flag'] = np.where((highvalue['call_center_cnt_w12'] > highvalue['AVG_call_center_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_intl_cl_cnt_1_to_11'] = highvalue[['intl_cl_cnt_w01','intl_cl_cnt_w02','intl_cl_cnt_w03','intl_cl_cnt_w04','intl_cl_cnt_w05','intl_cl_cnt_w06','intl_cl_cnt_w07','intl_cl_cnt_w08','intl_cl_cnt_w09','intl_cl_cnt_w10','intl_cl_cnt_w11']].mean(axis=1)
highvalue['intl_cl_cnt_flag'] = np.where((highvalue['intl_cl_cnt_w12'] > highvalue['AVG_intl_cl_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_intl_cl_dur_1_to_11'] = highvalue[['intl_actl_dur_w01','intl_actl_dur_w02','intl_actl_dur_w03','intl_actl_dur_w04','intl_actl_dur_w05','intl_actl_dur_w06','intl_actl_dur_w07','intl_actl_dur_w08','intl_actl_dur_w09','intl_actl_dur_w10','intl_actl_dur_w11']].mean(axis=1)
highvalue['intl_cl_dur_flag'] = np.where((highvalue['intl_actl_dur_w12'] > highvalue['AVG_intl_cl_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_spl_cl_cnt_1_to_11'] = highvalue[['spl_cl_cnt_w01','spl_cl_cnt_w02', 'spl_cl_cnt_w03', 'spl_cl_cnt_w04', 'spl_cl_cnt_w05', 'spl_cl_cnt_w06',
                                              'spl_cl_cnt_w07', 'spl_cl_cnt_w08', 'spl_cl_cnt_w09', 'spl_cl_cnt_w10','spl_cl_cnt_w11']].mean(axis=1)
highvalue['is_spl_og_mou_flag'] = np.where((highvalue['spl_cl_cnt_w12'] > highvalue['AVG_spl_cl_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_intrconnect_dur_1_to_11'] = highvalue[['intrconnect_voi_dur_w01','intrconnect_voi_dur_w02','intrconnect_voi_dur_w03','intrconnect_voi_dur_w04','intrconnect_voi_dur_w05','intrconnect_voi_dur_w06','intrconnect_voi_dur_w07','intrconnect_voi_dur_w08','intrconnect_voi_dur_w09','intrconnect_voi_dur_w10','intrconnect_voi_dur_w11']].mean(axis=1)
highvalue['intrconnect_dur_flag'] = np.where((highvalue['intrconnect_voi_dur_w12'] > highvalue['AVG_intrconnect_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_intrconnect_smscnt_1_to_11'] = highvalue[['intrconnect_smscnt_w01','intrconnect_smscnt_w02','intrconnect_smscnt_w03','intrconnect_smscnt_w04','intrconnect_smscnt_w05','intrconnect_smscnt_w06','intrconnect_smscnt_w07','intrconnect_smscnt_w08','intrconnect_smscnt_w09','intrconnect_smscnt_w10','intrconnect_smscnt_w11']].mean(axis=1)
highvalue['intrconnect_smscnt_flag'] = np.where((highvalue['intrconnect_smscnt_w12'] > highvalue['AVG_intrconnect_smscnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_intrconnect_cl_cnt_1_to_11'] = highvalue[['intrconnect_cl_cnt_w01','intrconnect_cl_cnt_w02','intrconnect_cl_cnt_w03','intrconnect_cl_cnt_w04','intrconnect_cl_cnt_w05','intrconnect_cl_cnt_w06','intrconnect_cl_cnt_w07','intrconnect_cl_cnt_w08','intrconnect_cl_cnt_w09','intrconnect_cl_cnt_w10','intrconnect_cl_cnt_w11']].mean(axis=1)
highvalue['intrconnect_cl_cnt_flag'] = np.where((highvalue['intrconnect_cl_cnt_w12'] > highvalue['AVG_intrconnect_cl_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_rfl_amt_1_to_11'] = highvalue[['rfl_amt_w01','rfl_amt_w02','rfl_amt_w03','rfl_amt_w04','rfl_amt_w05','rfl_amt_w06','rfl_amt_w07','rfl_amt_w08','rfl_amt_w09','rfl_amt_w10','rfl_amt_w11']].mean(axis=1)
highvalue['rfl_amt_flag'] = np.where((highvalue['rfl_amt_w12'] > highvalue['AVG_rfl_amt_1_to_11']), 0, 1)
highvalue.head()

highvalue['AVG_rfl_cnt_1_to_11'] = highvalue[['rfl_cnt_w01','rfl_cnt_w02','rfl_cnt_w03','rfl_cnt_w04','rfl_cnt_w05','rfl_cnt_w06','rfl_cnt_w07','rfl_cnt_w08','rfl_cnt_w09','rfl_cnt_w10','rfl_cnt_w11']].mean(axis=1)
highvalue['rfl_cnt_flag'] = np.where((highvalue['rfl_cnt_w12'] > highvalue['AVG_rfl_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['data_usg2gvol_1_to_11'] = highvalue[['data_usg2gvol_w01', 'data_usg2gvol_w02', 'data_usg2gvol_w03', 'data_usg2gvol_w04', 'data_usg2gvol_w05',
                                               'data_usg2gvol_w06', 'data_usg2gvol_w07', 'data_usg2gvol_w08', 'data_usg2gvol_w09', 'data_usg2gvol_w10',
                                               'data_usg2gvol_w11']].mean(axis=1)
highvalue['is_vol_2g_mb_flag'] = np.where((highvalue['data_usg2gvol_w12'] > highvalue['data_usg2gvol_1_to_11']), 0, 1)
highvalue.head()

highvalue['data_usg3g_vol_1_to_11'] = highvalue[['data_usg3g_vol_w01', 'data_usg3g_vol_w02', 'data_usg3g_vol_w03', 'data_usg3g_vol_w04', 'data_usg3g_vol_w05',
                                               'data_usg3g_vol_w06', 'data_usg3g_vol_w07', 'data_usg3g_vol_w08', 'data_usg3g_vol_w09', 'data_usg3g_vol_w10',
                                               'data_usg3g_vol_w11']].mean(axis=1)
highvalue['is_vol_3g_mb_flag'] = np.where((highvalue['data_usg3g_vol_w12'] > highvalue['data_usg3g_vol_1_to_11']), 0, 1)
highvalue.head()

highvalue['data_usg4g_vol_1_to_11'] = highvalue[['data_usg4g_vol_w01', 'data_usg4g_vol_w02', 'data_usg4g_vol_w03', 'data_usg4g_vol_w04', 'data_usg4g_vol_w05',
                                               'data_usg4g_vol_w06', 'data_usg4g_vol_w07', 'data_usg4g_vol_w08', 'data_usg4g_vol_w09', 'data_usg4g_vol_w10',
                                               'data_usg4g_vol_w11']].mean(axis=1)
highvalue['is_vol_4g_mb_flag'] = np.where((highvalue['data_usg4g_vol_w12'] > highvalue['data_usg4g_vol_1_to_11']), 0, 1)
highvalue.head()

highvalue['data_payg_rev_1_to_11'] = highvalue[['data_payg_rev_w01','data_payg_rev_w02', 'data_payg_rev_w03', 'data_payg_rev_w04', 'data_payg_rev_w05',
                                        'data_payg_rev_w06', 'data_payg_rev_w07', 'data_payg_rev_w08', 'data_payg_rev_w09', 'data_payg_rev_w10',
                                        'data_payg_rev_w11']].mean(axis=1)
highvalue['is_vbc_payg_flag'] = np.where((highvalue['data_payg_rev_w12'] > highvalue['data_payg_rev_1_to_11']), 0, 1)
highvalue.head()

highvalue['lcl_actl_dur_1_to_11'] = highvalue[['lcl_actl_dur_w01','lcl_actl_dur_w02','lcl_actl_dur_w03','lcl_actl_dur_w04','lcl_actl_dur_w05','lcl_actl_dur_w06',
                                            'lcl_actl_dur_w07', 'lcl_actl_dur_w08','lcl_actl_dur_w09','lcl_actl_dur_w10','lcl_actl_dur_w11']].mean(axis=1)
highvalue['is_loc_og_mou_flag'] = np.where((highvalue['lcl_actl_dur_w12'] > highvalue['lcl_actl_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['mo_cl_actl_dur_1_to_11'] = highvalue[['mo_cl_actl_dur_w01','mo_cl_actl_dur_w02','mo_cl_actl_dur_w03','mo_cl_actl_dur_w04','mo_cl_actl_dur_w05'
                                              ,'mo_cl_actl_dur_w06','mo_cl_actl_dur_w07','mo_cl_actl_dur_w08','mo_cl_actl_dur_w09','mo_cl_actl_dur_w10'
                                              ,'mo_cl_actl_dur_w11']].mean(axis=1)
highvalue['is_total_og_mou_flag'] = np.where((highvalue['mo_cl_actl_dur_w12'] > highvalue['mo_cl_actl_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['night_dur_1_to_11'] = highvalue[['night_dur_w01','night_dur_w02','night_dur_w03','night_dur_w04','night_dur_w05','night_dur_w06',
                                                'night_dur_w07','night_dur_w08','night_dur_w09','night_dur_w10','night_dur_w11']].mean(axis=1)
highvalue['is_night_dur_user_flag'] = np.where((highvalue['night_dur_w12'] > highvalue['night_dur_1_to_11']), 0, 1)
highvalue.head()

highvalue['night_cnt_1_to_11'] = highvalue[['night_cnt_w01','night_cnt_w02','night_cnt_w03','night_cnt_w04','night_cnt_w05','night_cnt_w06',
                                                'night_cnt_w07','night_cnt_w08','night_cnt_w09','night_cnt_w10','night_cnt_w11']].mean(axis=1)
highvalue['is_night_cnt_user_flag'] = np.where((highvalue['night_cnt_w12'] > highvalue['night_cnt_1_to_11']), 0, 1)
highvalue.head()

highvalue['night_data_usg_1_to_11'] = highvalue[['night_data_usg_w01','night_data_usg_w02','night_data_usg_w03','night_data_usg_w04',
                                                 'night_data_usg_w05','night_data_usg_w06', 'night_data_usg_w07','night_data_usg_w08',
                                                 'night_data_usg_w09','night_data_usg_w10','night_data_usg_w11']].mean(axis=1)
highvalue['is_night_data_user_flag'] = np.where((highvalue['night_data_usg_w12'] > highvalue['night_data_usg_1_to_11']), 0, 1)
highvalue.head()

# Create month on month change features to understand any risk associated with the churn
highvalue['arpu_2diff1'] = (highvalue['arpu_w05'] + highvalue['arpu_w06'] + highvalue['arpu_w07'] + highvalue['arpu_w08']) - (highvalue['arpu_w01'] + highvalue['arpu_w02'] + highvalue['arpu_w03'] + highvalue['arpu_w04'])
highvalue['onnet_cl_cnt_2diff1'] = (highvalue['onnet_cl_cnt_w05'] + highvalue['onnet_cl_cnt_w06'] + highvalue['onnet_cl_cnt_w07'] + highvalue['onnet_cl_cnt_w08']) - (highvalue['onnet_cl_cnt_w01'] + highvalue['onnet_cl_cnt_w02'] + highvalue['onnet_cl_cnt_w03'] + highvalue['onnet_cl_cnt_w04'])
highvalue['offnet_cl_cnt_2diff1'] = (highvalue['offnet_cl_cnt_w05'] + highvalue['offnet_cl_cnt_w06'] + highvalue['offnet_cl_cnt_w07'] + highvalue['offnet_cl_cnt_w08']) - (highvalue['offnet_cl_cnt_w01'] + highvalue['offnet_cl_cnt_w02'] + highvalue['offnet_cl_cnt_w03'] + highvalue['offnet_cl_cnt_w04'])
highvalue['onnet_actl_dur_2diff1'] = (highvalue['onnet_actl_dur_w05'] + highvalue['onnet_actl_dur_w06'] + highvalue['onnet_actl_dur_w07'] + highvalue['onnet_actl_dur_w08']) - (highvalue['onnet_actl_dur_w01'] + highvalue['onnet_actl_dur_w02'] + highvalue['onnet_actl_dur_w03'] + highvalue['onnet_actl_dur_w04'])
highvalue['roma_voice_mt_2diff1'] = (highvalue['roma_voice_mt_w05'] + highvalue['roma_voice_mt_w06'] + highvalue['roma_voice_mt_w07'] + highvalue['roma_voice_mt_w08']) - (highvalue['roma_voice_mt_w01'] + highvalue['roma_voice_mt_w02'] + highvalue['roma_voice_mt_w03'] + highvalue['roma_voice_mt_w04'])
highvalue['roma_voice_mo_2diff1'] = (highvalue['roma_voice_mo_w05'] + highvalue['roma_voice_mo_w06'] + highvalue['roma_voice_mo_w07'] + highvalue['roma_voice_mo_w08']) - (highvalue['roma_voice_mo_w01'] + highvalue['roma_voice_mo_w02'] + highvalue['roma_voice_mo_w03'] + highvalue['roma_voice_mo_w04'])
highvalue['fix_line_voi_dur_2diff1'] = (highvalue['fix_line_voi_dur_w05'] + highvalue['fix_line_voi_dur_w06'] + highvalue['fix_line_voi_dur_w07'] + highvalue['fix_line_voi_dur_w08']) - (highvalue['fix_line_voi_dur_w01'] + highvalue['fix_line_voi_dur_w02'] + highvalue['fix_line_voi_dur_w03'] + highvalue['fix_line_voi_dur_w04'])
highvalue['call_center_cnt_2diff1'] = (highvalue['call_center_cnt_w05'] + highvalue['call_center_cnt_w06'] + highvalue['call_center_cnt_w07'] + highvalue['call_center_cnt_w08']) - (highvalue['call_center_cnt_w01'] + highvalue['call_center_cnt_w02'] + highvalue['call_center_cnt_w03'] + highvalue['call_center_cnt_w04'])
highvalue['intl_cl_cnt_2diff1'] = (highvalue['intl_cl_cnt_w05'] + highvalue['intl_cl_cnt_w06'] + highvalue['intl_cl_cnt_w07'] + highvalue['intl_cl_cnt_w08']) - (highvalue['intl_cl_cnt_w01'] + highvalue['intl_cl_cnt_w02'] + highvalue['intl_cl_cnt_w03'] + highvalue['intl_cl_cnt_w04'])
highvalue['intl_actl_dur_2diff1'] = (highvalue['intl_actl_dur_w05'] + highvalue['intl_actl_dur_w06'] + highvalue['intl_actl_dur_w07'] + highvalue['intl_actl_dur_w08']) - (highvalue['intl_actl_dur_w01'] + highvalue['intl_actl_dur_w02'] + highvalue['intl_actl_dur_w03'] + highvalue['intl_actl_dur_w04'])
highvalue['spl_cl_cnt_2diff1'] = (highvalue['spl_cl_cnt_w05'] + highvalue['spl_cl_cnt_w06'] + highvalue['spl_cl_cnt_w07'] + highvalue['spl_cl_cnt_w08']) - (highvalue['spl_cl_cnt_w01'] + highvalue['spl_cl_cnt_w02'] + highvalue['spl_cl_cnt_w03'] + highvalue['spl_cl_cnt_w04'])
highvalue['intrconnect_voi_dur_2diff1'] = (highvalue['intrconnect_voi_dur_w05'] + highvalue['intrconnect_voi_dur_w06'] + highvalue['intrconnect_voi_dur_w07'] + highvalue['intrconnect_voi_dur_w08']) - (highvalue['intrconnect_voi_dur_w01'] + highvalue['intrconnect_voi_dur_w02'] + highvalue['intrconnect_voi_dur_w03'] + highvalue['intrconnect_voi_dur_w04'])
highvalue['intrconnect_smscnt_2diff1'] = (highvalue['intrconnect_smscnt_w05'] + highvalue['intrconnect_smscnt_w06'] + highvalue['intrconnect_smscnt_w07'] + highvalue['intrconnect_smscnt_w08']) - (highvalue['intrconnect_smscnt_w01'] + highvalue['intrconnect_smscnt_w02'] + highvalue['intrconnect_smscnt_w03'] + highvalue['intrconnect_smscnt_w04'])
highvalue['intrconnect_cl_cnt_2diff1'] = (highvalue['intrconnect_cl_cnt_w05'] + highvalue['intrconnect_cl_cnt_w06'] + highvalue['intrconnect_cl_cnt_w07'] + highvalue['intrconnect_cl_cnt_w08']) - (highvalue['intrconnect_cl_cnt_w01'] + highvalue['intrconnect_cl_cnt_w02'] + highvalue['intrconnect_cl_cnt_w03'] + highvalue['intrconnect_cl_cnt_w04'])
highvalue['rfl_amt_2diff1'] = (highvalue['rfl_amt_w05'] + highvalue['rfl_amt_w06'] + highvalue['rfl_amt_w07'] + highvalue['rfl_amt_w08']) - (highvalue['rfl_amt_w01'] + highvalue['rfl_amt_w02'] + highvalue['rfl_amt_w03'] + highvalue['rfl_amt_w04'])
highvalue['rfl_cnt_2diff1'] = (highvalue['rfl_cnt_w05'] + highvalue['rfl_cnt_w06'] + highvalue['rfl_cnt_w07'] + highvalue['rfl_cnt_w08']) - (highvalue['rfl_cnt_w01'] + highvalue['rfl_cnt_w02'] + highvalue['rfl_cnt_w03'] + highvalue['rfl_cnt_w04'])
highvalue['data_usg2gvol_2diff1'] = (highvalue['data_usg2gvol_w05'] + highvalue['data_usg2gvol_w06'] + highvalue['data_usg2gvol_w07'] + highvalue['data_usg2gvol_w08']) - (highvalue['data_usg2gvol_w01'] + highvalue['data_usg2gvol_w02'] + highvalue['data_usg2gvol_w03'] + highvalue['data_usg2gvol_w04'])
highvalue['data_usg3g_vol_2diff1'] = (highvalue['data_usg3g_vol_w05'] + highvalue['data_usg3g_vol_w06'] + highvalue['data_usg3g_vol_w07'] + highvalue['data_usg3g_vol_w08']) - (highvalue['data_usg3g_vol_w01'] + highvalue['data_usg3g_vol_w02'] + highvalue['data_usg3g_vol_w03'] + highvalue['data_usg3g_vol_w04'])
highvalue['data_usg4g_vol_2diff1'] = (highvalue['data_usg4g_vol_w05'] + highvalue['data_usg4g_vol_w06'] + highvalue['data_usg4g_vol_w07'] + highvalue['data_usg4g_vol_w08']) - (highvalue['data_usg4g_vol_w01'] + highvalue['data_usg4g_vol_w02'] + highvalue['data_usg4g_vol_w03'] + highvalue['data_usg4g_vol_w04'])
highvalue['data_payg_rev_2diff1'] = (highvalue['data_payg_rev_w05'] + highvalue['data_payg_rev_w06'] + highvalue['data_payg_rev_w07'] + highvalue['data_payg_rev_w08']) - (highvalue['data_payg_rev_w01'] + highvalue['data_payg_rev_w02'] + highvalue['data_payg_rev_w03'] + highvalue['data_payg_rev_w04'])
highvalue['lcl_actl_dur_2diff1'] = (highvalue['lcl_actl_dur_w05'] + highvalue['lcl_actl_dur_w06'] + highvalue['lcl_actl_dur_w07'] + highvalue['lcl_actl_dur_w08']) - (highvalue['lcl_actl_dur_w01'] + highvalue['lcl_actl_dur_w02'] + highvalue['lcl_actl_dur_w03'] + highvalue['lcl_actl_dur_w04'])
highvalue['mo_cl_actl_dur_2diff1'] = (highvalue['mo_cl_actl_dur_w05'] + highvalue['mo_cl_actl_dur_w06'] + highvalue['mo_cl_actl_dur_w07'] + highvalue['mo_cl_actl_dur_w08']) - (highvalue['mo_cl_actl_dur_w01'] + highvalue['mo_cl_actl_dur_w02'] + highvalue['mo_cl_actl_dur_w03'] + highvalue['mo_cl_actl_dur_w04'])
highvalue['night_dur_2diff1'] = (highvalue['night_dur_w05'] + highvalue['night_dur_w06'] + highvalue['night_dur_w07'] + highvalue['night_dur_w08']) - (highvalue['night_dur_w01'] + highvalue['night_dur_w02'] + highvalue['night_dur_w03'] + highvalue['night_dur_w04'])
highvalue['night_cnt_2diff1'] = (highvalue['night_cnt_w05'] + highvalue['night_cnt_w06'] + highvalue['night_cnt_w07'] + highvalue['night_cnt_w08']) - (highvalue['night_cnt_w01'] + highvalue['night_cnt_w02'] + highvalue['night_cnt_w03'] + highvalue['night_cnt_w04'])
highvalue['night_data_usg_2diff1'] = (highvalue['night_data_usg_w05'] + highvalue['night_data_usg_w06'] + highvalue['night_data_usg_w07'] + highvalue['night_data_usg_w08']) - (highvalue['night_data_usg_w01'] + highvalue['night_data_usg_w02'] + highvalue['night_data_usg_w03'] + highvalue['night_data_usg_w04'])

highvalue.head()

highvalue.info(verbose=True, null_counts=True)

highvalue.shape

correlation_matrix = highvalue.corr()
AbsoluteCorrelationMatrix = correlation_matrix.abs()
AbsoluteCorrelationMatrix = AbsoluteCorrelationMatrix.where(np.triu(np.ones(AbsoluteCorrelationMatrix.shape), k=1).astype(np.bool))
highCorrelatedIndices = np.where(AbsoluteCorrelationMatrix > 0.8)
correlated_pairs = [(AbsoluteCorrelationMatrix.index[x], AbsoluteCorrelationMatrix.columns[y])
                     for x,y in zip(*highCorrelatedIndices) if x!=y and x < y]
correlated_pairs

print("Total Number of correlated pairs: ", len(correlated_pairs))

corr_matrix = highvalue.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
to_drop

highvalue.drop(columns=to_drop, axis=1,inplace=True)
highvalue.info(verbose=True, null_counts=True)

highvalue.shape

highvalue.drop(columns = list(highvalue.select_dtypes(include=['category']).columns), axis =1, inplace = True)
highvalue.info(verbose=True, null_counts=True)

list(highvalue.select_dtypes(include=[object]).columns)

highvalue.drop(columns = list(highvalue.select_dtypes(include=[object]).columns), axis = 1, inplace = True)
highvalue.info(verbose=True, null_counts=True)

highvalue.replace([np.inf, -np.inf], np.nan,inplace=True)
highvalue.fillna(0,inplace=True)
highvalue.info(verbose=True, null_counts=True)

highvalue.isnull().values.any()

highvalue.shape

highvalue.skew()

X = highvalue[highvalue.columns[~highvalue.columns.isin(['churn_flag'])]]
X.head()

Y = highvalue['churn_flag']
Y.head()

from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer()
X_scale= scaler.fit_transform(X)

# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scale,Y, train_size=0.7,random_state=42)

print("Training dataset size       ",X_train.shape)
print("Training dataset target size",y_train.shape)
print("Test dataset size           ",X_test.shape)
print("Test dataset target size    ",y_test.shape)

from sklearn.linear_model import LogisticRegression
Model_LR1 = LogisticRegression()
Model_LR1.fit(X_train,y_train)
Model_LR1.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,Model_LR1.predict(X_test))
cm

from sklearn.metrics import classification_report
classification_report(y_test, predictions)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# predict probabilities
lr_probs = Model_LR1.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# The magic happens here
import matplotlib.pyplot as plt
import scikitplot as skplt
predicted_probas = Model_LR1.predict_proba(X_test)
skplt.metrics.plot_cumulative_gain(y_test, predicted_probas)
plt.show()

# pip install kds
import kds
y_prob = Model_LR1.predict_proba(X_test)
kds.metrics.plot_cumulative_gain(y_test, y_prob[:,1])

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

y_prob = model.predict_proba(X_test)
kds.metrics.plot_cumulative_gain(y_test, y_prob[:,1])

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=29))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set()

acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()

from sklearn.metrics import confusion_matrix

y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
labels = ['Legitimate', 'Fraudulent']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')