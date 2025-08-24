# 1. IMPORTS
# -----------------------------------------------------------------------------
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# We'll need CatBoost to load the model.
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------
# 2. APPLICATION INITIALIZATION
# -----------------------------------------------------------------------------

# Create the FastAPI app instance.
app = FastAPI(
    title="Telecomms Churn Prediction API",
    description="A simple API for predicting customer churn based on multiple numerical and categorical features."
)

# -----------------------------------------------------------------------------
# 3. DEFINE INPUT DATA MODEL
# -----------------------------------------------------------------------------

# Use Pydantic's BaseModel to define the expected structure of the input data.
# This ensures data validation and provides a clear API schema.
class ChurnFeatures(BaseModel):
    """
    Data model for the input features.
    """
    # Numerical features
    MONTANT: float = Field(..., description="Top-up amount over the last 3 months", example=1000.0)
    FREQUENCE_RECH: float = Field(..., description="Number of times the customer has topped up over the last 3 months", example=5.0)
    ARPU_SEGMENT: float = Field(..., description="Average Revenue Per User per segment", example=50.25)
    FREQUENCE: float = Field(..., description="Number of times the customer engaged in activities over the last 3 months", example=12.0)
    DATA_VOLUME: float = Field(..., description="Data volume consumed over the last 3 months", example=2048.0)
    ON_NET: float = Field(..., description="On-net calls duration over the last 3 months", example=150.0)
    ORANGE: float = Field(..., description="Calls to Orange network over the last 3 months", example=25.0)
    TIGO: float = Field(..., description="Calls to Tigo network over the last 3 months", example=10.0)
    REGULARITY: float = Field(..., description="Number of times the customer is active consecutively for 90 days", example=5.0)
    FREQ_TOP_PACK: float = Field(..., description="Frequency of top-up packages purchased", example=3.0)

    # One-hot encoded REGION features
    REGION_DIOURBEL: float = Field(..., description="Is the customer's region Diourbel? (1 or 0)", example=0.0)
    REGION_FATICK: float = Field(..., description="Is the customer's region Fatick? (1 or 0)", example=0.0)
    REGION_KAFFRINE: float = Field(..., description="Is the customer's region Kaffrine? (1 or 0)", example=0.0)
    REGION_KAOLACK: float = Field(..., description="Is the customer's region Kaolack? (1 or 0)", example=1.0)
    REGION_KEDOUGOU: float = Field(..., description="Is the customer's region Kedougou? (1 or 0)", example=0.0)
    REGION_KOLDA: float = Field(..., description="Is the customer's region Kolda? (1 or 0)", example=0.0)
    REGION_LOUGA: float = Field(..., description="Is the customer's region Louga? (1 or 0)", example=0.0)
    REGION_MATAM: float = Field(..., description="Is the customer's region Matam? (1 or 0)", example=0.0)
    REGION_Missing_REGION: float = Field(..., description="Is the customer's region missing? (1 or 0)", example=0.0)
    REGION_SAINT_LOUIS: float = Field(..., description="Is the customer's region Saint-Louis? (1 or 0)", example=0.0)
    REGION_SEDHIOU: float = Field(..., description="Is the customer's region Sedhiou? (1 or 0)", example=0.0)
    REGION_TAMBACOUNDA: float = Field(..., description="Is the customer's region Tambacounda? (1 or 0)", example=0.0)
    REGION_THIES: float = Field(..., description="Is the customer's region Thies? (1 or 0)", example=0.0)
    REGION_ZIGUINCHOR: float = Field(..., description="Is the customer's region Ziguinchor? (1 or 0)", example=0.0)

    # One-hot encoded TOP_PACK features
    TOP_PACK_1500_Unlimited7Day: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_150_unlimited_pilot_auto: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_200_Unlimited1Day: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_200_unlimited_pilot_auto: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_200F_10mnOnNetValid1H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_3017650070: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_3051550090: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_500_Unlimited3Day: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_APANews_monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_APANews_weekly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_1000_5000_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_1000F_3000F_On_3000F_Off_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_300_600_2d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_5000_20000off_20000on_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_500_4000off_4000on_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_500F_2000F_AllNet_Unlimited: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_500F_1250F_AllNet_1250_Onnet_48h: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_500F_2000F_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_500F_4000F_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_All_net_600F_3000F_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_100F_unlimited: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_100f_200_MB: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_100f_500_onNet: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_150F_unlimited: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_200f_400MB: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_500f_2GB: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_On_net_1300f_125000: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_On_net_400f_2200F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_CVM_on_net_bundle_500_50000: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_100_F_40MB_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_200_F_100MB_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_200F_1GB_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_490F_Night_00H_08H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_1000F_2GB_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_1000F_5GB_7d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_1000F_700MB_7d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_1500F_3GB_30D: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_1500F_SPPackage1_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_150F_SPPackage1_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_200F_Unlimited_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_3000F_10GB_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_300F_100MB_2d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_30Go_V_30_Days: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_490F_1GB_7d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_500F_2GB_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_50F_30MB_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_700F_1_5GB_7d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_700F_SPPackage1_7d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_DailyCycle_Pilot_1_5GB: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_New_GPRS_PKG_1500F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_OneTime_Pilot_1_5GB: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_DataPack_Incoming: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_EVC_2Go24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_Mifi_10Go: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_Mifi_10Go_Monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Data_Mifi_20Go: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_ESN_POSTPAID_CLASSIC_RENT: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_1000_6000_F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_100Mo: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_1Go: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_4900_12000F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_500_2000F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_700Mo: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_JOKKO300: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_Jokko_Weekly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_MEGA10000F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_EVC_PACK_2_2Go: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_FIFA_TS_daily: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_FIFA_TS_monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_FIFA_TS_weekly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_FNF2_JAPPANTE: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_FNF_Youth_ESN: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Facebook_MIX_2D: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_GPRS_3000Equal10GPORTAL: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_GPRS_5Go_7D_PORTAL: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_GPRS_BKG_1000F_MIFI: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_GPRS_PKG_5GO_ILLIMITE: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Go_NetPro_4_Go: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_IVR_Echat_Daily_50F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_IVR_Echat_Monthly_500F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_IVR_Echat_Weekly_200F: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Incoming_Bonus_woma: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Internat_1000F_Zone_1_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Internat_1000F_Zone_3_24h: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Internat_2000F_Zone_2_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Jokko_Daily: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Jokko_Monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Jokko_Weekly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Jokko_promo: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_200mnoff_net_unl_on_net_5Go_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_390F_04HOn_net_400SMS_400_Mo_4h: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_4900F_10H_on_net_1_5Go_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_5000F_80Konnet_20Koffnet_250Mo_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_500F_75_SMS_ONNET_Mo_1000FAllNet_24h: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_590F_02H_On_net_200SMS_200_Mo_24h: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_10000F_10hAllnet_3Go_1h_Zone3_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_1000F_4250_Off_net_4250F_On_net_100Mo_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MIXT_500F_2500F_on_net_2500F_off_net_2d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MROMO_TIMWES_OneDAY: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MROMO_TIMWES_RENEW: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_MegaChrono_3000F_12500F_TOUS_RESEAUX: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Missing_TOP_PACK: float = Field(..., description="Is the top pack missing? (1 or 0)", example=0.0)
    TOP_PACK_Mixt_250F_Unlimited_call24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Mixt_500F_2500Fonnet_2500Foffnet_5d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_NEW_CLIR_PERMANENT_LIBERTE_MOBILE: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_NEW_CLIR_TEMPALLOWED_LIBERTE_MOBILE: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_NEW_CLIR_TEMPRESTRICTED_LIBERTE_MOBILE: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_New_YAKALMA_4_ALL: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_200F_3000F_10Mo_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_200F_Unlimited_call24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_1000F_10MilF_10d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_2000f_One_Month_100H_30d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_200F_60mn_1d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_300F_1800F_3d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_500_4000_10d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_On_net_500F_FNF_3d: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Package3_Monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Pilot_Youth1_2900: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Pilot_Youth4_4900: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Postpaid_FORFAIT_10H_Package: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_SMS_Max: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_SUPERMAGIK_1000: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_SUPERMAGIK_5000: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Staff_CPE_Rent: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_TelmunCRBT_daily: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Twter_U2opia_Daily: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Twter_U2opia_Monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Twter_U2opia_Weekly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_VAS_IVR_Radio_Daily: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_VAS_IVR_Radio_Monthly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_VAS_IVR_Radio_Weekly: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_WIFI_Family_10MBPS: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_WIFI_Family_4MBPS: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_WIFI_Family_2MBPS: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_YMGX_100_1_hour_FNF_24H_1_month: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_YMGX_on_net_100_700F_24H: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_Yewouleen_PKG: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_pack_chinguitel_24h: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_pilot_offer4: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_pilot_offer5: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_pilot_offer6: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)
    TOP_PACK_pilot_offer7: float = Field(..., description="Did the customer use this top pack? (1 or 0)", example=0.0)

    # One-hot encoded TENURE features
    TENURE_E_6_9_month: float = Field(..., description="Is the customer's tenure 6-9 months? (1 or 0)", example=0.0)
    TENURE_F_9_12_month: float = Field(..., description="Is the customer's tenure 9-12 months? (1 or 0)", example=0.0)
    TENURE_G_12_15_month: float = Field(..., description="Is the customer's tenure 12-15 months? (1 or 0)", example=0.0)
    TENURE_H_15_18_month: float = Field(..., description="Is the customer's tenure 15-18 months? (1 or 0)", example=0.0)
    TENURE_I_18_21_month: float = Field(..., description="Is the customer's tenure 18-21 months? (1 or 0)", example=0.0)
    TENURE_J_21_24_month: float = Field(..., description="Is the customer's tenure 21-24 months? (1 or 0)", example=0.0)
    TENURE_K_24_month: float = Field(..., description="Is the customer's tenure > 24 months? (1 or 0)", example=0.0)

# -----------------------------------------------------------------------------
# 4. LOAD THE MACHINE LEARNING MODEL
# -----------------------------------------------------------------------------

# We'll use a variable to store our loaded model.
model = None

# A try/except block is a good practice to handle potential errors when loading
# the model file, in case it's missing or corrupted.
try:
    # Load the CatBoost model from the .pkl file.
    # The 'rb' flag means 'read binary' which is necessary for model files.
    model = joblib.load('Catboost.pkl')
    print("CatBoost model loaded successfully.")
except FileNotFoundError:
    # If the file is not found, we raise an exception to pSrevent the
    # application from starting without a model.
    raise RuntimeError("Error: 'catboost.pkl' model file not found. "
                       "Please ensure the file is in the same directory.")
except Exception as e:
    # Handle other potential loading errors.
    raise RuntimeError(f"Error loading model: {e}")

# -----------------------------------------------------------------------------
# 5. DEFINE THE API ENDPOINT
# -----------------------------------------------------------------------------

# The endpoint `'/predict'` will handle POST requests.
# It accepts the `ChurnFeatures` data model in the request body.
@app.post("/predict")
def predict_churn(features: ChurnFeatures):
    """
    Predicts the probability of a customer churning.

    Args:
        features (ChurnFeatures): A JSON object containing the input features.

    Returns:
        A JSON object with the predicted churn probability.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Convert the Pydantic model's data into a NumPy array, which is the
    # standard input format for many machine learning models.
    # We create a 2D array, as models typically expect a list of samples.
    input_data = np.array([[
        features.MONTANT, features.FREQUENCE_RECH, features.ARPU_SEGMENT,
        features.FREQUENCE, features.DATA_VOLUME, features.ON_NET,
        features.ORANGE, features.TIGO, features.REGULARITY,
        features.FREQ_TOP_PACK, features.REGION_DIOURBEL,
        features.REGION_FATICK, features.REGION_KAFFRINE,
        features.REGION_KAOLACK, features.REGION_KEDOUGOU,
        features.REGION_KOLDA, features.REGION_LOUGA, features.REGION_MATAM,
        features.REGION_Missing_REGION, features.REGION_SAINT_LOUIS,
        features.REGION_SEDHIOU, features.REGION_TAMBACOUNDA,
        features.REGION_THIES, features.REGION_ZIGUINCHOR,
        features.TOP_PACK_1500_Unlimited7Day,
        features.TOP_PACK_150_unlimited_pilot_auto,
        features.TOP_PACK_200_Unlimited1Day,
        features.TOP_PACK_200_unlimited_pilot_auto,
        features.TOP_PACK_200F_10mnOnNetValid1H,
        features.TOP_PACK_3017650070,
        features.TOP_PACK_3051550090,
        features.TOP_PACK_500_Unlimited3Day,
        features.TOP_PACK_APANews_monthly,
        features.TOP_PACK_APANews_weekly,
        features.TOP_PACK_All_net_1000_5000_5d,
        features.TOP_PACK_All_net_1000F_3000F_On_3000F_Off_5d,
        features.TOP_PACK_All_net_300_600_2d,
        features.TOP_PACK_All_net_5000_20000off_20000on_30d,
        features.TOP_PACK_All_net_500_4000off_4000on_24H,
        features.TOP_PACK_All_net_500F_2000F_AllNet_Unlimited,
        features.TOP_PACK_All_net_500F_1250F_AllNet_1250_Onnet_48h,
        features.TOP_PACK_All_net_500F_2000F_5d,
        features.TOP_PACK_All_net_500F_4000F_5d,
        features.TOP_PACK_All_net_600F_3000F_5d,
        features.TOP_PACK_CVM_100F_unlimited,
        features.TOP_PACK_CVM_100f_200_MB,
        features.TOP_PACK_CVM_100f_500_onNet,
        features.TOP_PACK_CVM_150F_unlimited,
        features.TOP_PACK_CVM_200f_400MB,
        features.TOP_PACK_CVM_500f_2GB,
        features.TOP_PACK_CVM_On_net_1300f_125000,
        features.TOP_PACK_CVM_On_net_400f_2200F,
        features.TOP_PACK_CVM_on_net_bundle_500_50000,
        features.TOP_PACK_Data_100_F_40MB_24H,
        features.TOP_PACK_Data_200_F_100MB_24H,
        features.TOP_PACK_Data_200F_1GB_24H,
        features.TOP_PACK_Data_490F_Night_00H_08H,
        features.TOP_PACK_Data_1000F_2GB_30d,
        features.TOP_PACK_Data_1000F_5GB_7d,
        features.TOP_PACK_Data_1000F_700MB_7d,
        features.TOP_PACK_Data_1500F_3GB_30D,
        features.TOP_PACK_Data_1500F_SPPackage1_30d,
        features.TOP_PACK_Data_150F_SPPackage1_24H,
        features.TOP_PACK_Data_200F_Unlimited_24H,
        features.TOP_PACK_Data_3000F_10GB_30d,
        features.TOP_PACK_Data_300F_100MB_2d,
        features.TOP_PACK_Data_30Go_V_30_Days,
        features.TOP_PACK_Data_490F_1GB_7d,
        features.TOP_PACK_Data_500F_2GB_24H,
        features.TOP_PACK_Data_50F_30MB_24H,
        features.TOP_PACK_Data_700F_1_5GB_7d,
        features.TOP_PACK_Data_700F_SPPackage1_7d,
        features.TOP_PACK_Data_DailyCycle_Pilot_1_5GB,
        features.TOP_PACK_Data_New_GPRS_PKG_1500F,
        features.TOP_PACK_Data_OneTime_Pilot_1_5GB,
        features.TOP_PACK_DataPack_Incoming,
        features.TOP_PACK_Data_EVC_2Go24H,
        features.TOP_PACK_Data_Mifi_10Go,
        features.TOP_PACK_Data_Mifi_10Go_Monthly,
        features.TOP_PACK_Data_Mifi_20Go,
        features.TOP_PACK_ESN_POSTPAID_CLASSIC_RENT,
        features.TOP_PACK_EVC_1000_6000_F,
        features.TOP_PACK_EVC_100Mo,
        features.TOP_PACK_EVC_1Go,
        features.TOP_PACK_EVC_4900_12000F,
        features.TOP_PACK_EVC_500_2000F,
        features.TOP_PACK_EVC_700Mo,
        features.TOP_PACK_EVC_JOKKO300,
        features.TOP_PACK_EVC_Jokko_Weekly,
        features.TOP_PACK_EVC_MEGA10000F,
        features.TOP_PACK_EVC_PACK_2_2Go,
        features.TOP_PACK_FIFA_TS_daily,
        features.TOP_PACK_FIFA_TS_monthly,
        features.TOP_PACK_FIFA_TS_weekly,
        features.TOP_PACK_FNF2_JAPPANTE,
        features.TOP_PACK_FNF_Youth_ESN,
        features.TOP_PACK_Facebook_MIX_2D,
        features.TOP_PACK_GPRS_3000Equal10GPORTAL,
        features.TOP_PACK_GPRS_5Go_7D_PORTAL,
        features.TOP_PACK_GPRS_BKG_1000F_MIFI,
        features.TOP_PACK_GPRS_PKG_5GO_ILLIMITE,
        features.TOP_PACK_Go_NetPro_4_Go,
        features.TOP_PACK_IVR_Echat_Daily_50F,
        features.TOP_PACK_IVR_Echat_Monthly_500F,
        features.TOP_PACK_IVR_Echat_Weekly_200F,
        features.TOP_PACK_Incoming_Bonus_woma,
        features.TOP_PACK_Internat_1000F_Zone_1_24H,
        features.TOP_PACK_Internat_1000F_Zone_3_24h,
        features.TOP_PACK_Internat_2000F_Zone_2_24H,
        features.TOP_PACK_Jokko_Daily,
        features.TOP_PACK_Jokko_Monthly,
        features.TOP_PACK_Jokko_Weekly,
        features.TOP_PACK_Jokko_promo,
        features.TOP_PACK_MIXT_200mnoff_net_unl_on_net_5Go_30d,
        features.TOP_PACK_MIXT_390F_04HOn_net_400SMS_400_Mo_4h,
        features.TOP_PACK_MIXT_4900F_10H_on_net_1_5Go_30d,
        features.TOP_PACK_MIXT_5000F_80Konnet_20Koffnet_250Mo_30d,
        features.TOP_PACK_MIXT_500F_75_SMS_ONNET_Mo_1000FAllNet_24h,
        features.TOP_PACK_MIXT_590F_02H_On_net_200SMS_200_Mo_24h,
        features.TOP_PACK_MIXT_10000F_10hAllnet_3Go_1h_Zone3_30d,
        features.TOP_PACK_MIXT_1000F_4250_Off_net_4250F_On_net_100Mo_5d,
        features.TOP_PACK_MIXT_500F_2500F_on_net_2500F_off_net_2d,
        features.TOP_PACK_MROMO_TIMWES_OneDAY,
        features.TOP_PACK_MROMO_TIMWES_RENEW,
        features.TOP_PACK_MegaChrono_3000F_12500F_TOUS_RESEAUX,
        features.TOP_PACK_Missing_TOP_PACK,
        features.TOP_PACK_Mixt_250F_Unlimited_call24H,
        features.TOP_PACK_Mixt_500F_2500Fonnet_2500Foffnet_5d,
        features.TOP_PACK_NEW_CLIR_PERMANENT_LIBERTE_MOBILE,
        features.TOP_PACK_NEW_CLIR_TEMPALLOWED_LIBERTE_MOBILE,
        features.TOP_PACK_NEW_CLIR_TEMPRESTRICTED_LIBERTE_MOBILE,
        features.TOP_PACK_New_YAKALMA_4_ALL,
        features.TOP_PACK_On_net_200F_3000F_10Mo_24H,
        features.TOP_PACK_On_net_200F_Unlimited_call24H,
        features.TOP_PACK_On_net_1000F_10MilF_10d,
        features.TOP_PACK_On_net_2000f_One_Month_100H_30d,
        features.TOP_PACK_On_net_200F_60mn_1d,
        features.TOP_PACK_On_net_300F_1800F_3d,
        features.TOP_PACK_On_net_500_4000_10d,
        features.TOP_PACK_On_net_500F_FNF_3d,
        features.TOP_PACK_Package3_Monthly,
        features.TOP_PACK_Pilot_Youth1_2900,
        features.TOP_PACK_Pilot_Youth4_4900,
        features.TOP_PACK_Postpaid_FORFAIT_10H_Package,
        features.TOP_PACK_SMS_Max,
        features.TOP_PACK_SUPERMAGIK_1000,
        features.TOP_PACK_SUPERMAGIK_5000,
        features.TOP_PACK_Staff_CPE_Rent,
        features.TOP_PACK_TelmunCRBT_daily,
        features.TOP_PACK_Twter_U2opia_Daily,
        features.TOP_PACK_Twter_U2opia_Monthly,
        features.TOP_PACK_Twter_U2opia_Weekly,
        features.TOP_PACK_VAS_IVR_Radio_Daily,
        features.TOP_PACK_VAS_IVR_Radio_Monthly,
        features.TOP_PACK_VAS_IVR_Radio_Weekly,
        features.TOP_PACK_WIFI_Family_10MBPS,
        features.TOP_PACK_WIFI_Family_4MBPS,
        features.TOP_PACK_WIFI_Family_2MBPS,
        features.TOP_PACK_YMGX_100_1_hour_FNF_24H_1_month,
        features.TOP_PACK_YMGX_on_net_100_700F_24H,
        features.TOP_PACK_Yewouleen_PKG,
        features.TOP_PACK_pack_chinguitel_24h,
        features.TOP_PACK_pilot_offer4,
        features.TOP_PACK_pilot_offer5,
        features.TOP_PACK_pilot_offer6,
        features.TOP_PACK_pilot_offer7,
        features.TENURE_E_6_9_month, features.TENURE_F_9_12_month,
        features.TENURE_G_12_15_month, features.TENURE_H_15_18_month,
        features.TENURE_I_18_21_month, features.TENURE_J_21_24_month,
        features.TENURE_K_24_month
    ]])

    # Make the prediction using the loaded CatBoost model.
    # The `predict_proba` method returns probabilities for all classes.
    # For a binary classification (churn/no-churn), it returns something like [[prob_no_churn, prob_churn]].
    prediction = model.predict_proba(input_data)

    # CatBoost's `predict_proba` returns a list of lists,
    # so we extract the probability for the "churn" class (index 1).
    churn_probability = prediction[0][1]

    # Return the result as a JSON response.
    return {"churn_probability": float(churn_probability)}
