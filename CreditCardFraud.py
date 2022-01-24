{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "555cd957-6c4b-49f3-af57-bd569583de36",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization\n",
    "from tensorflow.keras.layers import Conv1D, MaxPool1D\n",
    "from tensorflow.keras.optimizers import Adam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0831b709-e2cb-4965-a5b8-614308e51389",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2.5.0\n"
     ]
    }
   ],
   "source": [
    "print(tf.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c44268b2-74f9-460c-90a6-03f7439b564a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9efc85a5-4bca-4baf-858b-c3b2dc446845",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>149.62</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>...</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>378.66</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>123.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>69.99</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9  ...       V21       V22       V23       V24       V25  \\\n",
       "0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   \n",
       "1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   \n",
       "2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   \n",
       "3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   \n",
       "4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   \n",
       "\n",
       "        V26       V27       V28  Amount  Class  \n",
       "0 -0.189115  0.133558 -0.021053  149.62      0  \n",
       "1  0.125895 -0.008983  0.014724    2.69      0  \n",
       "2 -0.139097 -0.055353 -0.059752  378.66      0  \n",
       "3 -0.221929  0.062723  0.061458  123.50      0  \n",
       "4  0.502292  0.219422  0.215153   69.99      0  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe = pd.read_csv('creditcard.csv')\n",
    "dataframe.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "30e658c3-f8b0-4aaf-beab-f371eb914216",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(284807, 31)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "722958bb-0a22-4c90-abfa-9df182dbca18",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Time      0\n",
       "V1        0\n",
       "V2        0\n",
       "V3        0\n",
       "V4        0\n",
       "V5        0\n",
       "V6        0\n",
       "V7        0\n",
       "V8        0\n",
       "V9        0\n",
       "V10       0\n",
       "V11       0\n",
       "V12       0\n",
       "V13       0\n",
       "V14       0\n",
       "V15       0\n",
       "V16       0\n",
       "V17       0\n",
       "V18       0\n",
       "V19       0\n",
       "V20       0\n",
       "V21       0\n",
       "V22       0\n",
       "V23       0\n",
       "V24       0\n",
       "V25       0\n",
       "V26       0\n",
       "V27       0\n",
       "V28       0\n",
       "Amount    0\n",
       "Class     0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2348172f-5e40-432c-b429-e58e1bace7de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 284807 entries, 0 to 284806\n",
      "Data columns (total 31 columns):\n",
      " #   Column  Non-Null Count   Dtype  \n",
      "---  ------  --------------   -----  \n",
      " 0   Time    284807 non-null  float64\n",
      " 1   V1      284807 non-null  float64\n",
      " 2   V2      284807 non-null  float64\n",
      " 3   V3      284807 non-null  float64\n",
      " 4   V4      284807 non-null  float64\n",
      " 5   V5      284807 non-null  float64\n",
      " 6   V6      284807 non-null  float64\n",
      " 7   V7      284807 non-null  float64\n",
      " 8   V8      284807 non-null  float64\n",
      " 9   V9      284807 non-null  float64\n",
      " 10  V10     284807 non-null  float64\n",
      " 11  V11     284807 non-null  float64\n",
      " 12  V12     284807 non-null  float64\n",
      " 13  V13     284807 non-null  float64\n",
      " 14  V14     284807 non-null  float64\n",
      " 15  V15     284807 non-null  float64\n",
      " 16  V16     284807 non-null  float64\n",
      " 17  V17     284807 non-null  float64\n",
      " 18  V18     284807 non-null  float64\n",
      " 19  V19     284807 non-null  float64\n",
      " 20  V20     284807 non-null  float64\n",
      " 21  V21     284807 non-null  float64\n",
      " 22  V22     284807 non-null  float64\n",
      " 23  V23     284807 non-null  float64\n",
      " 24  V24     284807 non-null  float64\n",
      " 25  V25     284807 non-null  float64\n",
      " 26  V26     284807 non-null  float64\n",
      " 27  V27     284807 non-null  float64\n",
      " 28  V28     284807 non-null  float64\n",
      " 29  Amount  284807 non-null  float64\n",
      " 30  Class   284807 non-null  int64  \n",
      "dtypes: float64(30), int64(1)\n",
      "memory usage: 67.4 MB\n"
     ]
    }
   ],
   "source": [
    "dataframe.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b36569bb-aed4-4dae-a85a-bc54912962ec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    284315\n",
       "1       492\n",
       "Name: Class, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe['Class'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2455e265-77a5-4eae-885b-a77bef875f30",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Balance the Dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "dee35ef6-c112-4c49-aca3-c2e2259eb913",
   "metadata": {},
   "outputs": [],
   "source": [
    "nonFraud = dataframe[dataframe['Class']==0]\n",
    "fraud = dataframe[dataframe['Class']==1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "1d3247de-a0a1-492e-bc5b-e8701eba6867",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((284315, 31), (492, 31))"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nonFraud.shape, fraud.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "edd01ae9-a10c-469e-ab31-f59e5e43a26b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(492, 31)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nonFraud = nonFraud.sample(fraud.shape[0])\n",
    "nonFraud.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c9df7478-a067-45e1-8374-f02bcafba607",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>...</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>406.0</td>\n",
       "      <td>-2.312227</td>\n",
       "      <td>1.951992</td>\n",
       "      <td>-1.609851</td>\n",
       "      <td>3.997906</td>\n",
       "      <td>-0.522188</td>\n",
       "      <td>-1.426545</td>\n",
       "      <td>-2.537387</td>\n",
       "      <td>1.391657</td>\n",
       "      <td>-2.770089</td>\n",
       "      <td>...</td>\n",
       "      <td>0.517232</td>\n",
       "      <td>-0.035049</td>\n",
       "      <td>-0.465211</td>\n",
       "      <td>0.320198</td>\n",
       "      <td>0.044519</td>\n",
       "      <td>0.177840</td>\n",
       "      <td>0.261145</td>\n",
       "      <td>-0.143276</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>472.0</td>\n",
       "      <td>-3.043541</td>\n",
       "      <td>-3.157307</td>\n",
       "      <td>1.088463</td>\n",
       "      <td>2.288644</td>\n",
       "      <td>1.359805</td>\n",
       "      <td>-1.064823</td>\n",
       "      <td>0.325574</td>\n",
       "      <td>-0.067794</td>\n",
       "      <td>-0.270953</td>\n",
       "      <td>...</td>\n",
       "      <td>0.661696</td>\n",
       "      <td>0.435477</td>\n",
       "      <td>1.375966</td>\n",
       "      <td>-0.293803</td>\n",
       "      <td>0.279798</td>\n",
       "      <td>-0.145362</td>\n",
       "      <td>-0.252773</td>\n",
       "      <td>0.035764</td>\n",
       "      <td>529.00</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4462.0</td>\n",
       "      <td>-2.303350</td>\n",
       "      <td>1.759247</td>\n",
       "      <td>-0.359745</td>\n",
       "      <td>2.330243</td>\n",
       "      <td>-0.821628</td>\n",
       "      <td>-0.075788</td>\n",
       "      <td>0.562320</td>\n",
       "      <td>-0.399147</td>\n",
       "      <td>-0.238253</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.294166</td>\n",
       "      <td>-0.932391</td>\n",
       "      <td>0.172726</td>\n",
       "      <td>-0.087330</td>\n",
       "      <td>-0.156114</td>\n",
       "      <td>-0.542628</td>\n",
       "      <td>0.039566</td>\n",
       "      <td>-0.153029</td>\n",
       "      <td>239.93</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6986.0</td>\n",
       "      <td>-4.397974</td>\n",
       "      <td>1.358367</td>\n",
       "      <td>-2.592844</td>\n",
       "      <td>2.679787</td>\n",
       "      <td>-1.128131</td>\n",
       "      <td>-1.706536</td>\n",
       "      <td>-3.496197</td>\n",
       "      <td>-0.248778</td>\n",
       "      <td>-0.247768</td>\n",
       "      <td>...</td>\n",
       "      <td>0.573574</td>\n",
       "      <td>0.176968</td>\n",
       "      <td>-0.436207</td>\n",
       "      <td>-0.053502</td>\n",
       "      <td>0.252405</td>\n",
       "      <td>-0.657488</td>\n",
       "      <td>-0.827136</td>\n",
       "      <td>0.849573</td>\n",
       "      <td>59.00</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7519.0</td>\n",
       "      <td>1.234235</td>\n",
       "      <td>3.019740</td>\n",
       "      <td>-4.304597</td>\n",
       "      <td>4.732795</td>\n",
       "      <td>3.624201</td>\n",
       "      <td>-1.357746</td>\n",
       "      <td>1.713445</td>\n",
       "      <td>-0.496358</td>\n",
       "      <td>-1.282858</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.379068</td>\n",
       "      <td>-0.704181</td>\n",
       "      <td>-0.656805</td>\n",
       "      <td>-1.632653</td>\n",
       "      <td>1.488901</td>\n",
       "      <td>0.566797</td>\n",
       "      <td>-0.010016</td>\n",
       "      <td>0.146793</td>\n",
       "      <td>1.00</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>979</th>\n",
       "      <td>119133.0</td>\n",
       "      <td>2.109129</td>\n",
       "      <td>0.417976</td>\n",
       "      <td>-2.297894</td>\n",
       "      <td>1.295916</td>\n",
       "      <td>0.828339</td>\n",
       "      <td>-1.642801</td>\n",
       "      <td>1.080829</td>\n",
       "      <td>-0.619398</td>\n",
       "      <td>-0.199579</td>\n",
       "      <td>...</td>\n",
       "      <td>0.291688</td>\n",
       "      <td>1.028434</td>\n",
       "      <td>-0.223460</td>\n",
       "      <td>0.011920</td>\n",
       "      <td>0.871466</td>\n",
       "      <td>-0.191437</td>\n",
       "      <td>-0.056938</td>\n",
       "      <td>-0.080855</td>\n",
       "      <td>1.00</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>980</th>\n",
       "      <td>152070.0</td>\n",
       "      <td>0.035746</td>\n",
       "      <td>0.421419</td>\n",
       "      <td>1.341467</td>\n",
       "      <td>-1.015019</td>\n",
       "      <td>0.434587</td>\n",
       "      <td>-0.038892</td>\n",
       "      <td>0.745805</td>\n",
       "      <td>-0.365224</td>\n",
       "      <td>0.401936</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.125617</td>\n",
       "      <td>0.006409</td>\n",
       "      <td>0.066909</td>\n",
       "      <td>-0.397940</td>\n",
       "      <td>-1.472780</td>\n",
       "      <td>0.036774</td>\n",
       "      <td>-0.258074</td>\n",
       "      <td>-0.219911</td>\n",
       "      <td>6.99</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>981</th>\n",
       "      <td>65531.0</td>\n",
       "      <td>-0.275338</td>\n",
       "      <td>0.973317</td>\n",
       "      <td>1.336548</td>\n",
       "      <td>0.485929</td>\n",
       "      <td>0.204261</td>\n",
       "      <td>-0.955948</td>\n",
       "      <td>0.603334</td>\n",
       "      <td>-0.152652</td>\n",
       "      <td>-0.664599</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.143584</td>\n",
       "      <td>-0.420306</td>\n",
       "      <td>0.063512</td>\n",
       "      <td>0.362499</td>\n",
       "      <td>-0.713173</td>\n",
       "      <td>0.086180</td>\n",
       "      <td>0.136909</td>\n",
       "      <td>0.178064</td>\n",
       "      <td>0.89</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>982</th>\n",
       "      <td>121075.0</td>\n",
       "      <td>1.997760</td>\n",
       "      <td>0.144229</td>\n",
       "      <td>-1.716135</td>\n",
       "      <td>1.204166</td>\n",
       "      <td>0.626906</td>\n",
       "      <td>-0.638031</td>\n",
       "      <td>0.532977</td>\n",
       "      <td>-0.245477</td>\n",
       "      <td>-0.015969</td>\n",
       "      <td>...</td>\n",
       "      <td>0.081983</td>\n",
       "      <td>0.412385</td>\n",
       "      <td>-0.058321</td>\n",
       "      <td>-0.393304</td>\n",
       "      <td>0.465209</td>\n",
       "      <td>-0.486727</td>\n",
       "      <td>-0.020851</td>\n",
       "      <td>-0.074821</td>\n",
       "      <td>14.31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>983</th>\n",
       "      <td>62243.0</td>\n",
       "      <td>-0.534619</td>\n",
       "      <td>1.154315</td>\n",
       "      <td>1.371818</td>\n",
       "      <td>-0.020959</td>\n",
       "      <td>0.027602</td>\n",
       "      <td>-0.773524</td>\n",
       "      <td>0.623376</td>\n",
       "      <td>0.002602</td>\n",
       "      <td>-0.658592</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.178586</td>\n",
       "      <td>-0.504955</td>\n",
       "      <td>0.023662</td>\n",
       "      <td>0.493847</td>\n",
       "      <td>-0.210156</td>\n",
       "      <td>0.028652</td>\n",
       "      <td>0.111641</td>\n",
       "      <td>0.086296</td>\n",
       "      <td>1.79</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>984 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         Time        V1        V2        V3        V4        V5        V6  \\\n",
       "0       406.0 -2.312227  1.951992 -1.609851  3.997906 -0.522188 -1.426545   \n",
       "1       472.0 -3.043541 -3.157307  1.088463  2.288644  1.359805 -1.064823   \n",
       "2      4462.0 -2.303350  1.759247 -0.359745  2.330243 -0.821628 -0.075788   \n",
       "3      6986.0 -4.397974  1.358367 -2.592844  2.679787 -1.128131 -1.706536   \n",
       "4      7519.0  1.234235  3.019740 -4.304597  4.732795  3.624201 -1.357746   \n",
       "..        ...       ...       ...       ...       ...       ...       ...   \n",
       "979  119133.0  2.109129  0.417976 -2.297894  1.295916  0.828339 -1.642801   \n",
       "980  152070.0  0.035746  0.421419  1.341467 -1.015019  0.434587 -0.038892   \n",
       "981   65531.0 -0.275338  0.973317  1.336548  0.485929  0.204261 -0.955948   \n",
       "982  121075.0  1.997760  0.144229 -1.716135  1.204166  0.626906 -0.638031   \n",
       "983   62243.0 -0.534619  1.154315  1.371818 -0.020959  0.027602 -0.773524   \n",
       "\n",
       "           V7        V8        V9  ...       V21       V22       V23  \\\n",
       "0   -2.537387  1.391657 -2.770089  ...  0.517232 -0.035049 -0.465211   \n",
       "1    0.325574 -0.067794 -0.270953  ...  0.661696  0.435477  1.375966   \n",
       "2    0.562320 -0.399147 -0.238253  ... -0.294166 -0.932391  0.172726   \n",
       "3   -3.496197 -0.248778 -0.247768  ...  0.573574  0.176968 -0.436207   \n",
       "4    1.713445 -0.496358 -1.282858  ... -0.379068 -0.704181 -0.656805   \n",
       "..        ...       ...       ...  ...       ...       ...       ...   \n",
       "979  1.080829 -0.619398 -0.199579  ...  0.291688  1.028434 -0.223460   \n",
       "980  0.745805 -0.365224  0.401936  ... -0.125617  0.006409  0.066909   \n",
       "981  0.603334 -0.152652 -0.664599  ... -0.143584 -0.420306  0.063512   \n",
       "982  0.532977 -0.245477 -0.015969  ...  0.081983  0.412385 -0.058321   \n",
       "983  0.623376  0.002602 -0.658592  ... -0.178586 -0.504955  0.023662   \n",
       "\n",
       "          V24       V25       V26       V27       V28  Amount  Class  \n",
       "0    0.320198  0.044519  0.177840  0.261145 -0.143276    0.00      1  \n",
       "1   -0.293803  0.279798 -0.145362 -0.252773  0.035764  529.00      1  \n",
       "2   -0.087330 -0.156114 -0.542628  0.039566 -0.153029  239.93      1  \n",
       "3   -0.053502  0.252405 -0.657488 -0.827136  0.849573   59.00      1  \n",
       "4   -1.632653  1.488901  0.566797 -0.010016  0.146793    1.00      1  \n",
       "..        ...       ...       ...       ...       ...     ...    ...  \n",
       "979  0.011920  0.871466 -0.191437 -0.056938 -0.080855    1.00      0  \n",
       "980 -0.397940 -1.472780  0.036774 -0.258074 -0.219911    6.99      0  \n",
       "981  0.362499 -0.713173  0.086180  0.136909  0.178064    0.89      0  \n",
       "982 -0.393304  0.465209 -0.486727 -0.020851 -0.074821   14.31      0  \n",
       "983  0.493847 -0.210156  0.028652  0.111641  0.086296    1.79      0  \n",
       "\n",
       "[984 rows x 31 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe = fraud.append(nonFraud, ignore_index=True)\n",
    "dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6ca4f1d8-eed4-4764-b0da-3b69242b5aca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    492\n",
       "1    492\n",
       "Name: Class, dtype: int64"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe['Class'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "4a7ff9f6-0d1c-4d46-b25e-df45e8e34785",
   "metadata": {},
   "outputs": [],
   "source": [
    "x = dataframe.drop('Class', axis = 1)\n",
    "y = dataframe['Class']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "65d06ee3-11ff-4bd7-a99a-d5342235cf6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0, stratify = y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "cf031c30-c9de-4331-9b37-3d4ed8761f42",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((787, 30), (197, 30))"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape, x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "956d90e7-8b5d-45d4-9e4f-82515c7c2839",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "x_train = scaler.fit_transform(x_train)\n",
    "x_test = scaler.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "83cee749-71ef-4aa9-968c-4427e1c5c67c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train = y_train.to_numpy()\n",
    "y_test = y_test.to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "a4c4cb62-c65a-460e-86d7-9b097c995d46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(787, 30)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d98533fb-260d-4920-b2bd-60b461e57935",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)\n",
    "x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "01adf73f-69fe-4d10-9a38-6c00e78364d3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((787, 30, 1), (197, 30, 1))"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape, x_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "8f7f70a3-7bba-4f4d-9af1-cd506161f5fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Build The CNN ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "0638965f-814c-4057-ae99-f53f98bbc376",
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs = 20\n",
    "model = Sequential()\n",
    "model.add(Conv1D(32,2,activation='relu', input_shape = x_train[0].shape))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "\n",
    "model.add(Conv1D(64,2,activation='relu', input_shape = x_train[0].shape))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Dropout(0.5))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(64, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "\n",
    "\n",
    "model.add(Dense(1, activation='sigmoid'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "3f3bf3a2-c176-43b6-a9f6-bbf0d74796ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_3\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv1d_5 (Conv1D)            (None, 29, 32)            96        \n",
      "_________________________________________________________________\n",
      "batch_normalization_4 (Batch (None, 29, 32)            128       \n",
      "_________________________________________________________________\n",
      "dropout_6 (Dropout)          (None, 29, 32)            0         \n",
      "_________________________________________________________________\n",
      "conv1d_6 (Conv1D)            (None, 28, 64)            4160      \n",
      "_________________________________________________________________\n",
      "batch_normalization_5 (Batch (None, 28, 64)            256       \n",
      "_________________________________________________________________\n",
      "dropout_7 (Dropout)          (None, 28, 64)            0         \n",
      "_________________________________________________________________\n",
      "flatten_2 (Flatten)          (None, 1792)              0         \n",
      "_________________________________________________________________\n",
      "dense_4 (Dense)              (None, 64)                114752    \n",
      "_________________________________________________________________\n",
      "dropout_8 (Dropout)          (None, 64)                0         \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 1)                 65        \n",
      "=================================================================\n",
      "Total params: 119,457\n",
      "Trainable params: 119,265\n",
      "Non-trainable params: 192\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "3f8eefd8-9b1f-4ada-9e1d-6a0b2c9b04c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "76fae16b-047b-4fa6-9269-a5202d944962",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "25/25 [==============================] - 1s 9ms/step - loss: 0.7915 - accuracy: 0.7116 - val_loss: 0.5744 - val_accuracy: 0.7970\n",
      "Epoch 2/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.5852 - accuracy: 0.7853 - val_loss: 0.5335 - val_accuracy: 0.7665\n",
      "Epoch 3/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.4498 - accuracy: 0.8386 - val_loss: 0.4994 - val_accuracy: 0.7716\n",
      "Epoch 4/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.4445 - accuracy: 0.8501 - val_loss: 0.4660 - val_accuracy: 0.7817\n",
      "Epoch 5/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.4088 - accuracy: 0.8590 - val_loss: 0.4329 - val_accuracy: 0.7919\n",
      "Epoch 6/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3255 - accuracy: 0.8729 - val_loss: 0.3920 - val_accuracy: 0.8528\n",
      "Epoch 7/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3079 - accuracy: 0.8983 - val_loss: 0.3575 - val_accuracy: 0.8680\n",
      "Epoch 8/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2843 - accuracy: 0.9022 - val_loss: 0.3236 - val_accuracy: 0.8832\n",
      "Epoch 9/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2894 - accuracy: 0.8983 - val_loss: 0.2966 - val_accuracy: 0.8832\n",
      "Epoch 10/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3426 - accuracy: 0.8856 - val_loss: 0.2759 - val_accuracy: 0.8883\n",
      "Epoch 11/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2870 - accuracy: 0.8983 - val_loss: 0.2598 - val_accuracy: 0.8985\n",
      "Epoch 12/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2242 - accuracy: 0.9276 - val_loss: 0.2451 - val_accuracy: 0.8985\n",
      "Epoch 13/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2624 - accuracy: 0.9009 - val_loss: 0.2316 - val_accuracy: 0.9188\n",
      "Epoch 14/20\n",
      "25/25 [==============================] - 0s 9ms/step - loss: 0.2445 - accuracy: 0.9187 - val_loss: 0.2187 - val_accuracy: 0.9188\n",
      "Epoch 15/20\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2428 - accuracy: 0.9238 - val_loss: 0.2090 - val_accuracy: 0.9188\n",
      "Epoch 16/20\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2428 - accuracy: 0.9161 - val_loss: 0.2039 - val_accuracy: 0.9188\n",
      "Epoch 17/20\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2318 - accuracy: 0.9199 - val_loss: 0.1989 - val_accuracy: 0.9188\n",
      "Epoch 18/20\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2131 - accuracy: 0.9276 - val_loss: 0.1931 - val_accuracy: 0.9188\n",
      "Epoch 19/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2195 - accuracy: 0.9327 - val_loss: 0.1908 - val_accuracy: 0.9188\n",
      "Epoch 20/20\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2415 - accuracy: 0.9199 - val_loss: 0.1898 - val_accuracy: 0.9239\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "e424b32c-b002-4b82-b08d-183a27ae17b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_learningCurve(history, epoch):\n",
    "    #Plot training & validation accuracy values \n",
    "    epoch_range = range(1, epoch+1)\n",
    "    plt.plot(epoch_range, history.history['accuracy'])\n",
    "    plt.plot(epoch_range, history.history['val_accuracy'])\n",
    "    plt.title('Model_accuracy')\n",
    "    plt.ylabel('Accuracy')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.legend(['Train', 'Val'], loc='upper left')\n",
    "    plt.show()\n",
    "    \n",
    "    \n",
    "    #Plot Training and Validation Loss Values\n",
    "    plt.plot(epoch_range, history.history['loss'])\n",
    "    plt.plot(epoch_range, history.history['val_loss'])\n",
    "    plt.title('Model Loss')\n",
    "    plt.ylabel('Loss')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.legend(['Train', 'Val'], loc='upper left')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "71ef287b-4142-4824-a19e-ca774fe51acc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_learningCurve(history, epochs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce396a91-464e-479c-bccb-137350756a33",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Adding MaxPool"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "11960d21-f5ba-4c1b-a9b6-e3f76419e1e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "epochs = 50\n",
    "model = Sequential()\n",
    "model.add(Conv1D(32,2,activation='relu', input_shape = x_train[0].shape))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool1D(2))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "\n",
    "model.add(Conv1D(64,2,activation='relu', input_shape = x_train[0].shape))\n",
    "model.add(BatchNormalization())\n",
    "model.add(MaxPool1D(2))\n",
    "model.add(Dropout(0.5))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(64, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "\n",
    "\n",
    "model.add(Dense(1, activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "d1eca2cf-bc63-4f1c-ab95-a56627f44005",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "25/25 [==============================] - 1s 8ms/step - loss: 1.2124 - accuracy: 0.5540 - val_loss: 0.6901 - val_accuracy: 0.5990\n",
      "Epoch 2/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.9999 - accuracy: 0.5997 - val_loss: 0.6513 - val_accuracy: 0.7360\n",
      "Epoch 3/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.8237 - accuracy: 0.6811 - val_loss: 0.6178 - val_accuracy: 0.7107\n",
      "Epoch 4/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.5902 - accuracy: 0.7433 - val_loss: 0.5902 - val_accuracy: 0.7208\n",
      "Epoch 5/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.6090 - accuracy: 0.7421 - val_loss: 0.5642 - val_accuracy: 0.7208\n",
      "Epoch 6/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.5567 - accuracy: 0.7764 - val_loss: 0.5377 - val_accuracy: 0.7259\n",
      "Epoch 7/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.4977 - accuracy: 0.8005 - val_loss: 0.5074 - val_accuracy: 0.7462\n",
      "Epoch 8/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.4408 - accuracy: 0.8323 - val_loss: 0.4767 - val_accuracy: 0.7513\n",
      "Epoch 9/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.4802 - accuracy: 0.8094 - val_loss: 0.4451 - val_accuracy: 0.7716\n",
      "Epoch 10/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.4638 - accuracy: 0.8005 - val_loss: 0.4164 - val_accuracy: 0.7868\n",
      "Epoch 11/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.4142 - accuracy: 0.8513 - val_loss: 0.3867 - val_accuracy: 0.8274\n",
      "Epoch 12/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3608 - accuracy: 0.8577 - val_loss: 0.3621 - val_accuracy: 0.8376\n",
      "Epoch 13/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3835 - accuracy: 0.8374 - val_loss: 0.3423 - val_accuracy: 0.8528\n",
      "Epoch 14/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3782 - accuracy: 0.8602 - val_loss: 0.3235 - val_accuracy: 0.8731\n",
      "Epoch 15/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3456 - accuracy: 0.8691 - val_loss: 0.3092 - val_accuracy: 0.8731\n",
      "Epoch 16/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3454 - accuracy: 0.8780 - val_loss: 0.2984 - val_accuracy: 0.8731\n",
      "Epoch 17/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3807 - accuracy: 0.8691 - val_loss: 0.2910 - val_accuracy: 0.8782\n",
      "Epoch 18/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3114 - accuracy: 0.8844 - val_loss: 0.2838 - val_accuracy: 0.8782\n",
      "Epoch 19/50\n",
      "25/25 [==============================] - 0s 3ms/step - loss: 0.3213 - accuracy: 0.8793 - val_loss: 0.2790 - val_accuracy: 0.8782\n",
      "Epoch 20/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3275 - accuracy: 0.8653 - val_loss: 0.2764 - val_accuracy: 0.8934\n",
      "Epoch 21/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3388 - accuracy: 0.8793 - val_loss: 0.2723 - val_accuracy: 0.8934\n",
      "Epoch 22/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3060 - accuracy: 0.8780 - val_loss: 0.2733 - val_accuracy: 0.8934\n",
      "Epoch 23/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3475 - accuracy: 0.8679 - val_loss: 0.2726 - val_accuracy: 0.8883\n",
      "Epoch 24/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3341 - accuracy: 0.8818 - val_loss: 0.2681 - val_accuracy: 0.8934\n",
      "Epoch 25/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3194 - accuracy: 0.8793 - val_loss: 0.2665 - val_accuracy: 0.8934\n",
      "Epoch 26/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3034 - accuracy: 0.8767 - val_loss: 0.2649 - val_accuracy: 0.8934\n",
      "Epoch 27/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3020 - accuracy: 0.8945 - val_loss: 0.2637 - val_accuracy: 0.8934\n",
      "Epoch 28/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2635 - accuracy: 0.9009 - val_loss: 0.2583 - val_accuracy: 0.8934\n",
      "Epoch 29/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2738 - accuracy: 0.8996 - val_loss: 0.2552 - val_accuracy: 0.8934\n",
      "Epoch 30/50\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2969 - accuracy: 0.8818 - val_loss: 0.2569 - val_accuracy: 0.8934\n",
      "Epoch 31/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3233 - accuracy: 0.8895 - val_loss: 0.2541 - val_accuracy: 0.8985\n",
      "Epoch 32/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2636 - accuracy: 0.9111 - val_loss: 0.2537 - val_accuracy: 0.8985\n",
      "Epoch 33/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3003 - accuracy: 0.8920 - val_loss: 0.2535 - val_accuracy: 0.8985\n",
      "Epoch 34/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.3129 - accuracy: 0.8945 - val_loss: 0.2535 - val_accuracy: 0.8985\n",
      "Epoch 35/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2820 - accuracy: 0.9034 - val_loss: 0.2492 - val_accuracy: 0.8985\n",
      "Epoch 36/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2948 - accuracy: 0.8780 - val_loss: 0.2489 - val_accuracy: 0.8985\n",
      "Epoch 37/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2671 - accuracy: 0.9009 - val_loss: 0.2461 - val_accuracy: 0.9036\n",
      "Epoch 38/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2646 - accuracy: 0.9060 - val_loss: 0.2480 - val_accuracy: 0.9036\n",
      "Epoch 39/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2776 - accuracy: 0.8971 - val_loss: 0.2479 - val_accuracy: 0.9036\n",
      "Epoch 40/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2148 - accuracy: 0.9187 - val_loss: 0.2474 - val_accuracy: 0.9036\n",
      "Epoch 41/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2550 - accuracy: 0.9136 - val_loss: 0.2482 - val_accuracy: 0.9036\n",
      "Epoch 42/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2423 - accuracy: 0.9098 - val_loss: 0.2464 - val_accuracy: 0.9036\n",
      "Epoch 43/50\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2397 - accuracy: 0.9085 - val_loss: 0.2455 - val_accuracy: 0.9036\n",
      "Epoch 44/50\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2452 - accuracy: 0.9072 - val_loss: 0.2456 - val_accuracy: 0.9036\n",
      "Epoch 45/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2938 - accuracy: 0.8996 - val_loss: 0.2459 - val_accuracy: 0.9036\n",
      "Epoch 46/50\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2443 - accuracy: 0.9174 - val_loss: 0.2440 - val_accuracy: 0.9036\n",
      "Epoch 47/50\n",
      "25/25 [==============================] - 0s 5ms/step - loss: 0.2888 - accuracy: 0.8933 - val_loss: 0.2410 - val_accuracy: 0.9036\n",
      "Epoch 48/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2538 - accuracy: 0.9111 - val_loss: 0.2392 - val_accuracy: 0.9036\n",
      "Epoch 49/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2418 - accuracy: 0.9098 - val_loss: 0.2362 - val_accuracy: 0.9086\n",
      "Epoch 50/50\n",
      "25/25 [==============================] - 0s 4ms/step - loss: 0.2722 - accuracy: 0.9034 - val_loss: 0.2348 - val_accuracy: 0.9086\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])\n",
    "history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=1)\n",
    "plot_learningCurve(history, epochs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "13f70e94-43a1-496c-9850-86eec5d67f96",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
