{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3b73264f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "e925480d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset created successfully!\n"
     ]
    },
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
       "      <th>Date</th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2023-01-01</td>\n",
       "      <td>1800</td>\n",
       "      <td>1820</td>\n",
       "      <td>1790</td>\n",
       "      <td>1810</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2023-01-02</td>\n",
       "      <td>1810</td>\n",
       "      <td>1830</td>\n",
       "      <td>1800</td>\n",
       "      <td>1825</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2023-01-03</td>\n",
       "      <td>1825</td>\n",
       "      <td>1840</td>\n",
       "      <td>1815</td>\n",
       "      <td>1830</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2023-01-04</td>\n",
       "      <td>1830</td>\n",
       "      <td>1850</td>\n",
       "      <td>1820</td>\n",
       "      <td>1840</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2023-01-05</td>\n",
       "      <td>1840</td>\n",
       "      <td>1860</td>\n",
       "      <td>1830</td>\n",
       "      <td>1855</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date  Open  High   Low  Close\n",
       "0  2023-01-01  1800  1820  1790   1810\n",
       "1  2023-01-02  1810  1830  1800   1825\n",
       "2  2023-01-03  1825  1840  1815   1830\n",
       "3  2023-01-04  1830  1850  1820   1840\n",
       "4  2023-01-05  1840  1860  1830   1855"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Create dataset\n",
    "data = pd.DataFrame({\n",
    "    \"Date\": [\n",
    "        \"2023-01-01\",\"2023-01-02\",\"2023-01-03\",\"2023-01-04\",\"2023-01-05\",\n",
    "        \"2023-01-06\",\"2023-01-07\",\"2023-01-08\",\"2023-01-09\",\"2023-01-10\"\n",
    "    ],\n",
    "    \"Open\": [1800,1810,1825,1830,1840,1855,1860,1875,1880,1895],\n",
    "    \"High\": [1820,1830,1840,1850,1860,1870,1880,1890,1900,1910],\n",
    "    \"Low\":  [1790,1800,1815,1820,1830,1845,1850,1865,1870,1885],\n",
    "    \"Close\":[1810,1825,1830,1840,1855,1860,1875,1880,1895,1900]\n",
    "})\n",
    "\n",
    "# Save as CSV\n",
    "data.to_csv(\"gold_price.csv\", index=False)\n",
    "\n",
    "print(\"Dataset created successfully!\")\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7511abdd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "         Date  Open  High   Low  Close\n",
      "0  2023-01-01  1800  1820  1790   1810\n",
      "1  2023-01-02  1810  1830  1800   1825\n",
      "2  2023-01-03  1825  1840  1815   1830\n",
      "3  2023-01-04  1830  1850  1820   1840\n",
      "4  2023-01-05  1840  1860  1830   1855\n"
     ]
    }
   ],
   "source": [
    "data = pd.read_csv(\"gold_price.csv\")\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6f3c2f74",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "471a6ffa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10 entries, 0 to 9\n",
      "Data columns (total 5 columns):\n",
      " #   Column  Non-Null Count  Dtype \n",
      "---  ------  --------------  ----- \n",
      " 0   Date    10 non-null     object\n",
      " 1   Open    10 non-null     int64 \n",
      " 2   High    10 non-null     int64 \n",
      " 3   Low     10 non-null     int64 \n",
      " 4   Close   10 non-null     int64 \n",
      "dtypes: int64(4), object(1)\n",
      "memory usage: 532.0+ bytes\n",
      "None\n",
      "Date     0\n",
      "Open     0\n",
      "High     0\n",
      "Low      0\n",
      "Close    0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(data.info())\n",
    "print(data.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "df1b6064",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data[['Open', 'High', 'Low']]\n",
    "y = data['Close']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "808651c1",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fe8cf447",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d2603376",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "d8d47a1f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 5.889423076922924\n",
      "R2 Score: 0.9951923076923078\n"
     ]
    }
   ],
   "source": [
    "print(\"Mean Squared Error:\", mean_squared_error(y_test, y_pred))\n",
    "print(\"R2 Score:\", r2_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "22526132",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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tQCATZs2oaCgAM8995xc55FHHkFaWhquX7+OZ555Bq6urujXrx/WrFlj0j6CgoKwd+9e7Nu3D0eOHEFJSQkSEhJgaWlpVM/Z2dnoec01Ia+99prRB76FhQWmTZsGAPjjjz8AAJcuXYKTk1OtfddV9lcXL16Ei4vLHZ8GOn/+PL799ttafezbt2+tPv45sDWkjwDQo0cP+bqrP1uyZAn27t2LvXv3GpUXFxdDCFFrXgHAxcVF7lNdasrvdE5reHh4yD/7X375BSUlJVi9ejUkSTKqV1cf62LK79r58+fx008/1fp5dOrUCUII+edxt+9rat94DRHR/7dixQoIIfDVV1/hq6++qrX9008/xVtvvQVzc3M4ODjg999/v+N9WVtby0do/qzmD3eNtWvXwsLCAhs3boS1tbVcfjdLwb29vfHQQw9h5cqVmDp1KlauXAkXFxc5eNUYO3Ysxo4di8rKSmRlZSE+Ph7h4eHo2bMnAgICbrkPSZJMuu7lz0ciAKBr164AgLi4OIwbN67O13h6egIAunTpUufFyaZcsOzg4IBdu3ahurr6jkJR165dMWDAALz99tt1bq8JH126dEFOTs4d9REAHnvsMXz00UfYt2+f0Xx6eHjUWf+ee+6BmZkZCgoKam07d+6c3Pe61BwdvNM5rWFtbX1HP/v6mPK71rVrV6jVaqP/tPx1e427eV9T+8YjRES4edro008/hYeHB9LT02s9Xn31VRQUFOCHH34AcHMpcXp6er2nHwDIy8frOorTs2dPHD9+HJWVlXLZpUuX5BVONVQqFTp06ABzc3O57MqVK/jss8/uarzPPfccsrOzsWvXLnz77beIjIw02sdfxxEcHIx33nkHAHDw4MG72veteHp6onfv3jh06BAeeOCBOh+dOnUCAAwZMgQ//vij0UqjGzduYN26dbfdT1hYGK5evXrbGyNaWVnV+fMbNWoUfvnlF3h4eNTZx5pANGTIEFy+fBlardbo9aZc+A0AsbGx6NixI15++WVcvnz5tvVtbGzg5+eHb775xqjf1dXVWL16NVxdXXHffffV+VpPT084OztjzZo1RisYT58+Xet92ZxM+V0bNWoUTp48iS5dutT586jrppLN+b6mtoFHiIhw8zqFc+fO4Z133qlzOXy/fv2QmJiI5cuXY9SoUXjzzTfxww8/4JFHHsGcOXPQv39/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WVnw8/MDAGRlZSEgIABHjx6Fp6enSf0rLS2FJEkwGAzyvoiIiKh1M/Xzu1VfQxQUFAStVouzZ89CCIH09HQcP34cw4cPBwBUVlZCpVLByspKfo21tTXMzMywa9cuAEBmZiYkSZLDEAD4+/tDkiTs2bOn3n1XVlaitLTU6EFERETtU6sORAkJCfD29oarqyssLS0RGhqKpKQkBAUFAbgZbGxsbPDPf/4TFRUVKC8vx6xZs1BdXY2CggIAQGFhIRwdHWu17ejoiMLCwnr3HR8fL19zJEkS3NzcmmaQRERE1OJafSDKysqCVqvF/v37sWTJEkybNg1bt24FADg4OGD9+vX49ttvYWtrKx8S8/X1hbm5udyOSqWq1bYQos7yGnFxcTAYDPLjzJkzjT9AIiIiahU6tHQH6nPlyhXMmTMHqampGDlyJABgwIAB0Ov1WLx4MUJCQgAAw4YNw8mTJ/HHH3+gQ4cO6Ny5M5ycnODu7g4AcHJyqrXqDAAuXryIbt261bt/Kysro1NxRERE1H612iNEVVVVqKqqgpmZcRfNzc2NVpDV6Nq1Kzp37oxt27bhwoULGDNmDAAgICAABoMBOTk5ct3s7GwYDAYEBgY27SCIiIioTWjRI0RlZWXIy8uTn+fn50Ov18Pe3h49evRAcHAwZs2aBbVaDY1Gg+3bt2PVqlVYunSp/JqVK1fCy8sLDg4OyMzMxD/+8Q/ExsbKq8e8vLwQGhqKqKgoLFu2DAAQHR2NUaNGmbzCjIiIiNq3Fl12n5GRgSFDhtQqj4yMREpKCgoLCxEXF4fNmzejqKgIGo0G0dHRiI2Nla//mT17NlJSUlBUVISePXvixRdfNNoOAEVFRYiJiYFWqwUAjBkzBomJiejcubPJfeWyeyIiorbH1M/vVnMfotaOgYiIiKjtaRf3ISIiIiJqDgxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgtGoh27NiB0aNHw8XFBSqVCmlpaUbby8rKMH36dLi6ukKtVsPLywvJyclGdQoLCxEREQEnJyfY2NjA19cXX331lVGd4uJiREREQJIkSJKEiIgIlJSUNPHoiIiIqK1o0UBUXl4OHx8fJCYm1rk9NjYWOp0Oq1evRm5uLmJjYzFjxgxs2LBBrhMREYFjx45Bq9Xi559/xrhx4zBhwgQcPHhQrhMeHg69Xg+dTgedTge9Xo+IiIgmHx8RERG1DSohhGjpTgCASqVCamoqHn/8cbmsX79+mDBhAubOnSuXDRo0CCNGjMCiRYsAALa2tkhOTjYKOF26dMG7776L559/Hrm5ufD29kZWVhb8/PwAAFlZWQgICMDRo0fh6elpUv9KS0shSRIMBgPs7OwaYcRERETU1Ez9/G7V1xAFBQVBq9Xi7NmzEEIgPT0dx48fx/Dhw43qrFu3DkVFRaiursbatWtRWVmJwYMHAwAyMzMhSZIchgDA398fkiRhz5499e67srISpaWlRg8iIiJqn1p1IEpISIC3tzdcXV1haWmJ0NBQJCUlISgoSK6zbt06XL9+HV26dIGVlRWmTp2K1NRUeHh4ALh5jZGjo2Otth0dHVFYWFjvvuPj4+VrjiRJgpubW+MPkIiIiFqFVh+IsrKyoNVqsX//fixZsgTTpk3D1q1b5TpvvPEGiouLsXXrVuzbtw8zZ87EU089hZ9//lmuo1KparUthKizvEZcXBwMBoP8OHPmTOMOjoiIiFqNDi3dgfpcuXIFc+bMQWpqKkaOHAkAGDBgAPR6PRYvXoyQkBCcPHkSiYmJ+OWXX9C3b18AgI+PD3bu3ImPPvoIH3/8MZycnHD+/Pla7V+8eBHdunWrd/9WVlawsrJqmsERERFRq9JqjxBVVVWhqqoKZmbGXTQ3N0d1dTUAoKKiAgBuWScgIAAGgwE5OTny9uzsbBgMBgQGBjblEIiIiKiNaNEjRGVlZcjLy5Of5+fnQ6/Xw97eHj169EBwcDBmzZoFtVoNjUaD7du3Y9WqVVi6dCkAoE+fPrj33nsxdepULF68GF26dEFaWhq2bNmCjRs3AgC8vLwQGhqKqKgoLFu2DAAQHR2NUaNGmbzCjIiIiNq3Fl12n5GRgSFDhtQqj4yMREpKCgoLCxEXF4fNmzejqKgIGo0G0dHRiI2Nla//OXHiBGbPno1du3ahrKwM9957L1577TWjZfhFRUWIiYmBVqsFAIwZMwaJiYno3LmzyX3lsnsiIqK2x9TP71ZzH6LWjoGIiIio7WkX9yEiIiIiag4MRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4LRqIduzYgdGjR8PFxQUqlQppaWlG28vKyjB9+nS4urpCrVbDy8sLycnJ8vZTp05BpVLV+Vi/fr1cr7i4GBEREZAkCZIkISIiAiUlJc00SiIiImrtWjQQlZeXw8fHB4mJiXVuj42NhU6nw+rVq5Gbm4vY2FjMmDEDGzZsAAC4ubmhoKDA6LFw4ULY2NggLCxMbic8PBx6vR46nQ46nQ56vR4RERHNMkYiIiJq/Tq05M7DwsKMgstfZWZmIjIyEoMHDwYAREdHY9myZdi3bx/Gjh0Lc3NzODk5Gb0mNTUVEyZMgK2tLQAgNzcXOp0OWVlZ8PPzAwB88sknCAgIwLFjx+Dp6dk0gyMiIqI2o1VfQxQUFAStVouzZ89CCIH09HQcP34cw4cPr7P+/v37odfr8fzzz8tlmZmZkCRJDkMA4O/vD0mSsGfPnnr3XVlZidLSUqMHERERtU+tOhAlJCTA29sbrq6usLS0RGhoKJKSkhAUFFRn/eXLl8PLywuBgYFyWWFhIRwdHWvVdXR0RGFhYb37jo+Pl685kiQJbm5udz8gIiIiapVafSDKysqCVqvF/v37sWTJEkybNg1bt26tVffKlSv44osvjI4O1VCpVLXKhBB1lteIi4uDwWCQH2fOnLm7wRAREVGr1aLXEN3KlStXMGfOHKSmpmLkyJEAgAEDBkCv12Px4sUICQkxqv/VV1+hoqICzzzzjFG5k5MTzp8/X6v9ixcv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vb3h6uoKS0tLhIaGIikpCUFBQXKd2NhYBAYGGh0x+rPCwkI4OjrWKnd0dERhYWG9+46Pj5evOZIkCW5ubnc/ICIiImqVWvSU2e0kJCQgKysLWq0WGo0GO3bswLRp0+Ds7IyQkBBotVps27YNBw8evGU7dV1bJISo95ojAIiLi8PMmTPl56WlpQxFRERE7VSrDURXrlzBnDlzkJqaipEjRwIABgwYAL1ej8WLFyMkJATbtm3DyZMn0blzZ6PXPvHEE/jb3/6GjIwMODk54fz587Xav3jxIrp161bv/q2srGBlZdWoYyIiIqLWqdUGoqqqKlRVVcHMzPisnrm5OaqrqwEAs2fPxgsvvGC0vX///njvvfcwevRoAEBAQAAMBgNycnLw0EMPAQCys7NhMBgQGBjYDCMhIiKi1q5FA1FZWRny8vLk5/n5+dDr9bC3t0ePHj0QHByMWbNmQa1WQ6PRYPv27Vi1ahWWLl0KAHBycqrzQuoePXrA3d0dAODl5YXQ0FBERUVh2bJlAG6uVhs1ahRXmBERERGAFg5E+/btw5AhQ+TnNdfsREZGIiUlBWvXrkVcXBwmTZqEoqIiaDQavP3223jxxRcbtJ/PP/8cMTExGDZsGABgzJgx9d77iIiIiJRHJYQQLd2JtqC0tBSSJMFgMMDOzq6lu0NEREQmMPXzu1UvuyciIiJqDgxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgMRERERKR4DERERESkeAxEREREpHgtGoh27NiB0aNHw8XFBSqVCmlpaUbby8rKMH36dLi6ukKtVsPLywvJyclGdaZOnQoPDw+o1Wo4ODhg7NixOHr0qFGd4uJiREREQJIkSJKEiIgIlJSUNPHoiIiIqK1o0UBUXl4OHx8fJCYm1rk9NjYWOp0Oq1evRm5uLmJjYzFjxgxs2LBBrjNo0CCsXLkSubm52LRpE4QQGDZsGG7cuCHXCQ8Ph16vh06ng06ng16vR0RERJOPj4iIiNoGlRBCtHQnAEClUiE1NRWPP/64XNavXz9MmDABc+fOlcsGDRqEESNGYNGiRXW289NPP8HHxwd5eXnw8PBAbm4uvL29kZWVBT8/PwBAVlYWAgICcPToUXh6eprUv9LSUkiSBIPBADs7uzsfKBERETUbUz+/W/U1REFBQdBqtTh79iyEEEhPT8fx48cxfPjwOuuXl5dj5cqVcHd3h5ubGwAgMzMTkiTJYQgA/P39IUkS9uzZU+++KysrUVpaavQgIiKi9qlVB6KEhAR4e3vD1dUVlpaWCA0NRVJSEoKCgozqJSUlwdbWFra2ttDpdNiyZQssLS0BAIWFhXB0dKzVtqOjIwoLC+vdd3x8vHzNkSRJcsAiIiKi9qfVB6KsrCxotVrs378fS5YswbRp07B161ajepMmTcLBgwexfft29O7dG+PHj8fVq1fl7SqVqlbbQog6y2vExcXBYDDIjzNnzjTewIiIiKhV6dDSHajPlStXMGfOHKSmpmLkyJEAgAEDBkCv12Px4sUICQmR69Ycxenduzf8/f1xzz33IDU1FRMnToSTkxPOnz9fq/2LFy+iW7du9e7fysoKVlZWjT8wIiIianVa7RGiqqoqVFVVwczMuIvm5uaorq6+5WuFEKisrAQABAQEwGAwICcnR96enZ0Ng8GAwMDAxu84ERERtTkteoSorKwMeXl58vP8/Hzo9XrY29ujR48eCA4OxqxZs6BWq6HRaLB9+3asWrUKS5cuBQD8+uuvWLduHYYNGwYHBwecPXsW77zzDtRqNUaMGAEA8PLyQmhoKKKiorBs2TIAQHR0NEaNGmXyCjMiIiJq31p02X1GRgaGDBlSqzwyMhIpKSkoLCxEXFwcNm/ejKKiImg0GkRHRyM2NhYqlQrnzp3DCy+8gP3796O4uBjdunXDI488gnnz5hmFnaKiIsTExECr1QIAxowZg8TERHTu3NnkvnLZPRERUdtj6ud3q7kPUWvHQERERNT2tIv7EBERERE1BwYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlI8BiIiIiJSPAYiIiIiUjwGIiIiIlK8Di3dASW7US2Qk1+EC5evwrGTNR5yt4e5maqlu0VERKQ4DEQtRPdLARZ+ewQFhqtymbNkjfmjvRHaz7kFe0ZERKQ8PGXWAnS/FOCl1QeMwhAAFBqu4qXVB6D7paCFekZERKRMDETN7Ea1wMJvj0DUsa2mbOG3R3Cjuq4aRERE1BQYiJpZTn5RrSNDfyYAFBiuIie/qPk6RUREpHAMRM3swuX6w9Cd1CMiIqK7x0DUzBw7WTdqPSIiIrp7DETN7CF3ezhL1qhvcb0KN1ebPeRu35zdIiIiUrQWDUQ7duzA6NGj4eLiApVKhbS0NKPtZWVlmD59OlxdXaFWq+Hl5YXk5GR5e1FREWbMmAFPT0907NgRPXr0QExMDAwGg1E7xcXFiIiIgCRJkCQJERERKCkpaYYR1mZupsL80d4AUCsU1TyfP9qb9yMiIiJqRi0aiMrLy+Hj44PExMQ6t8fGxkKn02H16tXIzc1FbGwsZsyYgQ0bNgAAzp07h3PnzmHx4sX4+eefkZKSAp1Oh+eff96onfDwcOj1euh0Ouh0Ouj1ekRERDT5+OoT2s8ZyZN94SQZnxZzkqyRPNmX9yEiIiJqZiohRKtY361SqZCamorHH39cLuvXrx8mTJiAuXPnymWDBg3CiBEjsGjRojrbWb9+PSZPnozy8nJ06NABubm58Pb2RlZWFvz8/AAAWVlZCAgIwNGjR+Hp6WlS/0pLSyFJEgwGA+zs7O58oH/CO1UTERE1LVM/v1v1NURBQUHQarU4e/YshBBIT0/H8ePHMXz48HpfUzPgDh1u3oQ7MzMTkiTJYQgA/P39IUkS9uzZU287lZWVKC0tNXo0NnMzFQI8umDs/d0R4NGFYYiIiKiFtOpAlJCQAG9vb7i6usLS0hKhoaFISkpCUFBQnfUvXbqERYsWYerUqXJZYWEhHB0da9V1dHREYWFhvfuOj4+XrzmSJAlubm53PyAiIiJqlVp9IMrKyoJWq8X+/fuxZMkSTJs2DVu3bq1Vt7S0FCNHjoS3tzfmz59vtE2lqn3kRQhRZ3mNuLg4GAwG+XHmzJm7HxARERG1Sq32y12vXLmCOXPmIDU1FSNHjgQADBgwAHq9HosXL0ZISIhc9/LlywgNDYWtrS1SU1NhYWEhb3NycsL58+drtX/x4kV069at3v1bWVnBysqqEUdERERErVWrPUJUVVWFqqoqmJkZd9Hc3BzV1dXy89LSUgwbNgyWlpbQarWwtjZeuRUQEACDwYCcnBy5LDs7GwaDAYGBgU07CCIiImoTWvQIUVlZGfLy8uTn+fn50Ov1sLe3R48ePRAcHIxZs2ZBrVZDo9Fg+/btWLVqFZYuXQrg5pGhYcOGoaKiAqtXrza6+NnBwQHm5ubw8vJCaGgooqKisGzZMgBAdHQ0Ro0aZfIKMyIiImrfWnTZfUZGBoYMGVKrPDIyEikpKSgsLERcXBw2b96MoqIiaDQaREdHIzY2FiqVqt7XAzfDVc+ePQHcvIFjTEwMtFotAGDMmDFITExE586dTe5rUyy7JyIioqZl6ud3q7kPUWvHQERERNT2tIv7EBERERE1BwYiIiIiUrxWu+y+tak5s9gUd6wmIiKiplHzuX27K4QYiEx0+fJlAOAdq4mIiNqgy5cvQ5KkerfzomoTVVdX49y5c+jUqdMt73DdFpWWlsLNzQ1nzpxR5AXjHL+yxw9wDpQ+foBz0J7HL4TA5cuX4eLiUuvehn/GI0QmMjMzg6ura0t3o0nZ2dm1u1+EhuD4lT1+gHOg9PEDnIP2Ov5bHRmqwYuqiYiISPEYiIiIiEjxGIgIVlZWmD9/vmK/zJbjV/b4Ac6B0scPcA6UPn6AF1UTERER8QgREREREQMRERERKR4DERERESkeAxEREREpHgNRO7Bjxw6MHj0aLi4uUKlUSEtLM9peVlaG6dOnw9XVFWq1Gl5eXkhOTjaqM3XqVHh4eECtVsPBwQFjx47F0aNHjeoUFxcjIiICkiRBkiRERESgpKSkiUdnmsaYgxpCCISFhdXZTmudg8YY/+DBg6FSqYweTz/9tFGd1jp+oPHeA5mZmRg6dChsbGzQuXNnDB48GFeuXJG3t9Y5uNvxnzp1qtbPv+axfv16uV57HT8AFBYWIiIiAk5OTrCxsYGvry+++uorozqtdfxA48zByZMn8fe//x0ODg6ws7PD+PHjcf78eaM6rXkO7gYDUTtQXl4OHx8fJCYm1rk9NjYWOp0Oq1evRm5uLmJjYzFjxgxs2LBBrjNo0CCsXLkSubm52LRpE4QQGDZsGG7cuCHXCQ8Ph16vh06ng06ng16vR0RERJOPzxSNMQc13n///Xq/nqW1zkFjjT8qKgoFBQXyY9myZUbbW+v4gcaZg8zMTISGhmLYsGHIycnB3r17MX36dKPb/bfWObjb8bu5uRn97AsKCrBw4ULY2NggLCxMbqe9jh8AIiIicOzYMWi1Wvz8888YN24cJkyYgIMHD8p1Wuv4gbufg/LycgwbNgwqlQrbtm3D7t27ce3aNYwePRrV1dVyO615Du6KoHYFgEhNTTUq69u3r3jzzTeNynx9fcUbb7xRbzuHDh0SAEReXp4QQogjR44IACIrK0uuk5mZKQCIo0ePNt4AGsHdzIFerxeurq6ioKCgVjttZQ7udPzBwcHiH//4R73ttpXxC3Hnc+Dn53fL34u2MgeN9Xfg/vvvF1OmTJGft/fx29jYiFWrVhnVsbe3F//973+FEG1n/ELc2Rxs2rRJmJmZCYPBIG8vKioSAMSWLVuEEG1rDhqKR4gUICgoCFqtFmfPnoUQAunp6Th+/DiGDx9eZ/3y8nKsXLkS7u7ucHNzA3Dzf86SJMHPz0+u5+/vD0mSsGfPnmYZx90wZQ4qKiowceJEJCYmwsnJqVYbbXkOTH0PfP755+jatSv69u2L1157DZcvX5a3teXxA7efgwsXLiA7OxuOjo4IDAxEt27dEBwcjF27dslttOU5aOjfgf3790Ov1+P555+Xy9r7+IOCgrBu3ToUFRWhuroaa9euRWVlJQYPHgygbY8fuP0cVFZWQqVSGd2c0draGmZmZvLvQVufg1thIFKAhIQEeHt7w9XVFZaWlggNDUVSUhKCgoKM6iUlJcHW1ha2trbQ6XTYsmULLC0tAdw8t+7o6FirbUdHRxQWFjbLOO6GKXMQGxuLwMBAjB07ts422vIcmDL+SZMmYc2aNcjIyMDcuXPx9ddfY9y4cfL2tjx+4PZz8OuvvwIAFixYgKioKOh0Ovj6+uLRRx/FiRMnALTtOTD170CN5cuXw8vLC4GBgXJZex//unXrcP36dXTp0gVWVlaYOnUqUlNT4eHhAaBtjx+4/Rz4+/vDxsYG//znP1FRUYHy8nLMmjUL1dXVKCgoAND25+BW+G33CpCQkICsrCxotVpoNBrs2LED06ZNg7OzM0JCQuR6kyZNwmOPPYaCggIsXrwY48ePx+7du2FtbQ0AdV5XI4So93qb1uR2c6DVarFt2zajawXq0lbnwJT3QFRUlFy/X79+6N27Nx544AEcOHAAvr6+ANru+IHbz0HNNRJTp07Fc889BwAYOHAgfvzxR6xYsQLx8fEA2u4cmPp3AACuXLmCL774AnPnzq3VTnse/xtvvIHi4mJs3boVXbt2RVpaGp566ins3LkT/fv3B9B2xw/cfg4cHBywfv16vPTSS0hISICZmRkmTpwIX19fmJuby+205Tm4pZY4T0dNB385b1xRUSEsLCzExo0bjeo9//zzYvjw4fW2U1lZKTp27Ci++OILIYQQy5cvF5Ik1aonSZJYsWJFo/S9sdzJHPzjH/8QKpVKmJubyw8AwszMTAQHBwsh2s4cNNZ7oLq6WlhYWIi1a9cKIdrO+IW4szn49ddfBQDx2WefGdUZP368CA8PF0K0nTm42/fAqlWrhIWFhbhw4YJReXsef15engAgfvnlF6M6jz76qJg6daoQou2MX4i7fw9cvHhRFBcXCyGE6Natm3j33XeFEG1rDhqKp8zauaqqKlRVVRmtkgEAc3Nzo1UDdRFCoLKyEgAQEBAAg8GAnJwceXt2djYMBoPRIfXWyJQ5mD17Nn766Sfo9Xr5AQDvvfceVq5cCaDtzsGdvgcOHz6MqqoqODs7A2i74wdMm4OePXvCxcUFx44dM6pz/PhxaDQaAG13Dhr6Hli+fDnGjBkDBwcHo/L2PP6KigoAuGWdtjp+oOHvga5du6Jz587Ytm0bLly4gDFjxgBo23NwWy2dyOjuXb58WRw8eFAcPHhQABBLly4VBw8eFKdPnxZC3Fw91LdvX5Geni5+/fVXsXLlSmFtbS2SkpKEEEKcPHlS/Pvf/xb79u0Tp0+fFnv27BFjx44V9vb24vz58/J+QkNDxYABA0RmZqbIzMwU/fv3F6NGjWqRMf/V3c5BXVDHKo3WOgd3O/68vDyxcOFCsXfvXpGfny++++470adPHzFw4EBx/fp1eT+tdfxCNM574L333hN2dnZi/fr14sSJE+KNN94Q1tbW8mpLIVrvHDTW78CJEyeESqUSP/zwQ537aa/jv3btmrj33nvF3/72N5GdnS3y8vLE4sWLhUqlEt999528n9Y6fiEa5z2wYsUKkZmZKfLy8sRnn30m7O3txcyZM43205rn4G4wELUD6enpAkCtR2RkpBBCiIKCAvHss88KFxcXYW1tLTw9PcWSJUtEdXW1EEKIs2fPirCwMOHo6CgsLCyEq6urCA8Pr7WE8tKlS2LSpEmiU6dOolOnTmLSpEnyIdWWdrdzUJe6AlFrnYO7Hf9vv/0mHnnkEWFvby8sLS2Fh4eHiImJEZcuXTLaT2sdvxCN9x6Ij48Xrq6uomPHjiIgIEDs3LnTaHtrnYPGGn9cXJxwdXUVN27cqHM/7Xn8x48fF+PGjROOjo6iY8eOYsCAAbWW4bfW8QvROHPwz3/+U3Tr1k1YWFiI3r171/keac1zcDdUQgjRNMeeiIiIiNoGXkNEREREisdARERERIrHQERERESKx0BEREREisdARERERIrHQERERESKx0BEREREisdARESEm19YmZaW1ujtDh48GK+88kqjt0tEjYuBiIia1Z49e2Bubo7Q0NAGv7Znz554//33G79TJnj22WehUqmgUqlgYWGBXr164bXXXkN5efktX/fNN99g0aJFzdRLIrpTDERE1KxWrFiBGTNmYNeuXfjtt99aujsNEhoaioKCAvz666946623kJSUhNdee63OulVVVQAAe3t7dOrUqTm7SUR3gIGIiJpNeXk5vvzyS7z00ksYNWoUUlJSatXRarV44IEHYG1tja5du2LcuHEAbp56On36NGJjY+UjNQCwYMEC3H///UZtvP/+++jZs6f8fO/evXjsscfQtWtXSJKE4OBgHDhwoMH9t7KygpOTE9zc3BAeHo5JkybJp9lq+rFixQr06tULVlZWEELUOmVWWVmJ119/HW5ubrCyskLv3r2xfPlyefuRI0cwYsQI2Nraolu3boiIiMAff/zR4L4SUcMwEBFRs1m3bh08PT3h6emJyZMnY+XKlfjz1yl+9913GDduHEaOHImDBw/ixx9/xAMPPADg5qknV1dXvPnmmygoKEBBQYHJ+718+TIiIyOxc+dOZGVloXfv3hgxYgQuX758V+NRq9XykSAAyMvLw5dffomvv/4aer2+ztc888wzWLt2LRISEpCbm4uPP/4Ytra2AICCggIEBwfj/vvvx759+6DT6XD+/HmMHz/+rvpJRLfXoaU7QETKsXz5ckyePBnAzdNPZWVl+PHHHxESEgIAePvtt/H0009j4cKF8mt8fHwA3Dz1ZG5ujk6dOsHJyalB+x06dKjR82XLluGee+7B9u3bMWrUqDsaS05ODr744gs8+uijctm1a9fw2WefwcHBoc7XHD9+HF9++SW2bNkij7lXr17y9uTkZPj6+uLf//63XLZixQq4ubnh+PHjuO++++6or0R0ezxCRETN4tixY8jJycHTTz8NAOjQoQMmTJiAFStWyHX0er1RwGgsFy5cwIsvvoj77rsPkiRBkiSUlZU1+BqmjRs3wtbWFtbW1ggICMAjjzyCDz/8UN6u0WjqDUPAzfGZm5sjODi4zu379+9Heno6bG1t5UefPn0AACdPnmxQX4moYXiEiIiaxfLly3H9+nV0795dLhNCwMLCAsXFxbjnnnugVqsb3K6ZmZnRaTcARqexgJsrxC5evIj3338fGo0GVlZWCAgIwLVr1xq0ryFDhiA5ORkWFhZwcXGBhYWF0XYbG5tbvv5246uursbo0aPxzjvv1Nrm7OzcoL4SUcPwCBERNbnr169j1apVWLJkCfR6vfw4dOgQNBoNPv/8cwDAgAED8OOPP9bbjqWlJW7cuGFU5uDggMLCQqNQ9Nfrd3bu3ImYmBiMGDECffv2hZWV1R1dqGxjY4N7770XGo2mVhgyRf/+/VFdXY3t27fXud3X1xeHDx9Gz549ce+99xo9bhe2iOjuMBARUZPbuHEjiouL8fzzz6Nfv35GjyeffFJeZTV//nysWbMG8+fPR25uLn7++We8++67cjs9e/bEjh07cPbsWTnQDB48GBcvXsS7776LkydP4qOPPsIPP/xgtP97770Xn332GXJzc5GdnY1Jkybd0dGou9WzZ09ERkZiypQpSEtLQ35+PjIyMvDll18CAF5++WUUFRVh4sSJyMnJwa+//orNmzdjypQptYIgETUuBiIianLLly9HSEgIJEmqte2JJ56AXq/HgQMHMHjwYKxfvx5arRb3338/hg4diuzsbLnum2++iVOnTsHDw0O+VsfLywtJSUn46KOP4OPjg5ycnFr3BlqxYgWKi4sxcOBAREREICYmBo6Ojk076HokJyfjySefxLRp09CnTx9ERUXJN3d0cXHB7t27cePGDQwfPhz9+vXDP/7xD0iSBDMz/rkmakoq8deT70REREQKw/9yEBERkeIxEBEREZHiMRARERGR4jEQERERkeIxEBEREZHiMRARERGR4jEQERERkeIxEBEREZHiMRARERGR4jEQERERkeIxEBEREZHiMRARERGR4v0/ubec0MbbcfwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(y_test, y_pred)\n",
    "plt.xlabel(\"Actual Price\")\n",
    "plt.ylabel(\"Predicted Price\")\n",
    "plt.title(\"Actual vs Predicted Gold Prices\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d41a9f8d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
