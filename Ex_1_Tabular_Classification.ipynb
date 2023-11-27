{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# 1. Initializations and Dataset Download"
      ],
      "metadata": {
        "id": "x15poca7jW-Z"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VttBdozUd2PL",
        "outputId": "d490c112-7d71-453d-bece-1f288028a2fd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Please provide your Kaggle credentials to download this dataset. Learn more: http://bit.ly/kaggle-creds\n",
            "Your Kaggle username: omaratef3221\n",
            "Your Kaggle Key: ··········\n",
            "Downloading rice-type-classification.zip to ./rice-type-classification\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 888k/888k [00:00<00:00, 965kB/s]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ],
      "source": [
        "!pip install opendatasets --quiet\n",
        "import opendatasets as od\n",
        "od.download(\"https://www.kaggle.com/datasets/mssmartypants/rice-type-classification\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 2. Imports\n",
        "Lets start by getting all our imports, keep in mind that PyTorch is not automatically detects and trains on GPU, you have to tell it to use cuda. In case you want to train on Mac Silicon replace cuda with mps."
      ],
      "metadata": {
        "id": "l7fKCtFQjakg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch # Torch main framework\n",
        "import torch.nn as nn # Used for getting the NN Layers\n",
        "from torch.optim import Adam # Adam Optimizer\n",
        "from torch.utils.data import Dataset, DataLoader # Dataset class and DataLoader for creatning the objects\n",
        "from torchsummary import summary # Visualize the model layers and number of parameters\n",
        "from sklearn.model_selection import train_test_split # Split the dataset (train, validation, test)\n",
        "from sklearn.metrics import accuracy_score # Calculate the testing Accuracy\n",
        "import matplotlib.pyplot as plt # Plotting the training progress at the end\n",
        "import pandas as pd # Data reading and preprocessing\n",
        "import numpy as np # Mathematical operations\n",
        "\n",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu' # detect the GPU if any, if not use CPU, change cuda to mps if you have a mac"
      ],
      "metadata": {
        "id": "IgTlDLBpeqcq"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3. Dataset\n",
        "Now lets read the dataset, this lab was developed by Google Colab, so dataset downloaded and read from the path shown below. We will be reading the dataset using pandas `read_csv` function, then we will remove the nulls/missing data from our dataframe as a filteration process, keep in mind that this process is essential as missing data will stop the code from training. You can skip dropping the missing values if you are 100% sure that there are no missing values in your data. Also, we dropped the `id` column because it will not affect our classification at all. We printed the output possibilities as we can see its a binary classification. We printed also the data shape (rows, columns) After that we printed the dataset shape and we used the `head()` function to visualize the first 5 rows, this step is optional as it just allows us to see the first 5 rows and will not affect the training process."
      ],
      "metadata": {
        "id": "o4nmUYiVjfMm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data_df = pd.read_csv(\"/content/rice-type-classification/riceClassification.csv\") # Read the data\n",
        "data_df.dropna(inplace = True) # Drop missing/null values\n",
        "data_df.drop([\"id\"], axis =1, inplace = True) # Drop Id column\n",
        "print(\"Output possibilities: \", data_df[\"Class\"].unique()) # Possible Outputs\n",
        "print(\"Data Shape (rows, cols): \", data_df.shape) # Print data shape\n",
        "data_df.head() # Print/visualize the first 5 rows of the data"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 242
        },
        "id": "_ZokM1pVfAGN",
        "outputId": "5a19489f-db59-4a5f-9d5c-f1456d7e2cf3"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Output possibilities:  [1 0]\n",
            "Data Shape (rows, cols):  (18185, 11)\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Area  MajorAxisLength  MinorAxisLength  Eccentricity  ConvexArea  \\\n",
              "0  4537        92.229316        64.012769      0.719916        4677   \n",
              "1  2872        74.691881        51.400454      0.725553        3015   \n",
              "2  3048        76.293164        52.043491      0.731211        3132   \n",
              "3  3073        77.033628        51.928487      0.738639        3157   \n",
              "4  3693        85.124785        56.374021      0.749282        3802   \n",
              "\n",
              "   EquivDiameter    Extent  Perimeter  Roundness  AspectRation  Class  \n",
              "0      76.004525  0.657536    273.085   0.764510      1.440796      1  \n",
              "1      60.471018  0.713009    208.317   0.831658      1.453137      1  \n",
              "2      62.296341  0.759153    210.012   0.868434      1.465950      1  \n",
              "3      62.551300  0.783529    210.657   0.870203      1.483456      1  \n",
              "4      68.571668  0.769375    230.332   0.874743      1.510000      1  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-18fb680a-57db-4df4-920e-b09a7a4547d5\" class=\"colab-df-container\">\n",
              "    <div>\n",
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
              "      <th>Area</th>\n",
              "      <th>MajorAxisLength</th>\n",
              "      <th>MinorAxisLength</th>\n",
              "      <th>Eccentricity</th>\n",
              "      <th>ConvexArea</th>\n",
              "      <th>EquivDiameter</th>\n",
              "      <th>Extent</th>\n",
              "      <th>Perimeter</th>\n",
              "      <th>Roundness</th>\n",
              "      <th>AspectRation</th>\n",
              "      <th>Class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>4537</td>\n",
              "      <td>92.229316</td>\n",
              "      <td>64.012769</td>\n",
              "      <td>0.719916</td>\n",
              "      <td>4677</td>\n",
              "      <td>76.004525</td>\n",
              "      <td>0.657536</td>\n",
              "      <td>273.085</td>\n",
              "      <td>0.764510</td>\n",
              "      <td>1.440796</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2872</td>\n",
              "      <td>74.691881</td>\n",
              "      <td>51.400454</td>\n",
              "      <td>0.725553</td>\n",
              "      <td>3015</td>\n",
              "      <td>60.471018</td>\n",
              "      <td>0.713009</td>\n",
              "      <td>208.317</td>\n",
              "      <td>0.831658</td>\n",
              "      <td>1.453137</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>3048</td>\n",
              "      <td>76.293164</td>\n",
              "      <td>52.043491</td>\n",
              "      <td>0.731211</td>\n",
              "      <td>3132</td>\n",
              "      <td>62.296341</td>\n",
              "      <td>0.759153</td>\n",
              "      <td>210.012</td>\n",
              "      <td>0.868434</td>\n",
              "      <td>1.465950</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3073</td>\n",
              "      <td>77.033628</td>\n",
              "      <td>51.928487</td>\n",
              "      <td>0.738639</td>\n",
              "      <td>3157</td>\n",
              "      <td>62.551300</td>\n",
              "      <td>0.783529</td>\n",
              "      <td>210.657</td>\n",
              "      <td>0.870203</td>\n",
              "      <td>1.483456</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3693</td>\n",
              "      <td>85.124785</td>\n",
              "      <td>56.374021</td>\n",
              "      <td>0.749282</td>\n",
              "      <td>3802</td>\n",
              "      <td>68.571668</td>\n",
              "      <td>0.769375</td>\n",
              "      <td>230.332</td>\n",
              "      <td>0.874743</td>\n",
              "      <td>1.510000</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-18fb680a-57db-4df4-920e-b09a7a4547d5')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-18fb680a-57db-4df4-920e-b09a7a4547d5 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-18fb680a-57db-4df4-920e-b09a7a4547d5');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-ce0e3dce-2fa0-4fac-9c07-d1c3dbb0795b\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-ce0e3dce-2fa0-4fac-9c07-d1c3dbb0795b')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-ce0e3dce-2fa0-4fac-9c07-d1c3dbb0795b button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 4. Data Preprocessing\n",
        "Now as you saw previously, data values are so big which may cause bad results. Its a crucial steps to normalize the dataset before we proceed. Lets normalize the dataset in the cell below."
      ],
      "metadata": {
        "id": "VMzuXfJnbBpr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "original_df = data_df.copy() # Creating a copy of the original Dataframe to use to normalize inference\n",
        "\n",
        "for column in data_df.columns:\n",
        "    data_df[column] = data_df[column]/data_df[column].abs().max() # Divide by the maximum of the column which will make max value of each column is 1\n",
        "data_df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "waDVfvecMiGK",
        "outputId": "b43f8861-309f-4196-f094-4499e7bddaef"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       Area  MajorAxisLength  MinorAxisLength  Eccentricity  ConvexArea  \\\n",
              "0  0.444368         0.503404         0.775435      0.744658    0.424873   \n",
              "1  0.281293         0.407681         0.622653      0.750489    0.273892   \n",
              "2  0.298531         0.416421         0.630442      0.756341    0.284520   \n",
              "3  0.300979         0.420463         0.629049      0.764024    0.286791   \n",
              "4  0.361704         0.464626         0.682901      0.775033    0.345385   \n",
              "\n",
              "   EquivDiameter    Extent  Perimeter  Roundness  AspectRation  Class  \n",
              "0       0.666610  0.741661   0.537029   0.844997      0.368316    1.0  \n",
              "1       0.530370  0.804230   0.409661   0.919215      0.371471    1.0  \n",
              "2       0.546380  0.856278   0.412994   0.959862      0.374747    1.0  \n",
              "3       0.548616  0.883772   0.414262   0.961818      0.379222    1.0  \n",
              "4       0.601418  0.867808   0.452954   0.966836      0.386007    1.0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8b9d42cf-ecad-459e-9194-1b3d443c4f26\" class=\"colab-df-container\">\n",
              "    <div>\n",
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
              "      <th>Area</th>\n",
              "      <th>MajorAxisLength</th>\n",
              "      <th>MinorAxisLength</th>\n",
              "      <th>Eccentricity</th>\n",
              "      <th>ConvexArea</th>\n",
              "      <th>EquivDiameter</th>\n",
              "      <th>Extent</th>\n",
              "      <th>Perimeter</th>\n",
              "      <th>Roundness</th>\n",
              "      <th>AspectRation</th>\n",
              "      <th>Class</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.444368</td>\n",
              "      <td>0.503404</td>\n",
              "      <td>0.775435</td>\n",
              "      <td>0.744658</td>\n",
              "      <td>0.424873</td>\n",
              "      <td>0.666610</td>\n",
              "      <td>0.741661</td>\n",
              "      <td>0.537029</td>\n",
              "      <td>0.844997</td>\n",
              "      <td>0.368316</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.281293</td>\n",
              "      <td>0.407681</td>\n",
              "      <td>0.622653</td>\n",
              "      <td>0.750489</td>\n",
              "      <td>0.273892</td>\n",
              "      <td>0.530370</td>\n",
              "      <td>0.804230</td>\n",
              "      <td>0.409661</td>\n",
              "      <td>0.919215</td>\n",
              "      <td>0.371471</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.298531</td>\n",
              "      <td>0.416421</td>\n",
              "      <td>0.630442</td>\n",
              "      <td>0.756341</td>\n",
              "      <td>0.284520</td>\n",
              "      <td>0.546380</td>\n",
              "      <td>0.856278</td>\n",
              "      <td>0.412994</td>\n",
              "      <td>0.959862</td>\n",
              "      <td>0.374747</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.300979</td>\n",
              "      <td>0.420463</td>\n",
              "      <td>0.629049</td>\n",
              "      <td>0.764024</td>\n",
              "      <td>0.286791</td>\n",
              "      <td>0.548616</td>\n",
              "      <td>0.883772</td>\n",
              "      <td>0.414262</td>\n",
              "      <td>0.961818</td>\n",
              "      <td>0.379222</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.361704</td>\n",
              "      <td>0.464626</td>\n",
              "      <td>0.682901</td>\n",
              "      <td>0.775033</td>\n",
              "      <td>0.345385</td>\n",
              "      <td>0.601418</td>\n",
              "      <td>0.867808</td>\n",
              "      <td>0.452954</td>\n",
              "      <td>0.966836</td>\n",
              "      <td>0.386007</td>\n",
              "      <td>1.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8b9d42cf-ecad-459e-9194-1b3d443c4f26')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-8b9d42cf-ecad-459e-9194-1b3d443c4f26 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-8b9d42cf-ecad-459e-9194-1b3d443c4f26');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-52329d7b-6a7e-4992-9215-ce57fc625722\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-52329d7b-6a7e-4992-9215-ce57fc625722')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-52329d7b-6a7e-4992-9215-ce57fc625722 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 5. Data Splitting\n",
        "We will detect the inputs and the outputs of the data which are X and Y respectively.\n",
        "\n",
        "Then we will split our data into the following:\n",
        "\n",
        "* Training Size 70%\n",
        "* Validation Size 15%\n",
        "* Testing Size 15%\n",
        "\n",
        "We will do this by splitting our data twice using the train_test_split function in sklearn the function takes inputs, outputs and the testing size. After that we will print the training, validation and testing shapes and sizes. Then we will print the new shapes of our data."
      ],
      "metadata": {
        "id": "jE9tvEBqmL5E"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "X = np.array(data_df.iloc[:,:-1]) # Get the inputs, all rows and all columns except last column (output)\n",
        "Y = np.array(data_df.iloc[:, -1]) # Get the ouputs, all rows and last column only (output column)\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3) # Create the training split\n",
        "X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5) # Create the validation split\n",
        "\n",
        "print(\"Training set is: \", X_train.shape[0], \" rows which is \", round(X_train.shape[0]/data_df.shape[0],4)*100, \"%\") # Print training shape\n",
        "print(\"Validation set is: \",X_val.shape[0], \" rows which is \", round(X_val.shape[0]/data_df.shape[0],4)*100, \"%\") # Print validation shape\n",
        "print(\"Testing set is: \",X_test.shape[0], \" rows which is \", round(X_test.shape[0]/data_df.shape[0],4)*100, \"%\") # Print testing shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "04A0-y-BfUoX",
        "outputId": "074b7d49-2e3c-4e27-fed8-be10a4dfeba4"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Training set is:  12729  rows which is  70.0 %\n",
            "Validation set is:  2728  rows which is  15.0 %\n",
            "Testing set is:  2728  rows which is  15.0 %\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 6. Dataset Object\n",
        "Now we will create the dataset object. This part is not complex but its a bit tricky. We need to convert our dataset to PyTorch Dataset object as it will be more efficient during training, you can use the dataset as its, but lets keep things professional and efficient. First we define our class that will be taking the main Dataset class with the concept of inheritance. Let's make the concept simpler. There is a big class that PyTorch, this class has several functions inside it, we will recreate that class and modify some functions to match our needs.\n",
        "\n",
        "In the cell below, we rebuilt the constructor function which is `__init__`. We put X and Y as a parameters to this function which are the inputs and outputs respectively, then inside the function we define the inputs and converting it to tensors, then converting the outputs to tensors and we make the numbers as a `float32` Additionally, we moved all our data to cuda device. Then we modified the `__len__` and the `__getitem__` to match our needs which gets the specific length/shape of the data, and the data of specific row in our data respectively."
      ],
      "metadata": {
        "id": "1qABq32QnQ6-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class dataset(Dataset):\n",
        "    def __init__(self, X, Y):\n",
        "        self.X = torch.tensor(X, dtype = torch.float32).to(device)\n",
        "        self.Y = torch.tensor(Y, dtype = torch.float32).to(device)\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.X)\n",
        "    def __getitem__(self, index):\n",
        "        return self.X[index], self.Y[index]\n",
        "\n",
        "training_data = dataset(X_train, y_train)\n",
        "validation_data = dataset(X_val, y_val)\n",
        "testing_data = dataset(X_test, y_test)"
      ],
      "metadata": {
        "id": "iju76hoWfaHe"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 7. Training Hyper Parameters\n",
        "Now we are setting the training hyperparameters, we defined some variables which are the batch size, number of training epochs, Hidden Neurons and learning rate."
      ],
      "metadata": {
        "id": "6rm0laL0n34L"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "BATCH_SIZE = 32\n",
        "EPOCHS = 10\n",
        "HIDDEN_NEURONS = 10\n",
        "LR = 1e-3"
      ],
      "metadata": {
        "id": "8C6bsRjEg08S"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 8. Data Loaders\n",
        "This concept may seem complicated, but its very easy, PyTorch provides a class called DataLoader which allows us to create objects of it to simplify the training.\n",
        "\n",
        "Dataloader is an object that we can loop through it to train according to batches. When we start training, we loop through epochs, if you skip the batch size it means that the amount of training data in one batch is equal to the complete amount of training data, this method is not efficient and in most of the cases you need to train through using batches. Dataloader allows you to loop through the batches easily during the training. When you create a dataloader. You define the batch size and enable the shuffle to randomize the data and then you can loop through it in each epoch to train normally.\n",
        "\n",
        "In the next cell, we defined a dataloader for each of our data (training, validation and testing)."
      ],
      "metadata": {
        "id": "PoeZE7D2oCe9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle= True)\n",
        "validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle= True)\n",
        "testing_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle= True)"
      ],
      "metadata": {
        "id": "6zs9yu6BgkR1"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 9. Model Class\n",
        "Creating a model in PyTorch seems not very straight forward in the beggining, but if you understand how machine learning and deep learning works, it will be easy for you to understand PyTorch structure easily.\n",
        "\n",
        "In the next cell we define a new class MyModel that inherits from nn.Module like we did for the dataset. Remember, in simple terms, we want to redefine some functions in the class to match our needs. In the constructor which is `__init__` and we give it the bert model. Then super`(MyModel, self).__init__()` This line calls the constructor of the parent class nn.Module to ensure it's properly initialized. Which means we have the original constructor together with our part of it!\n",
        "\n",
        "Then, we create our layers, a linear layer which represents the input and having the input size of 10 which is the number of columns of the input and the output of the number of hidden neurons, next layer is the output layer which have the input of hidden neurons and one output since we have a binary classification. Finally we have the activation function which is the sigmoid. Our model is pretty simple and straight forward, we are using a simple dataset and we just want to see how PyTorch can be used to build that.\n",
        "\n",
        "In the function forward, this function is the forward propagation of the model, how is the data flowing inside the model from the input to the output. This means we can control this completely. That's how PyTorch is so customizable! In the below cell, we define the flow as follows, starting by input layer and followed by the output layer then the activation layer."
      ],
      "metadata": {
        "id": "SAbIDVA_oKpg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class MyModel(nn.Module):\n",
        "    def __init__(self):\n",
        "\n",
        "        super(MyModel, self).__init__()\n",
        "\n",
        "        self.input_layer = nn.Linear(X.shape[1], HIDDEN_NEURONS)\n",
        "        self.linear = nn.Linear(HIDDEN_NEURONS, 1)\n",
        "        self.sigmoid = nn.Sigmoid()\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.input_layer(x)\n",
        "        x = self.linear(x)\n",
        "        x = self.sigmoid(x)\n",
        "        return x"
      ],
      "metadata": {
        "id": "Q0Wo9UO2gwA3"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 10. Model Creation\n",
        "Now lets create our model and move it to the assigned device (cuda if you have GPU or the CPU if you don't have any GPUs). Additionally, we will print the a `summary` of the model using the function summary which will take our model and the input size."
      ],
      "metadata": {
        "id": "h4rlzDFNbZ0X"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = MyModel().to(device)\n",
        "summary(model, (X.shape[1],))"
      ],
      "metadata": {
        "id": "DRUUWbpohF7L",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "aef37c47-b246-4697-c8f8-90126128ef83"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "----------------------------------------------------------------\n",
            "        Layer (type)               Output Shape         Param #\n",
            "================================================================\n",
            "            Linear-1                   [-1, 10]             110\n",
            "            Linear-2                    [-1, 1]              11\n",
            "           Sigmoid-3                    [-1, 1]               0\n",
            "================================================================\n",
            "Total params: 121\n",
            "Trainable params: 121\n",
            "Non-trainable params: 0\n",
            "----------------------------------------------------------------\n",
            "Input size (MB): 0.00\n",
            "Forward/backward pass size (MB): 0.00\n",
            "Params size (MB): 0.00\n",
            "Estimated Total Size (MB): 0.00\n",
            "----------------------------------------------------------------\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 11. Loss and Optimizer\n",
        "In the next cell, we will create the loss function which will be Binary Cross entropy and the optimizer `Adam` which will take the model parameters/weights and the learning rate."
      ],
      "metadata": {
        "id": "wX_UTkNSb8Kf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "criterion = nn.BCELoss()\n",
        "optimizer = Adam(model.parameters(), lr= LR)"
      ],
      "metadata": {
        "id": "LzFKRv9oiOSI"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 12. Training\n",
        "Now comes the exciting part. As we mentioned several times, nothing is complex here if you know how deep learning works. You just need to know PyTorch syntax. We start by initializing the for loop with the specified number of epochs. But before that we define 4 lists and inside the loop we define another 4 variables.\n",
        "\n",
        "## Variables:\n",
        "* `total_acc_train`: here we will keep tracking of the training accuracy progress during each epoch, we calculate the accuracy during the batch size and we print it in the end for tracking the accuracy on each epoch.\n",
        "\n",
        "* `total_loss_train`: here we will keep tracking of the training loss progress during each epoch, we calculate the accuracy during the batch size and we use the loss value to optimize and modify the model parameters.\n",
        "\n",
        "* `total_acc_val`: here we will keep tracking of the validation accuracy progress during each epoch, we calculate the accuracy during the batch size and we print it in the end for tracking the accuracy on each epoch and help us know if there is any overfitting.\n",
        "\n",
        "* `total_loss_val`: here we will keep tracking of the validation loss progress during each epoch, we calculate the accuracy during the batch size.\n",
        "\n",
        "## Lists:\n",
        "* `total_acc_train_plot`: We append the losses of the training accuracy to visualize them at the end.\n",
        "\n",
        "* `total_loss_train_plot`: We append the losses of the training to visualize them at the end.\n",
        "\n",
        "* `total_acc_validation_plot`: We append the losses of the validation accuracy to visualize them at the end.\n",
        "\n",
        "* `total_loss_validation_plot`: We append the losses of the validation to visualize them at the end.\n",
        "\n",
        "## Training\n",
        "Then, we start to loop through the training dataloaders, we use the enumerate functionality to loop through data and indices at the same time. We are not using the indices here, but lets kept it just if you want to try different stuff with the loop or debug. In the second loop, we start by getting our data from the data loader, then we move the inputs and labels to the cuda device. We allow the model to make a prediction or what is called forward propagation, then we get the output of the model and compare it with our original output using the loss criteration. We use the squeeze function to modify the shape of the output and make it returns a tensor with all specified dimensions of input of size 1 removed. we add the loss amount to `total_loss_train`. Additionally, we get the accuracy by comparing the correct batch with the predicted batch and we add it to the total_acc_train. Then we do the `batch_loss.backward()` which makes the backpropagation and we use the optimizer to do a step on the weights using `optimizer.step()` and then we reset the optimizer gradients using `optimizer.zero_grad()` which is a very important step that has to be done before proceeding.\n",
        "\n",
        "## Validation\n",
        "After that we exit the batch loop (train dataloader loop) and we start with the validation. Don't forget that we are still in the same epoch. In side that we start by with `torch.no_grad()` which means that we need the model to do predicitons without being trained. We just need to see the validation preformance. Then we do the same steps which are predicting and calculating loss and accuracy and storing these values.\n",
        "\n",
        "At the end we print after each epoch the epoch number, training loss, training accuracy, validation loss and validation accuracy. We use the printing of \"=\" signs just for making the printing output looks clean."
      ],
      "metadata": {
        "id": "wvGMerrzcI-e"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "total_loss_train_plot = []\n",
        "total_loss_validation_plot = []\n",
        "total_acc_train_plot = []\n",
        "total_acc_validation_plot = []\n",
        "\n",
        "for epoch in range(EPOCHS):\n",
        "    total_acc_train = 0\n",
        "    total_loss_train = 0\n",
        "    total_acc_val = 0\n",
        "    total_loss_val = 0\n",
        "    ## Training and Validation\n",
        "    for data in train_dataloader:\n",
        "\n",
        "        inputs, labels = data\n",
        "\n",
        "        prediction = model(inputs).squeeze(1)\n",
        "\n",
        "        batch_loss = criterion(prediction, labels)\n",
        "\n",
        "        total_loss_train += batch_loss.item()\n",
        "\n",
        "        acc = ((prediction).round() == labels).sum().item()\n",
        "\n",
        "        total_acc_train += acc\n",
        "\n",
        "        batch_loss.backward()\n",
        "        optimizer.step()\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "    ## Validation\n",
        "    with torch.no_grad():\n",
        "        for data in validation_dataloader:\n",
        "            inputs, labels = data\n",
        "\n",
        "            prediction = model(inputs).squeeze(1)\n",
        "\n",
        "            batch_loss = criterion(prediction, labels)\n",
        "\n",
        "            total_loss_val += batch_loss.item()\n",
        "\n",
        "            acc = ((prediction).round() == labels).sum().item()\n",
        "\n",
        "            total_acc_val += acc\n",
        "\n",
        "    total_loss_train_plot.append(round(total_loss_train/1000, 4))\n",
        "    total_loss_validation_plot.append(round(total_loss_val/1000, 4))\n",
        "    total_acc_train_plot.append(round(total_acc_train/(training_data.__len__())*100, 4))\n",
        "    total_acc_validation_plot.append(round(total_acc_val/(validation_data.__len__())*100, 4))\n",
        "\n",
        "    print(f'''Epoch no. {epoch + 1} Train Loss: {total_loss_train/1000:.4f} Train Accuracy: {(total_acc_train/(training_data.__len__())*100):.4f} Validation Loss: {total_loss_val/1000:.4f} Validation Accuracy: {(total_acc_val/(validation_data.__len__())*100):.4f}''')\n",
        "    print(\"=\"*50)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FuJVA9r3ie2k",
        "outputId": "ae29da0d-923e-4bc5-c0d6-de4b0d017915"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch no. 1 Train Loss: 0.2681 Train Accuracy: 64.4513 Validation Loss: 0.0499 Validation Accuracy: 94.4282\n",
            "==================================================\n",
            "Epoch no. 2 Train Loss: 0.1640 Train Accuracy: 97.4939 Validation Loss: 0.0212 Validation Accuracy: 98.7903\n",
            "==================================================\n",
            "Epoch no. 3 Train Loss: 0.0699 Train Accuracy: 98.3424 Validation Loss: 0.0101 Validation Accuracy: 98.9003\n",
            "==================================================\n",
            "Epoch no. 4 Train Loss: 0.0404 Train Accuracy: 98.4602 Validation Loss: 0.0066 Validation Accuracy: 98.8636\n",
            "==================================================\n",
            "Epoch no. 5 Train Loss: 0.0297 Train Accuracy: 98.5231 Validation Loss: 0.0051 Validation Accuracy: 98.9003\n",
            "==================================================\n",
            "Epoch no. 6 Train Loss: 0.0248 Train Accuracy: 98.5466 Validation Loss: 0.0044 Validation Accuracy: 98.9003\n",
            "==================================================\n",
            "Epoch no. 7 Train Loss: 0.0219 Train Accuracy: 98.5938 Validation Loss: 0.0039 Validation Accuracy: 98.9003\n",
            "==================================================\n",
            "Epoch no. 8 Train Loss: 0.0202 Train Accuracy: 98.6330 Validation Loss: 0.0038 Validation Accuracy: 98.9003\n",
            "==================================================\n",
            "Epoch no. 9 Train Loss: 0.0191 Train Accuracy: 98.6016 Validation Loss: 0.0034 Validation Accuracy: 98.9003\n",
            "==================================================\n",
            "Epoch no. 10 Train Loss: 0.0183 Train Accuracy: 98.6173 Validation Loss: 0.0032 Validation Accuracy: 98.9003\n",
            "==================================================\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 13. Testing\n",
        "Now in this section, we will be testing our model, we will start the code by with `torch.no_grad():` which means that we are telling PyTorch that we don't want to train the model we will be using it only for testing. Then we will declare initial loss and accuracy as zeros, we will start by looping through the testing dataloader like we did before during training. Inside the loop, we got our data and we moved it to our GPU (cuda) and then we ran our model on the data and we got the predictions. After that we get the loss and then we add it to our overall loss, we do the same for accuracy, and finally we print the accuracy."
      ],
      "metadata": {
        "id": "8AaERXtheCKE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "with torch.no_grad():\n",
        "  total_loss_test = 0\n",
        "  total_acc_test = 0\n",
        "  for data in testing_dataloader:\n",
        "    inputs, labels = data\n",
        "\n",
        "    prediction = model(inputs).squeeze(1)\n",
        "\n",
        "    batch_loss_test = criterion((prediction), labels)\n",
        "    total_loss_test += batch_loss_test.item()\n",
        "    acc = ((prediction).round() == labels).sum().item()\n",
        "    total_acc_test += acc\n",
        "\n",
        "print(f\"Accuracy Score is: {round((total_acc_test/X_test.shape[0])*100, 2)}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vPHhFljXnoDB",
        "outputId": "23e482f6-2741-4c9e-af94-fa5612e45980"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy Score is: 98.39%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 14. Plotting and Visualizations (Optional)\n",
        "The results may not be very good, feel free to play with the layers, hyperparameters and text filteration to achieve better performance!"
      ],
      "metadata": {
        "id": "plRG5lj2eFjT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))\n",
        "\n",
        "axs[0].plot(total_loss_train_plot, label='Training Loss')\n",
        "axs[0].plot(total_loss_validation_plot, label='Validation Loss')\n",
        "axs[0].set_title('Training and Validation Loss over Epochs')\n",
        "axs[0].set_xlabel('Epochs')\n",
        "axs[0].set_ylabel('Loss')\n",
        "axs[0].set_ylim([0, 2])\n",
        "axs[0].legend()\n",
        "\n",
        "axs[1].plot(total_acc_train_plot, label='Training Accuracy')\n",
        "axs[1].plot(total_acc_validation_plot, label='Validation Accuracy')\n",
        "axs[1].set_title('Training and Validation Accuracy over Epochs')\n",
        "axs[1].set_xlabel('Epochs')\n",
        "axs[1].set_ylabel('Accuracy')\n",
        "axs[1].set_ylim([0, 100])\n",
        "axs[1].legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 421
        },
        "id": "wtSp2YGUxPZ6",
        "outputId": "6aa955ce-5d8b-451f-994e-aadb02d33677"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1500x500 with 2 Axes>"
            ],
            "image/png": "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LPuUqfpfxXnFVpJzxhVrBc33ByN1KCJGrg52GtK60u0PAACgACtqc+F58+bpiSeeMG8/8cQTatu2rWbOnKnixYtr3759atCgQaaV8vv27dOQIUOyPEd23Hpfk5OT9fbbb2vFihU6c+aMEhISFB8fL2dnZ0mpc/X4+Hh16NAhw/EcHR3NS9U89thj2rt3rw4cOGCxhE5BQiIdAAAgDaditjo4OdBq584rLi6WVa9jx45VSEiI3n//fVWpUkVOTk565JFHlJCQkMkIqW79AiGTyZTlRD+j/oZh5DD6vDV48GAFBgZq3bp1+v777zVlyhRNmzZNI0eOVOfOnXXy5EmtX79eISEh6tChg4YPH67333/fqjGjgEhJkc7/9V/F+cmfpYRYyz5u5VKT5pXapibQ3XysE+sdSEkx9GFIajX6Uy0rqiTV6ABwz2IubKkgzIUPHjyoX375Rbt377b4gtHk5GQtW7ZMQ4YMkZOTU5Zj3G5/RnEmJiam63frfX3vvff00Ucfafr06apTp45cXFw0evRo83293Xml1Ll6/fr1dfr0ac2fP18PPPCAfH19b3ucNZBIBwAASMNkMuXZR0oLkh07dmjAgAHmj5HGxsbqxIkT+RqDu7u7PD09tWfPHrVp00ZS6huAvXv3qn79+rkas0aNGkpKStKuXbvMleSXLl3S4cOHVbNmTXO/8uXL6+mnn9bTTz9tXqdx5MiRkiQPDw8FBwcrODhYrVu31osvvkgi/V5lGNLl4/9VnJ/4KfVLQdNyKiX5tf43ed5OKlUp378UNK+tPxChw+euqrijnQa1ohodAO5lzIXvntzOhefOnas2bdro448/tmifP3++5s6dqyFDhqhu3br64osvdPny5Qyr0uvWravQ0FANHDgww3N4eHhYfCnqkSNHdO3atdte044dO9S9e3dztXxKSor++ecf8zzc399fTk5OCg0N1eDBgzMco06dOmrcuLE+//xzLVmyRLNmzbrtea2l6P1mAAAAIB1/f3+tXr1a3bp1k8lk0uuvv57r5VTuxMiRIzVlyhRVqVJF1atX18yZM3XlypVsrYH4559/qnjx4uZtk8mkevXqqXv37hoyZIg+/fRTFS9eXK+88orKlSun7t27S0pdy7Fz586qWrWqrly5oh9//FE1atSQJI0fP16NGjVSrVq1FB8fr//7v/8z78M9aP8y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22+88YbWrFmjb7/9ViNGjNA///yjFStWKCQkRAEBAZIs59sDBgzQ+PHjtXv3bjVt2lSJiYlasmRJuir1vMbSLgAAAIVQ9erV1aJFC82bN0+SdPToUf30008aNGiQJCk5OVlvvPGG6tSpo1KlSsnV1VWbNm1SeHh4tsY/dOiQypcvb37jIEnNmzdP12/58uVq2bKlvLy85Orqqtdeey3b50h7rnr16pnfOEhSy5YtlZKSosOHD5vbatWqJVvb/6ppvb29df78+RydK+05y5cvb06iS1LNmjVVokQJHTp0SJI0ZswYDR48WAEBAXrnnXcsPuI6atQovfnmm2rZsqUmTJiQqy+0Au62awlJmv1vNfqoB/xlb8fbPwBA0cBc+PZz4eTkZC1cuFBPPPGEue2JJ57QggULlJKSIil1OZTWrVubk+hpnT9/XmfPnlWHDh1ydD0Zady4scV2bGysxo4dqxo1aqhEiRJydXXVoUOHzPdu3759srW1Vdu2bTMcz8fHR127djU//999953i4+P16KOP3nGsWaEiHQAAIK1izqnVMNY6dw4MGjRII0eO1Mcff6z58+ercuXK5snme++9p48++kjTp09XnTp15OLiotGjRyshISHPwt25c6f69eunSZMmKTAw0FzNMm3atDw7R1q3TvBNJpP5TcDdMHHiRD3++ONat26dNmzYoAkTJmjZsmXq0aOHBg8erMDAQK1bt07ff/+9pkyZomnTpmnkyJF3LR4gp77ceVKX4hJUoZSzejQsZ+1wAACFAXPhbCvoc+FNmzbpzJkz6t27t0V7cnKyQkND1bFjRzk5OWV6fFb7JMnGJvUP9IZhmNsyW7M97R8JJGns2LEKCQnR+++/rypVqsjJyUmPPPKI+fm53bklafDgwXryySf14Ycfav78+erdu/dd/7JYShIAAADSMplSP1JqjUc21oRM67HHHpONjY2WLFmiRYsW6amnnjKvEbljxw51795dTzzxhOrVq6dKlSrpn3/+yfbYNWrU0KlTpxQREWFu++WXXyz6/Pzzz/L19dWrr76qxo0by9/fXydPnrToY29vr+Tk5Nuea//+/YqLizO37dixQzY2NqpWrVq2Y86Jm9d36tQpc9vBgwcVFRWlmjVrmtuqVq2q559/Xt9//7169uyp+fPnm/eVL19eTz/9tFavXq0XXnhBn3/++V2J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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 15. Inference\n",
        "Now lets see how we can use the model in inference mode, inference means making predictions like you are in production."
      ],
      "metadata": {
        "id": "GVJbrC7SCldp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "area = float(input(\"Area: \"))/original_df['Area'].abs().max()\n",
        "MajorAxisLength = float(input(\"Major Axis Length: \"))/original_df['MajorAxisLength'].abs().max()\n",
        "MinorAxisLength = float(input(\"Minor Axis Length: \"))/original_df['MinorAxisLength'].abs().max()\n",
        "Eccentricity = float(input(\"Eccentricity: \"))/original_df['Eccentricity'].abs().max()\n",
        "ConvexArea = float(input(\"Convex Area: \"))/original_df['ConvexArea'].abs().max()\n",
        "EquivDiameter = float(input(\"EquivDiameter: \"))/original_df['EquivDiameter'].abs().max()\n",
        "Extent = float(input(\"Extent: \"))/original_df['Extent'].abs().max()\n",
        "Perimeter = float(input(\"Perimeter: \"))/original_df['Perimeter'].abs().max()\n",
        "Roundness = float(input(\"Roundness: \"))/original_df['Roundness'].abs().max()\n",
        "AspectRation = float(input(\"AspectRation: \"))/original_df['AspectRation'].abs().max()\n",
        "\n",
        "my_inputs = [area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation]\n",
        "\n",
        "print(\"=\"*20)\n",
        "model_inputs = torch.Tensor(my_inputs).to(device)\n",
        "prediction = (model(model_inputs))\n",
        "print(prediction)\n",
        "print(\"Class is: \", round(prediction.item()))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z5WFb1OLCnNb",
        "outputId": "d0fb5df6-392d-4c42-e7be-27f5e60bf8cb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Area: 6431.279\n",
            "Major Axis Length: 145.21338\n",
            "Minor Axis Length: 56.902\n",
            "Eccentricity: 0.919981821\n",
            "Convex Area: 6518.93759999\n",
            "EquivDiameter: 90.483541\n",
            "Extent: 0.8506668\n",
            "Perimeter: 329.972\n",
            "Roundness: 0.742255516\n",
            "AspectRation: 2.551696\n",
            "====================\n",
            "tensor([0.8427], device='cuda:0', grad_fn=<SigmoidBackward0>)\n",
            "Class is:  1\n"
          ]
        }
      ]
    }
  ]
}