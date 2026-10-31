
# Project Introduction
The Renewable Asset Fault Anomaly Detection Co-Pilot is an AI - powered application designed to monitor 
solar PV and wind turbines by analysing time - series telemetry data. It detects anomalous sensor readings and leverages Generative AI to identify potential causes , explain abnormal behaviour and recommend inspection , troubleshooting and maintenance actions.

The goal is to help operators understand asset anomalies and make informed maintenance decisions.


## Problem Statement 
Anomaly Detection in Time Series Data(Telemetry Data) and Possible Cause Detection and Seek for Appropriate Meaures 



## The Components of the Sytem
### 1. The FrontEnd - Co-Pilot UI 
    - The Operator Facing UI recieves Telematory Inputs

### 2. An Inference Layer-Baesd on Rolling Median/MAD algorithm
    -  Powerful Statistical Model Working in this layer for Anomlay Detection
    
### 3. Gen AI RAG System
    - A strong RAG system to investigate for anomlaus behavoiur form knowledge sources



### Key Features
1. Intelligent Anomaly Detection: Identifies abnormal operational behaviour in solar inverters and wind turbine generators using historical telemetry data and asset-specific machine learning models.

2. AI-Powered Root Cause Analysis: Analyzes detected anomalies and generates contextual explanations, potential causes, and actionable recommendations to support maintenance decisions.

3. Retrieval-Augmented Generation (RAG): Retrieves relevant technical knowledge from research papers and domain-specific documents to provide evidence-informed diagnostic responses.

4. Interactive AI Copilot: Provides an operator-friendly interface for submitting asset telemetry, triggering anomaly analysis, and viewing structured, AI-generated diagnostic insights.


### The Like to the Deployment in Huggig Face - https://huggingface.co/spaces/Roshan-58/Anomaly_Detection_GenAI_Assignment_Project_Greenko/tree/main


# Assumptions
1. Asset-Specific Operating Characteristics: Solar inverters and wind turbine generators exhibit different operating characteristics. Therefore, anomaly detection is performed separately for each asset type rather than applying a common detection approach to both.

2. Solar Power Generation and Irradiance Relationship: Solar power generation is expected to depend substantially on available solar irradiance, subject to equipment condition. Zero or low power output under negligible irradiance is considered normal operating behaviour, whereas unexpectedly low or zero power during high irradiance may indicate an anomaly.

3. Wind Power Generation and Wind Conditions: Wind turbine power output is evaluated in the context of prevailing wind conditions and the asset's historical operating behaviour. Anomalies are identified by considering deviations from expected generation patterns under similar conditions.



# How It Works
1. Telemetry Submission: The operator submits asset telemetry through the AI Copilot interface.

2. Data Processing: The FastAPI backend forwards the telemetry to the inference system for data validation, preprocessing, and feature engineering.

3. Anomaly Detection: The prepared telemetry is analyzed using a Rolling Median and MAD-based approach to identify abnormal operating behaviour.

4. AI-Powered Diagnosis: If an anomaly is detected, the corresponding telemetry and relevant features are passed to the LLM, which uses the RAG knowledge base to identify potential causes.

5. Diagnostic Recommendations: The Copilot presents possible causes, recommended inspection checks, and preventive measures to the operator.



### Technology Stack

#### Generative AI & RAG
Python, NumPy, Pandas, Docker, Sentence Transformers, BAAI/bge-small-en-v1.5, ChromaDB, LlamaIndex Markdown Splitter, BM25, Cross-Encoder, OpenAI GPT-5.4 mini, OpenAI Python Client

##### Machine Learning & Anomaly Detection
Python, NumPy, Pandas, Scikit-learn

##### AI Copilot & Backend
HTML5, CSS3, JavaScript, FastAPI


-----Some more information are available in 'Info_Docs' folder in the main directory ----

#### Prerequisites
- Python 3.10+
- VS Code or any Python-compatible IDE
- Git
- OpenAI API Key
- Modern web browser
- requirements.txt (contains the required Python dependencies)