THE Evaluations in RAG

1.  The Retrieval Metrics for 3 analomus chunks flagged by the Rolling Median Method Eralier - 

 Retrieval Evaluation Results
Record_ID	Candidate_K	Reranked_K	Relevant_Retrieved	Total_Relevant_Reference	Precision@5	Recall@5
0 WTG_07_19	     10	           5	       0	          10	                      0.0	     0.0
1 WTG_07_200	 10         	5	       0	           10	                      0.0	     0.0
2 WTG_07_217	 10	           5	       0	           10	                       0.0	      0.0


2. The Answer Qukaity Determintaion in RAG 


	Evaluation Metric	         Average Score
0	total_records	               3
1	successful_records	           0.3
2	partially_evaluated_records	      3
3	metrics	                     {'faithfulness': {'mean': None, 'scored_record}}
4	context_metric_counts	        {'context_utilization': 3}



THE Evaluation in Rolling Median/MAD Alogrithm 


The Normal Statistical evaluation- 

"SOLAR_INV_01": {
            "observations": 24,
            "anomalies": 4,
            "anomaly_percentage": 16.67
        },
        "WTG_07": {
            "observations": 24,
            "anomalies": 1,
            "anomaly_percentage": 4.17
        }s