# Hierarchical Prompt Tuning for System-incremental Log Analysis
System-incremental log analysis, involves the ongoing training of a model using logs from diverse systems to enable effective resolution of log analysis tasks across an expanding array of systems. Existing continual learning methods, which are based on prompt tuning, have shown challenges in insufficient knowledge transfer and increasing catastrophic forgetting. To tackle these challenges, we present LogHPT, a novel continual learning method based on a hierarchical prompt tuning framework specifically tailored for system-incremental log analysis. LogHPT incorporates four types of prompt meticulously crafted to capture log knowledge across various granularities, thereby enhancing knowledge transfer. Subsequently, we employ a key-value mechanism to discern the most suitable prompts for the input logs. Additionally, we integrate a knowledge distillation-based technique to mitigate catastrophic forgetting typically associated with general prompt learning. To evaluate the performance of LogHPT, we conduct comprehensive experiments focusing on two fundamental subtasks: log parsing and log anomaly detection. The results show that LogHPT achieves state-of-the-art (SOTA) performance.

# Requirements:

transformer==4.26.0

pytorch >= 1.12.1


# Log Parsing
python Log_Parsing/main_LP.py

# Log Anomaly Detection
python Anomaly_Detection/main_LAD.py
