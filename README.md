# Master Thesis Data Science and Society 
Title: Predicting Vasovagal Reactions Based on Facial Action Units During Blood Donation: A Machine Learning Approach 

Student: Dionne Spaltman 

Abstract: 
To ensure an adequate blood supply, it is important for blood banks to retain the individuals in their donor pool. However, extreme emotional and physical reactions (vasovagal reactions, VVRs) caused by blood donation can cause donors to not return. To prevent this, the FAINT project is investigating methods to measure the unconscious microexpressions that occur before individuals experience a VVR. This study uniquely utilized video recordings of donors in the donation chair, capturing their facial microexpressions with the OpenFace software. This approach marks the first instance of such methodology in this context. We employed four machine learning models to classify donors into two groups: those with low VVR risk (exhibiting no significant reactions) and those with high VVR risk. We utilized three feature sets for this classification: self-reported VVR measurements taken while donors were in the waiting room, the intensity of their microexpressions, and a combination of both. The Support Vector Machine model, which incorporated the combined features, emerged as the best performer, achieving an F1-score of 0.90 for the negative class (low VVR) and 0.67 for the positive class (high VVR). These results suggest that while the models were proficient in predicting low VVR occurrences, they were less effective for high VVR predictions. Notably, the intensity of microexpressions alone provided minimal predictive power relative to the self-reported VVR measurements.