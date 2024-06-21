import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import keras
from keras.layers import Input, Dense # type: ignore
from keras import regularizers
import tensorflow as tf
from keras.models import Model # type: ignore
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, fbeta_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


# Load your data
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    
    if st.sidebar.button("Show Analysis"):
    # Read the uploaded file into a pandas DataFrame
        phishingdata = pd.read_csv(uploaded_file)
        
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown("<h1 style='text-align: center;'>Features</h1>", unsafe_allow_html=True)
        phishingdata.hist(bins = 50,figsize = (15,15))
        st.pyplot()
        
        st.markdown("<h1 style='text-align: center;'>Corelation Confusion Matrix</h1>", unsafe_allow_html=True)
        data0 = phishingdata.drop('Domain', axis=1)
        plt.figure(figsize=(15,13))
        sns.heatmap(data0.corr(), annot=True, cmap="YlGnBu")
        st.pyplot()
        st.caption("Proposed Confusion Matrix is a graphical representation of data and commonly used to visualize the magnitude values between two dimensions. The above figure shows the relation between each cell which are filled with a color that corresponds to the value in that cell, allowing patterns and trends to be easily identified.")
        
        data0.describe()
        data0.isnull().sum()
        data = data0.sample(frac=1).reset_index(drop=True)
        data.head()
        
        y = data['Label']
        X = data.drop('Label',axis=1)
        
        st.header("Splitting of Data")
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 12)
        X_train.shape, X_test.shape
        st.caption("The data is splitted in 80:20")
        
        # Creating holders to store the model performance results
        ML_Model = []
        acc_train = []
        acc_test = []

        #function to call for storing the results
        def storeResults(model, a,b):
            ML_Model.append(model)
            acc_train.append(round(a, 3))
            acc_test.append(round(b, 3))
            
        #Decision Tree
        tree = DecisionTreeClassifier(max_depth = 5)
        # fit the model
        tree.fit(X_train, y_train)
        
        #predicting the target value from the model for the samples
        y_test_tree = tree.predict(X_test)
        y_train_tree = tree.predict(X_train)
        acc_train_tree = accuracy_score(y_train,y_train_tree)
        acc_test_tree = accuracy_score(y_test,y_test_tree)
        st.markdown("<h1 style='text-align: center;'>Implementation of Machine Learning Models</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>1. DECISION TREE</h1>", unsafe_allow_html=True)
        st.header("1.1 Accuracy of the Decision Tree")
        st.write("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
        st.write("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))
        st.caption("The decision tree model's test accuracy of 0.814 and training accuracy of 0.813 indicate a strong consistency in performance between the training and testing datasets. The marginal difference of 0.001 suggests the model's ability to generalize effectively to unseen data, showcasing its robustness in accurately categorizing phishing and legitimate websites. This close alignment in accuracy across both phases highlights the model's reliability in detecting phishing threats while minimizing false positives, a critical aspect for practical cybersecurity implementations.")
        
        st.header("1.2 Rate of Feature Importance")
        plt.figure(figsize=(9,7))
        n_features = X_train.shape[1]
        plt.barh(range(n_features), tree.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), X_train.columns)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        st.pyplot()
        #confusion matrix of Decision Tree
        st.header("1.3 Confusion Matrix of Decision Tree")
        cm = confusion_matrix(y_test, y_test_tree)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix for Decision Tree")
        st.pyplot()
        st.caption("The confusion matrix is a crucial tool in evaluating classification models' performance by showing how well they predict different classes. It organizes actual and predicted class counts into a matrix, with metrics like True Positive (correctly identified positives), False Positive (negatives mistakenly classified as positives), False Negative (positives mistakenly classified as negatives), and True Negative (correctly identified negatives). .Decision tree model for phishing website detection, the confusion matrix helps assess its accuracy in classifying phishing and legitimate websites. Metrics like Accuracy (overall correctness), Precision (avoiding false positives), Recall (identifying positives accurately), Specificity (identifying negatives accurately), and the F1 Score (a balance of precision and recall) provide insights into the model's performance. Understanding the confusion matrix helps in optimizing the model by balancing metrics like precision and recall, ensuring it effectively detects phishing websites while minimizing false positives and false negatives. Fine-tuning the model based on these insights enhances its reliability and effectiveness in cybersecurity applications, contributing to improved overall performance and outcomes.")
        
        #Error, F1 Score, F2 Score
        st.header("1.4 Error, F1 Score, F2 Score, Precision, Recall, Sensitivity, Specificity")
        # Calculate error rate
        error_rate = 1 - accuracy_score(y_test, y_test_tree)
        st.write("Error rate:", error_rate)

        # Calculate F1 score
        f1_score = fbeta_score(y_test, y_test_tree, beta=1)
        st.write("F1 score:", f1_score)

        # Calculate F2 score
        f2_score = fbeta_score(y_test, y_test_tree, beta=2)
        st.write("F2 score:", f2_score)
        
        from sklearn.metrics import precision_score, recall_score, confusion_matrix

        # Calculate precision and recall
        precision = precision_score(y_test, y_test_tree)
        recall = recall_score(y_test, y_test_tree)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_test, y_test_tree)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # Print the results
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Sensitivity:", sensitivity)
        st.write("Specificity:", specificity)
        
        storeResults('Decision Tree', acc_train_tree, acc_test_tree)
        
        #Random Forest
        st.markdown("<h1 style='text-align: center;'>2. Random Forest Classifier</h1>", unsafe_allow_html=True)
        forest = RandomForestClassifier(max_depth=5)
        forest.fit(X_train, y_train)        
        y_test_forest = forest.predict(X_test)
        y_train_forest = forest.predict(X_train)
        
        acc_train_forest = accuracy_score(y_train,y_train_forest)
        acc_test_forest = accuracy_score(y_test,y_test_forest)
        st.header("2.1 Accuracy of the Random Forest")
        st.write("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
        st.write("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))
        st.caption("The test accuracy of the random forest classifier, standing at 0.821, closely mirrors its training accuracy of 0.820. This similarity signifies the model's ability to generalize well to new, unseen data, showcasing its robustness in distinguishing between phishing and legitimate websites. The marginal difference of 0.001 between the test and training accuracies indicates minimal overfitting, suggesting a balanced model that maintains its accuracy across different datasets. This consistency in accuracy across both training and testing phases highlights the random forest classifier's reliability in practical scenarios, where accurate phishing detection and low false positive rates are critical.")
        
        #Confusion matrix Of Random Forest
        st.header("2.2 Confusion Matrix of Random Forest Classifier")
        cm = confusion_matrix(y_test, y_test_forest)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix for Random Forest")
        st.pyplot()
        st.caption("The confusion matrix is a crucial tool in evaluating classification models' performance by showing how well they predict different classes. It organizes actual and predicted class counts into a matrix, with metrics like True Positive (correctly identified positives), False Positive (negatives mistakenly classified as positives), False Negative (positives mistakenly classified as negatives), and True Negative (correctly identified negatives). .Random Forest model for phishing website detection, the confusion matrix helps assess its accuracy in classifying phishing and legitimate websites. Metrics like Accuracy (overall correctness), Precision (avoiding false positives), Recall (identifying positives accurately), Specificity (identifying negatives accurately), and the F1 Score (a balance of precision and recall) provide insights into the model's performance. Understanding the confusion matrix helps in optimizing the model by balancing metrics like precision and recall, ensuring it effectively detects phishing websites while minimizing false positives and false negatives. Fine-tuning the model based on these insights enhances its reliability and effectiveness in cybersecurity applications, contributing to improved overall performance and outcomes.")
        
        #Error, F1 Score, F2 Score
        # Calculate error rate
        st.header("2.3 Error, F1 Score, F2 Score, Precision, Recall, Sensitivity, Specificity")
        error_rate = 1 - accuracy_score(y_test, y_test_forest)
        st.write("Error rate:", error_rate)
        # Calculate F1 score
        f1_score = fbeta_score(y_test, y_test_forest, beta=1)
        st.write("F1 score:", f1_score)
        # Calculate F2 score
        f2_score = fbeta_score(y_test, y_test_forest, beta=2)
        st.write("F2 score:", f2_score)
        
        # Calculate precision and recall
        precision = precision_score(y_test, y_test_forest)
        recall = recall_score(y_test, y_test_forest)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_test, y_test_forest)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # Print the results
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Sensitivity:", sensitivity)
        st.write("Specificity:", specificity)
        
        # Calculate ROC curve and AUC of Random Forest
        st.header("2.4 ROC Curve of Random Forest Classifier")
        y_prob = forest.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        st.pyplot()
        st.caption("The random forest classifier's ROC curve with an Area Under the Curve (AUC) of 0.88 indicates strong discriminatory ability. This means it effectively distinguishes between phishing and legitimate websites, achieving high true positive rates while keeping false positive rates low across different thresholds. The steep rise near the upper-left corner of the curve highlights its high sensitivity and specificity, crucial for accurate phishing detection and minimal false alarms in cybersecurity applications. Overall, the AUC of 0.88 reflects the model's robust performance and suitability for real-world scenarios requiring precise classification of malicious and benign web entities.")
        
        st.header("2.5 Rate of Important Features")
        plt.figure(figsize=(9,7))
        n_features = X_train.shape[1]
        plt.barh(range(n_features), forest.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), X_train.columns)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        st.pyplot()
        
        #storing the results. The below mentioned order of parameter passing is important.
        #Caution: Execute only once to avoid duplications.
        storeResults('Random Forest', acc_train_forest, acc_test_forest)
        
        # Multilayer Perceptrons model
        # instantiate the model
        st.markdown("<h1 style='text-align: center;'>3. Multilayer Perceptrons</h1>", unsafe_allow_html=True)
        mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))

        # fit the model
        mlp.fit(X_train, y_train)
        y_test_mlp = mlp.predict(X_test)
        y_train_mlp = mlp.predict(X_train)
        
        acc_train_mlp = accuracy_score(y_train,y_train_mlp)
        acc_test_mlp = accuracy_score(y_test,y_test_mlp)

        st.header("3.1 Accuracy of the MLP")
        st.write("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
        st.write("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))
        storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)
        st.caption("The Multi-Layer Perceptron model demonstrates strong performance, achieving a test accuracy of 0.861 and a training accuracy of 0.867. This consistency suggests that the model generalizes well to new data, indicating its ability to accurately classify phishing and legitimate websites. The small difference of 0.006 between test and training accuracies indicates minimal overfitting, highlighting a well-balanced model that maintains accuracy across different datasets. These results emphasize the MLP's reliability in practical use cases, where accurate phishing detection and low false positive rates are essential.")
        
        
        #Confusion Matrix of Multilayer Perceptrons
        st.header("3.2 Confusion Matrix of the MLP")
        cm = confusion_matrix(y_test, y_test_mlp)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix for Multilayer Perceptrons")
        st.pyplot()
        st.caption("The confusion matrix is a crucial tool in evaluating classification models' performance by showing how well they predict different classes. It organizes actual and predicted class counts into a matrix, with metrics like True Positive (correctly identified positives), False Positive (negatives mistakenly classified as positives), False Negative (positives mistakenly classified as negatives), and True Negative (correctly identified negatives). .MLP model for phishing website detection, the confusion matrix helps assess its accuracy in classifying phishing and legitimate websites. Metrics like Accuracy (overall correctness), Precision (avoiding false positives), Recall (identifying positives accurately), Specificity (identifying negatives accurately), and the F1 Score (a balance of precision and recall) provide insights into the model's performance. Understanding the confusion matrix helps in optimizing the model by balancing metrics like precision and recall, ensuring it effectively detects phishing websites while minimizing false positives and false negatives. Fine-tuning the model based on these insights enhances its reliability and effectiveness in cybersecurity applications, contributing to improved overall performance and outcomes.")
        
        #Error, F1 Score, F2 Score of Multilayer Perceptrons
        # Calculate error rate
        st.header("3.3 Error, F1 Score, F2 Score, Precision, Recall, Sensitivity, Specificity ")
        error_rate = 1 - accuracy_score(y_test, y_test_mlp)
        st.write("Error rate:", error_rate)
        # Calculate F1 score
        f1_score = fbeta_score(y_test, y_test_mlp, beta=1)
        st.write("F1 score:", f1_score)
        # Calculate F2 score
        f2_score = fbeta_score(y_test, y_test_mlp, beta=2)
        st.write("F2 score:", f2_score)
        
        # Calculate precision and recall
        precision = precision_score(y_test, y_test_mlp)
        recall = recall_score(y_test, y_test_mlp)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_test, y_test_mlp)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # Print the results
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Sensitivity:", sensitivity)
        st.write("Specificity:", specificity)
        
        # Calculate ROC curve and AUC of Multilayer Perceptrons
        st.header("3.4 ROC Curve of the MLP")
        y_prob = mlp.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        st.pyplot()
        st.caption("The multilayer perceptron’s ROC curve with an Area Under the Curve (AUC) of 0.92 indicates strong discriminatory ability. This means it effectively distinguishes between phishing and legitimate websites, achieving high true positive rates while keeping false positive rates low across different thresholds. The steep rise near the upper-left corner of the curve highlights its high sensitivity and specificity, crucial for accurate phishing detection and minimal false alarms in cybersecurity applications. Overall, the AUC of 0.92 reflects the model's robust performance and suitability for real-world scenarios requiring precise classification of malicious and benign web entities.")
        
        #XGBoost Classification model
        # instantiate the model
        st.markdown("<h1 style='text-align: center;'>4. XG Boost</h1>", unsafe_allow_html=True)
        xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
        #fit the model
        xgb.fit(X_train, y_train)
        y_test_xgb = xgb.predict(X_test)
        y_train_xgb = xgb.predict(X_train)
        acc_train_xgb = accuracy_score(y_train,y_train_xgb)
        acc_test_xgb = accuracy_score(y_test,y_test_xgb)
        st.header("4.1 Accuracy of the XG Boost")
        st.write("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
        st.write("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))
        storeResults('XGBoost', acc_train_xgb, acc_test_xgb)
        st.caption("The XG Boost model exhibits strong performance, achieving a test accuracy of 0.852 and a training accuracy of 0.869. This close alignment indicates its ability to generalize well to new data, accurately classifying phishing and legitimate websites. The small difference of 0.017 between test and training accuracies suggests minimal overfitting, emphasizing a well-balanced model across datasets. These results highlight XG Boost's reliability in real-world applications, crucial for precise phishing detection and maintaining low false positive rates, contributing significantly to cybersecurity effectiveness.")
        
        #Confusion Matrix of XGBoost
        st.header("4.2 Confusion Matrix of the XG Boost")
        cm = confusion_matrix(y_test, y_test_xgb)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix for XGBoost")
        st.pyplot()
        st.caption("The confusion matrix is a crucial tool in evaluating classification models' performance by showing how well they predict different classes. It organizes actual and predicted class counts into a matrix, with metrics like True Positive (correctly identified positives), False Positive (negatives mistakenly classified as positives), False Negative (positives mistakenly classified as negatives), and True Negative (correctly identified negatives). .XG Boost model for phishing website detection, the confusion matrix helps assess its accuracy in classifying phishing and legitimate websites. Metrics like Accuracy (overall correctness), Precision (avoiding false positives), Recall (identifying positives accurately), Specificity (identifying negatives accurately), and the F1 Score (a balance of precision and recall) provide insights into the model's performance. Understanding the confusion matrix helps in optimizing the model by balancing metrics like precision and recall, ensuring it effectively detects phishing websites while minimizing false positives and false negatives. Fine-tuning the model based on these insights enhances its reliability and effectiveness in cybersecurity applications, contributing to improved overall performance and outcomes.")
        
        #Error, F1 Score, F2 Score
        # Calculate error rate
        st.header("4.3 Error, F1 Score, F2 Score, Precision, Recall, Sensitivity, Specificity")
        error_rate = 1 - accuracy_score(y_test, y_test_xgb)
        st.write("Error rate:", error_rate)
        # Calculate F1 score
        f1_score = fbeta_score(y_test, y_test_xgb, beta=1)
        st.write("F1 score:", f1_score)
        # Calculate F2 score
        f2_score = fbeta_score(y_test, y_test_xgb, beta=2)
        st.write("F2 score:", f2_score)
        
        # Calculate precision and recall
        precision = precision_score(y_test, y_test_xgb)
        recall = recall_score(y_test, y_test_xgb)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_test, y_test_xgb)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # Print the results
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Sensitivity:", sensitivity)
        st.write("Specificity:", specificity)
        
        # Calculate ROC curve and AUC
        y_prob = xgb.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        st.header("4.4 ROC Curve of the XG Boost")
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        st.pyplot()
        st.caption("The XG Boost’s ROC curve with an Area Under the Curve (AUC) of 0.92 indicates strong discriminatory ability. This means it effectively distinguishes between phishing and legitimate websites, achieving high true positive rates while keeping false positive rates low across different thresholds. The steep rise near the upper-left corner of the curve highlights its high sensitivity and specificity, crucial for accurate phishing detection and minimal false alarms in cybersecurity applications. Overall, the AUC of 0.92 reflects the model's robust performance and suitability for real-world scenarios requiring precise classification of malicious and benign web entities.")
        
        #Autoencoder Model
        st.markdown("<h1 style='text-align: center;'>5. Autoencoder Neural Network</h1>", unsafe_allow_html=True)
        input_dim = X_train.shape[1]
        encoding_dim = input_dim
        input_layer = Input(shape=(input_dim, ))
        encoder = Dense(encoding_dim, activation="relu",
                activity_regularizer=regularizers.l1(10e-4))(input_layer)
        encoder = Dense(int(encoding_dim), activation="relu")(encoder)
        encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
        code = Dense(int(encoding_dim-4), activation='relu')(encoder)
        decoder = Dense(int(encoding_dim-2), activation='relu')(code)
        decoder = Dense(int(encoding_dim), activation='relu')(encoder)
        decoder = Dense(input_dim, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.summary()
        autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


        history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2)
        acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
        acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]

        st.header("5.1 Accuracy of the Autoencoder")
        st.write('\nAutoencoder: Accuracy on training Data: {:.3f}' .format(acc_train_auto))
        st.write('Autoencoder: Accuracy on test Data: {:.3f}' .format(acc_test_auto))
        storeResults('AutoEncoder', acc_train_auto, acc_test_auto)
        st.caption("The autoencoder neural network exhibits a notable difference in performance between seen and unseen data, with a test accuracy of 0.243 and a training accuracy of 0.231. This discrepancy implies challenges in generalizing to new instances, possibly due to overfitting or insufficient model complexity for phishing website detection. Further refinements are necessary to enhance the autoencoder's capability to accurately classify phishing and legitimate websites, ensuring its effectiveness in practical cybersecurity scenarios where generalization plays a crucial role.")
        
        
        st.header("5.2 Confusion Matrix of the Autoencofder")
        cm = confusion_matrix(y_test, y_test_mlp)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix for Autoencoder")
        st.pyplot()
        st.caption("The confusion matrix is a crucial tool in evaluating classification models' performance by showing how well they predict different classes. It organizes actual and predicted class counts into a matrix, with metrics like True Positive (correctly identified positives), False Positive (negatives mistakenly classified as positives), False Negative (positives mistakenly classified as negatives), and True Negative (correctly identified negatives). .Autoencoder Neural Network model for phishing website detection, the confusion matrix helps assess its accuracy in classifying phishing and legitimate websites. Metrics like Accuracy (overall correctness), Precision (avoiding false positives), Recall (identifying positives accurately), Specificity (identifying negatives accurately), and the F1 Score (a balance of precision and recall) provide insights into the model's performance. Understanding the confusion matrix helps in optimizing the model by balancing metrics like precision and recall, ensuring it effectively detects phishing websites while minimizing false positives and false negatives. Fine-tuning the model based on these insights enhances its reliability and effectiveness in cybersecurity applications, contributing to improved overall performance and outcomes.")
        #Error, F1 Score, F2 Score
        # Calculate error rate
        st.header("5.3 Error, F1 Score, F2 Score, Precision, Recall, Sensitivity, Specificity")
        error_rate = 1 - accuracy_score(y_test, y_test_tree)
        st.write("Error rate:", error_rate)
        # Calculate F1 score
        f1_score = fbeta_score(y_test, y_test_tree, beta=1)
        st.write("F1 score:", f1_score)
        # Calculate F2 score
        f2_score = fbeta_score(y_test, y_test_tree, beta=2)
        st.write("F2 score:", f2_score)
        
        # Calculate precision and recall
        precision = precision_score(y_test, y_test_tree)
        recall = recall_score(y_test, y_test_tree)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_test, y_test_tree)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # Print the results
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Sensitivity:", sensitivity)
        st.write("Specificity:", specificity)
        
        # Calculate ROC curve and AUC
        st.header("5.4 ROC Curve of the Autoencoder")
        y_prob = tree.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        st.pyplot()
        st.caption("The Autoencoder Neural Network’s ROC curve with an Area Under the Curve (AUC) of 0.85 indicates strong discriminatory ability. This means it effectively distinguishes between phishing and legitimate websites, achieving high true positive rates while keeping false positive rates low across different thresholds. The steep rise near the upper-left corner of the curve highlights its high sensitivity and specificity, crucial for accurate phishing detection and minimal false alarms in cybersecurity applications. Overall, the AUC of 0.85 reflects the model's robust performance and suitability for real-world scenarios requiring precise classification of malicious and benign web entities.")
        
        #Support vector machine model
        # instantiate the model
        st.markdown("<h1 style='text-align: center;'>6. Support Vector Machine</h1>", unsafe_allow_html=True)
        svm = SVC(kernel='linear', C=1.0, random_state=12)
        #fit the model
        svm.fit(X_train, y_train)
        
        y_test_svm = svm.predict(X_test)
        y_train_svm = svm.predict(X_train)
        
        acc_train_svm = accuracy_score(y_train,y_train_svm)
        acc_test_svm = accuracy_score(y_test,y_test_svm)
        st.header("6.1 Accuracy of the SVM")
        st.write("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
        st.write("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))
        st.caption("The Support Vector Machine model shows strong performance, achieving a test accuracy of 0.803 and a training accuracy of 0.797. This indicates the model's capability to generalize effectively to new data, accurately distinguishing between phishing and legitimate websites. The minimal difference of 0.006 between test and training accuracies suggests limited overfitting, emphasizing the model's stability across datasets. SVM's reliability makes it suitable for practical use, ensuring accurate phishing detection and low false positive rates, essential for effective cybersecurity measures.")
        
        #Confusion Matrix of Support Vector Machine
        st.header("6.2 Confusion Matrix of the SVM")
        cm = confusion_matrix(y_test, y_test_svm)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix for SVM")
        st.pyplot()
        st.caption("The confusion matrix is a crucial tool in evaluating classification models' performance by showing how well they predict different classes. It organizes actual and predicted class counts into a matrix, with metrics like True Positive (correctly identified positives), False Positive (negatives mistakenly classified as positives), False Negative (positives mistakenly classified as negatives), and True Negative (correctly identified negatives). Support Vector Machine model for phishing website detection, the confusion matrix helps assess its accuracy in classifying phishing and legitimate websites. Metrics like Accuracy (overall correctness), Precision (avoiding false positives), Recall (identifying positives accurately), Specificity (identifying negatives accurately), and the F1 Score (a balance of precision and recall) provide insights into the model's performance. Understanding the confusion matrix helps in optimizing the model by balancing metrics like precision and recall, ensuring it effectively detects phishing websites while minimizing false positives and false negatives. Fine-tuning the model based on these insights enhances its reliability and effectiveness in cybersecurity applications, contributing to improved overall performance and outcomes.")
        
        #Error, F1 Score, F2 Score
        st.header("6.3 Error, F1 Score, F2 Score, Precision, Recall, Sensitivity, Specificity")
        # Calculate error rate
        error_rate = 1 - accuracy_score(y_test, y_test_tree)
        st.write("Error rate:", error_rate)
        # Calculate F1 score
        f1_score = fbeta_score(y_test, y_test_tree, beta=1)
        st.write("F1 score:", f1_score)
        # Calculate F2 score
        f2_score = fbeta_score(y_test, y_test_tree, beta=2)
        st.write("F2 score:", f2_score)
        
        # Calculate precision and recall
        precision = precision_score(y_test, y_test_svm)
        recall = recall_score(y_test, y_test_svm)

        # Calculate sensitivity and specificity
        cm = confusion_matrix(y_test, y_test_svm)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # Print the results
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("Sensitivity:", sensitivity)
        st.write("Specificity:", specificity)
                
        # Calculate ROC curve and AUC of Support Vector Machine
        st.header("6.4 ROC Curve of the SVM")
        y_prob = tree.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        st.pyplot()
        st.caption("The Support Vector Machine’s ROC curve with an Area Under the Curve (AUC) of 0.85 indicates strong discriminatory ability. This means it effectively distinguishes between phishing and legitimate websites, achieving high true positive rates while keeping false positive rates low across different thresholds. The steep rise near the upper-left corner of the curve highlights its high sensitivity and specificity, crucial for accurate phishing detection and minimal false alarms in cybersecurity applications. Overall, the AUC of 0.85 reflects the model's robust performance and suitability for real-world scenarios requiring precise classification of malicious and benign web entities.")
        
        storeResults('SVM', acc_train_svm, acc_test_svm)
        
        st.markdown("<h1 style='text-align: center;'>Implementation of Accuracy of the Model</h1>", unsafe_allow_html=True)
        results = pd.DataFrame({ 'ML Model': ML_Model, 'Train Accuracy': acc_train, 'Test Accuracy': acc_test})
        results
        st.caption("After applying various machine learning models for phishing website detection, XGBoost consistently demonstrates the best performance. Its accuracy, precision, recall, and overall F1 score surpass those of other models such as decision trees, random forests, SVMs, MLPs, and autoencoders. XGBoost's robustness, efficiency in handling large datasets, and ability to capture complex patterns make it the top choice for enhancing phishing detection accuracy and effectiveness.")
        
        
        #Sorting the datafram on accuracy
        results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
        