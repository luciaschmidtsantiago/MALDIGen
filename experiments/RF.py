
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.data import load_data

LABEL_NAMES = [
	'Enterobacter_cloacae_complex',
	'Enterococcus_Faecium',
	'Escherichia_Coli',
	'Klebsiella_Pneumoniae',
	'Pseudomonas_Aeruginosa',
	'Staphylococcus_Aureus'
]
LABEL_TO_COLOR = {lbl: plt.cm.Spectral(i / 5) for i, lbl in enumerate(LABEL_NAMES)}

def main():
	# Load pickles (same as generative models)
	pickle_marisma = "pickles/MARISMa_study.pkl"
	pickle_driams = "pickles/DRIAMS_study.pkl"
	train, val, test, ood = load_data(pickle_marisma, pickle_driams, logger=None, get_labels=True)

	# Prepare data
	X_train = train.data.cpu().numpy() if hasattr(train.data, 'cpu') else np.array(train.data)
	y_train = train.labels.cpu().numpy() if hasattr(train.labels, 'cpu') else np.array(train.labels)
	X_test = test.data.cpu().numpy() if hasattr(test.data, 'cpu') else np.array(test.data)
	y_test = test.labels.cpu().numpy() if hasattr(test.labels, 'cpu') else np.array(test.labels)
	X_ood = ood.data.cpu().numpy() if hasattr(ood.data, 'cpu') else np.array(ood.data)
	y_ood = ood.labels.cpu().numpy() if hasattr(ood.labels, 'cpu') else np.array(ood.labels)

	# Train Random Forest
	print("Training Random Forest...")
	rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
	rf.fit(X_train, y_train)

	# Evaluate on test set
	print("\nEvaluating on test set:")
	y_pred_test = rf.predict(X_test)
	print(classification_report(y_test, y_pred_test))
	print("Confusion matrix (test):\n", confusion_matrix(y_test, y_pred_test))
	print("Accuracy (test):", accuracy_score(y_test, y_pred_test))

	# Evaluate on OOD set
	print("\nEvaluating on OOD set:")
	y_pred_ood = rf.predict(X_ood)
	print(classification_report(y_ood, y_pred_ood))
	print("Confusion matrix (OOD):\n", confusion_matrix(y_ood, y_pred_ood))
	print("Accuracy (OOD):", accuracy_score(y_ood, y_pred_ood))

	# Optionally save results
	results_dir = "results/rf"
	os.makedirs(results_dir, exist_ok=True)
	cm_test = confusion_matrix(y_test, y_pred_test)
	cm_ood = confusion_matrix(y_ood, y_pred_ood)
	pd.DataFrame(cm_test).to_csv(os.path.join(results_dir, "confusion_matrix_test.csv"))
	pd.DataFrame(cm_ood).to_csv(os.path.join(results_dir, "confusion_matrix_ood.csv"))
	print(f"Confusion matrices saved to {results_dir}/")

	# Use train.label_convergence dict for label mapping
	label_correspondence = train.label_convergence
	label_ids = sorted(label_correspondence.keys())
	label_names = [label_correspondence[i] for i in label_ids]

	# Plot confusion matrices with seaborn and experiment label names
	plt.figure(figsize=(8, 6))
	ax1 = sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues")
	ax1.set_title("Confusion Matrix (Test)")
	ax1.set_xlabel("Predicted")
	ax1.set_ylabel("True")
	ax1.set_xticks(np.arange(len(label_names)) + 0.5)
	ax1.set_yticks(np.arange(len(label_names)) + 0.5)
	ax1.set_xticklabels(label_names, rotation=45, ha='right', fontsize=10)
	ax1.set_yticklabels(label_names, rotation=0, fontsize=10)
	plt.tight_layout()
	plt.savefig(os.path.join(results_dir, "confusion_matrix_test.png"))
	plt.close()

	plt.figure(figsize=(8, 6))
	ax2 = sns.heatmap(cm_ood, annot=True, fmt="d", cmap="Blues")
	ax2.set_title("Confusion Matrix (OOD)")
	ax2.set_xlabel("Predicted")
	ax2.set_ylabel("True")
	ax2.set_xticks(np.arange(len(label_names)) + 0.5)
	ax2.set_yticks(np.arange(len(label_names)) + 0.5)
	ax2.set_xticklabels(label_names, rotation=45, ha='right', fontsize=10)
	ax2.set_yticklabels(label_names, rotation=0, fontsize=10)
	plt.tight_layout()
	plt.savefig(os.path.join(results_dir, "confusion_matrix_ood.png"))
	plt.close()
	print(f"Confusion matrix plots saved to {results_dir}/")

if __name__ == "__main__":
	main()
