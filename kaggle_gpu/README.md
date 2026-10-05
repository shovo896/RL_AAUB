# Kaggle GPU from VS Code

This folder is the remote execution copy of the project. VS Code is used to edit the notebook and submit it; the actual GPU run occurs in a Kaggle Notebook.

## First-time connection

1. In VS Code, open **Terminal → Run Task** and choose **Kaggle: Sign in**. Complete the browser sign-in; it uses Kaggle's OAuth flow and does not put a token in this repository.
2. Choose **Kaggle: Configure Notebook ID** and enter `your-kaggle-username/rl-aaub-gpu`. This name is the URL suffix of the private Kaggle Notebook that will be created on the first submission.
3. Run **Kaggle: Show GPU Quota** to see the account's remaining accelerator time.

## Edit, execute, retrieve

1. Open `rl_aaub_gpu.ipynb` in VS Code and edit/save it. To run an existing project notebook, copy the required saved cells into this file; Kaggle uploads only files inside `kaggle_gpu`.
2. Run **Kaggle: Submit GPU Run**. It uploads the notebook and starts a remote `NvidiaTeslaT4` GPU run.
3. Run **Kaggle: Check Run Status**, or **Kaggle: Stream Remote Logs** while it is running.
4. Save models, CSVs, figures, and other artifacts under `/kaggle/working` in the notebook. Run **Kaggle: Download Latest Output** to fetch them to `kaggle_gpu/output/`.

Kaggle datasets listed in `kernel-metadata.json` are mounted at `/kaggle/input`. Add a dataset slug to `dataset_sources`, for example `"owner/dataset-slug"`, before submitting. Keep `enable_internet` off unless the remote notebook really needs network access.

## Important difference from Colab

Running a cell with VS Code's normal Jupyter controls uses the local `.venv`, not the Kaggle GPU. The supported remote workflow is save → submit → inspect logs/output. Each submit creates a Kaggle notebook version and consumes Kaggle account quota.
