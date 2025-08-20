# Feature Generation Script

This document provides instructions on how to set up the environment and run the `generate_features.sh` script to process your data.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 2.7** or a more recent version.

You can check your Python version by running:

```bash
python --version
```

## Setup and Installation

It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other projects.

### Create a Virtual Environment (Optional but Recommended)

Navigate to your project's root directory and run the following command to create a virtual environment named `venv`:

```bash
python -m venv venv
```

**Note:** Depending on your system, you might need to use `python3` instead of `python`.

### Activate the Virtual Environment

**On macOS and Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
.\venv\Scripts\activate
```

Your command prompt should now be prefixed with `(venv)`, indicating that the virtual environment is active.

### Clone the repository

```bash
git clone git@github.com:IMScience-PPGINF-PucMinas/SkimCap-features.git
```

### Install Dependencies

With the virtual environment active, install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Run the Script

The main script for generating features is `generate_features.sh`. It accepts four command-line arguments.

### Usage

First make it executable:

```bash
chmod +x generate_features.sh
```

The script is executed as follows:

```bash
bash generate_features.sh [features_path] [summary_path] [gen_summary_method] [hierarchy]
```

### Arguments

- **`features_path`**: (String) The file path where the output features will be saved.
- **`summary_path`**: (String) The file path to the input summary file.
- **`gen_summary_method`**: (String) The method to be used for generating the summary. Options: group_sparse_central_features and group_central_frames 
- **`hierarchy`**: (String) The path to the hierarchy file. Options: watershed_hierarchy_by_attribute, watershed_hierarchy_by_minima_ordering, watershed_hierarchy_by_volume, watershed_hierarchy_by_area, watershed_hierarchy_by_dynamics, watershed_hierarchy_by_number_of_parents

### Example

Here is an example of how to run the script with all the required arguments:

```bash
bash generate_features.sh \
    path/to/recurrent-transformer/video_feature/rt_anet_feat/trainval \
    path/to/skim-8-vgg \
    group_sparse_central_features \
    watershed_hierarchy_by_area
```

This command will:

- Use the summary located at `path/to/recurrent-transformer/video_feature/rt_anet_feat/trainval`
- Use the hierarchy definition watershed_hierarchy_by_area
- Apply the `group_sparse_central_features` method for summary generation
