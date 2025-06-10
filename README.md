# My Project

This project requires Python 3.8.20 and uses several common deep learning and image processing libraries. This README will guide you through setting up your development environment on Windows, including installing Python and Conda (via Anaconda), creating a virtual environment, and installing the necessary dependencies.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step-by-Step Setup](#step-by-step-setup)
  - [1. Install Anaconda (Recommended)](#1-install-anaconda-recommended)
  - [2. Open Anaconda Prompt](#2-open-anaconda-prompt)
  - [3. Create a New Conda Environment](#3-create-a-new-conda-environment)
  - [4. Activate Your Project Environment](#4-activate-your-project-environment)
  - [5. Install Project Dependencies](#5-install-project-dependencies)
- [VS Code Extensions for Notebooks](#vs-code-extensions-for-notebooks)
- [Running the Project](#running-the-project)

## Prerequisites

* **Windows Operating System:** This guide is specifically for Windows users.

## Step-by-Step Setup

Follow these steps chronologically to set up your project environment.

### 1. Install Anaconda (Recommended)

Anaconda is a comprehensive distribution that includes Conda, Python, and many commonly used packages for data science and machine learning.

1.  **Download Anaconda:**
    * Go to the official Anaconda Individual Edition website: [https://www.anaconda.com/download](https://www.anaconda.com/download)
    * Scroll down and download the **Windows 64-bit Graphical Installer** (Python 3.x version). It will be an `.exe` file and is quite large.

2.  **Run the Installer:**
    * Find the downloaded `.exe` file (usually in your "Downloads" folder) and double-click it to start the installation.
    * Click "Next" to proceed through the installation wizard.
    * Agree to the license terms.
    * Choose "Just Me" for the installation type (recommended unless you're managing users on a shared computer).
    * Accept the default installation location (e.g., `C:\Users\YourUser\anaconda3`).
    * **IMPORTANT STEP:** On the "Advanced Installation Options" screen, make sure to **check the box that says "Add Anaconda3 to my PATH environment variable"**. This is crucial as it allows you to use `conda` commands from any command prompt. If you miss this, don't worry, you can still use the "Anaconda Prompt" specifically.
    * Click "Install" and wait for the installation to complete. This might take some time due to the large number of pre-installed packages.
    * Once finished, you can uncheck the boxes for "Learn more about Anaconda Navigator" and "Learn more about Anaconda distribution" and click "Finish".

### 2. Open Anaconda Prompt

After installing Anaconda, you'll have a special command prompt called "Anaconda Prompt" which is set up to work with Conda.

* **How to open:** Press the **Windows key** on your keyboard, then type `Anaconda Prompt`. You should see an option like "Anaconda Prompt (anaconda3)". Click on it to open.
* You will see a command window open, likely starting with `(base)` on the left side of the prompt. This `(base)` indicates that you are currently in the default Conda environment.

### 3. Create a New Conda Environment

It's a good practice to create a separate, isolated environment for each project. This prevents different projects from interfering with each other's software requirements. We will create an environment specifically for this project, using Python version 3.8.20.

In the Anaconda Prompt, copy and paste this command and press **Enter**:

```bash
conda create -n myproject_env python=3.8.20
