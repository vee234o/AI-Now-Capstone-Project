# Project Title: Insurance Claim Prediction System

## Project Overview

This project provides an automated tool for insurance companies to assess the risk of a claim filing based on building characteristics. It uses a machine learning model to categorize insurance policies into high or low risk based on building dimensions, occupancy, and location.

## Key Features

The system includes a data processing pipeline that handles missing values and categorical encoding. It features a predictive model trained on historical data and a web application built with Streamlit for real time risk assessment and probability scoring.

## Repository Structure

The data directory contains the training datasets and variable descriptions. The models directory stores the serialized machine learning model and scaling parameters. The notebooks directory includes the initial exploratory data analysis and model training steps. The scripts directory contains the application code.

## Installation and Setup

To set up the environment install the required libraries listed in the requirements file.
python m pip install r requirements.txt

## Running the Application

To launch the risk assessment interface navigate to the project directory and execute the following command.
streamlit run scripts/app.py

## Usage Instructions

Input the building details into the form including year of observation building dimensions and settlement type. Click the predict button to receive a risk classification and the calculated probability of a claim.