# Candidate Evaluation System (CES) Portal - AI Repository

## Purpose and Objectives
The **Candidate Evaluation System (CES) portal** aims to streamline the process of **feedback and improvement**.  

### Short-Term Goals:
- Provide a platform where a candidate can dynamically seek feedback.  
- Allow candidates to upload their **resume** and compare it with a **domain-specific job description** to get scopes of improvement and online resources for the same.  
- Enable contributions via **Pull Requests (PRs)**.  

### How to Contribute:
- Pick issues as highlighted in the repository.
- Contribute solutions and submit a **Pull Request (PR)**.

## Setting Up The Repository
To clone and set up the repository, run the following commands in your terminal:

```sh
git clone https://github.com/Training-Committee-NIT-Rourkela/CES-AI.git
pip install -r requirements.txt
```

Initialize the following keys to test and run the code locally. Please note that we're using Llama 3.2 as the LLM and the YouTube API exclusively. **DO NOT** use any other LLM/SLM for the application:
- HuggingFace Token, and
- YouTube API key

Your .env file should look somelthing like this:

```sh
YOUTUBE_API_KEY="your_youtubeapi_key"
HUGGINGFACEHUB_API_TOKEN="your_hf_token"
```
