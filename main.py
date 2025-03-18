import os
import re
import json
import fitz
import ast
import logging
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.tools import Tool
from langchain_core.tools import tool
from langchain import hub
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError



# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# paths
PDF_PATH = "resume.pdf"
JOB_DESCRIPTION_PDF_PATH = "JD.pdf"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Hugging Face API token is missing.")

if not YOUTUBE_API_KEY:
    raise ValueError("YouTube API key is missing. Please set YOUTUBE_API_KEY in your .env file.")


# Initialize Llama 3.2 model from Hugging Face Hub
llama_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

#initialize youtube api
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def extract_json_from_text(text: str) -> Dict:
    """Extract and parse JSON from a text string safely."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # Extract JSON block
        if match:
            json_string = match.group(0)  
            return json.loads(json_string)  # Parse JSON
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
    
    logger.error("No valid JSON found in the text.")
    return {}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    # Check if file exists before processing 
    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return "" 


def combine_text(resume_text: str, job_description_text: str) -> str:
    """Combine resume and job description text into a single formatted string."""
    return f"Resume: {resume_text}\n\nJob Description: {job_description_text}"


def extract_list_from_text(text: str) -> list | None:
    """Extract a Python list from text using regex and safe parsing."""
    match = re.search(r"\[.*?\]", text)
    if match:
        try:
            extracted_list = ast.literal_eval(match.group(0))
            return extracted_list if isinstance(extracted_list, list) else None
        except (SyntaxError, ValueError):
            logging.warning("Failed to parse list from text.")
    return None

@tool
def analyze_resume(combined_text: str) -> List[Dict[str, str]]:
    """
    Analyze a resume against a job description and return 3 technical improvement areas.
    """
    prompt = f"""
Analyze the following resume and job description to identify **EXACTLY 3 key technical improvement areas**.  
Focus strictly on **technical skills**, not managerial or soft skills.  

### Input:  
{combined_text}  

### Output Format (STRICTLY FOLLOW THIS):  
Return **only one valid JSON object** without any extra text, explanations, or multiple outputs.  
Ensure the JSON structure exactly matches the format below:

{{
    "Improvement Areas": [
        {{
            "Title": "Improvement Area 1",
            "Description": "Brief description of the first improvement area."
        }},
        {{
            "Title": "Improvement Area 2",
            "Description": "Brief description of the second improvement area."
        }},
        {{
            "Title": "Improvement Area 3",
            "Description": "Brief description of the third improvement area."
        }}
    ]
}}
"""

    try:
        analysis = llama_llm.invoke(prompt) 
        improvement_areas = extract_json_from_text(analysis).get("Improvement Areas", [])
        
        if not improvement_areas:
            logger.warning("No improvement areas found in the analysis.")
            return []

        return improvement_areas
    
    except Exception as e:
        logger.error(f"Error in resume analysis: {e}")
        return [{"Title": "Error", "Description": "Failed to analyze resume."}]



@tool
def search_youtube_videos(queries: List[str]) -> Dict[str, List[str]]:
    """
    Search YouTube for videos related to the given queries and return links to relevant videos.
    """
    recommended_videos = {}

    for query in queries:
        try:
            request = youtube.search().list(
                part="snippet", q=query, maxResults=10 # get top 7 results may contain video or playlist
            )
            response = request.execute()

            video_urls = [
                f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                for item in response.get("items", [])
                if "videoId" in item["id"]
            ]

            if not video_urls:
                logging.warning(f"No video results for query: {query}")
                continue  # Skip retrying the exact same query

            # Get top 3 videos out of the 10 results
            recommended_videos[query] = video_urls[:3]

        except HttpError as e:
            logging.error(f"YouTube API error for query '{query}': {e}")
            recommended_videos[query] = ["YouTube API error occurred"]

        except Exception as e:
            logging.error(f"Unexpected error for query '{query}': {e}")
            recommended_videos[query] = ["An error occurred"]

    return recommended_videos


def create_tools(combined_text: str) -> list:
    """Create Langchain tools for resume analysis and YouTube search."""
    return [
        Tool(
            name="analyze_resume",
            func=lambda _: analyze_resume(combined_text),
            description="Analyze a resume against a job description and return improvement areas.",
        ),
        Tool(
            name="search_youtube_videos",
            func=lambda queries: search_youtube_videos({"queries": extract_list_from_text(queries)}),
            description="Search YouTube for videos related to the improvement areas.",
        ),
    ]


def main():
    """Main function to run the AI agent."""
    try:
        resume_text = extract_text_from_pdf(PDF_PATH)
        job_description_text = extract_text_from_pdf(JOB_DESCRIPTION_PDF_PATH)
        combined_text = combine_text(resume_text, job_description_text)

        tools = create_tools(combined_text)

        #load react template from langchain hub
        prompt_template = hub.pull("hwchase17/react")

        react_agent = create_react_agent(llm=llama_llm, tools=tools, prompt=prompt_template)
        agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=False, return_intermediate_steps=True)

        query = (
            "You are an AI assistant that evaluates resumes based on job descriptions. "
            "Use the analyze_resume tool to analyze the combined resume and job description text, "
            "and search YouTube for relevant resources using the search_youtube_videos tool. "
            "Provide short descriptions of the improvement areas as the final answer."
        )

        response = agent_executor.invoke({"input": query})
        intermediate_steps = response["intermediate_steps"]

        improvement_areas = intermediate_steps[0][1]
        youtube_links = intermediate_steps[1][1]

        print("\nIMPROVEMENT AREAS:\n", improvement_areas)
        print("\nYOUTUBE LINKS:\n", youtube_links)

    except Exception as e:
        logging.error("An error occurred: %s", str(e))


if __name__ == "__main__":
    main()
