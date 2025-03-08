# initialize library for YouTube summarizer app
import streamlit as st
import json
import os
import requests
from openai import OpenAI

# get GitHub API key, endpoint, model name
github_key = os.getenv('GITHUB_API_KEY')
github_endpoint = os.getenv('GITHUB_API_ENDPOINT')
github_model = os.getenv('GITHUB_API_MODEL_NAME')

# read the GitHub API key, endpoint, model name
print("GitHub API and Information:")
print("GitHub API Key:", github_key)
print("GitHub API Endpoint:", github_endpoint)
print("GitHub API Model Name:", github_model)

# Get video ID from the video URL
def get_video_id(video_url):
    if "v=" in video_url and "youtube.com" in video_url:
        video_id = video_url.split("v=")[-1]
        return video_id
    return None

# Get the video title
def get_video_title(video_id):
    video_url = f"https://yt.vl.comp.polyu.edu.hk/transcript?password=for_demo&video_id={video_id}"
    response = requests.get(video_url)
    
    # get content of response
    if response.status_code == 200:
        video_info = response.json()
        return video_info.get("video_title", "No Video Title")
    return "No Video Title"

# Get the transcript
def get_transcript(video_id, language_code):
    # get language code
    lang_url = f"https://yt.vl.comp.polyu.edu.hk/lang?password=for_demo&language=en&video_id={video_id}"
    lang_response = requests.get(lang_url)
    
    if lang_response.status_code == 200:
        language = lang_response.json()
        print(f"Available language code in video: {language}")
        
        # Extract available language codes
        available_languages = [lang['language_code'] for lang in language if 'language_code' in lang]
        if available_languages == []:
            language_code = "en"
        else:
            language_code = available_languages[0]
            print(f"Using language code: {language_code}")

    # get transcript
    transcript_url = f"https://yt.vl.comp.polyu.edu.hk/transcript?language_code={language_code}&password=for_demo&video_id={video_id}"
    transcript_response = requests.get(transcript_url)
    
    if transcript_response.status_code == 200:
        transcript = transcript_response.json()
        return transcript
    return None

# format time to (hh:mm:ss)
def format_time(s):
    hrs = int(s // 3600)
    min = int((s % 3600) // 60)
    s = int(s % 60)
    return f"{hrs:02}:{min:02}:{s:02}"

# format transcript to (hh:mm:ss) text
def format_transcript(transcript):
    # initialize formatted transcript
    formatted_transcript = []
    for item in transcript:
        if "start" in item and "text" in item:
            start_time = float(item['start'])
            formatted_time = format_time(start_time)
            text = item['text']
            formatted_transcript.append(f"({formatted_time}) {text}")
    return formatted_transcript

# get answers from OpenAI
def llm_answers(system_prompt, user_prompt):
    # create an instance of OpenAI
    client = OpenAI(
        base_url=github_endpoint,
        api_key=github_key,
    )
    
    # check whether correct model name
    if github_model == "gpt-4o-mini":
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            temperature=0.5,
            max_tokens=2000,
            model=github_model
        )
    
    else:
        st.error("Invalid model name. Please check the model name.")
        return None
    
    return response

# initialize both approaches
if "selected_language" not in st.session_state:
    st.session_state["selected_language"] = "en"
if "video_title" not in st.session_state:
    st.session_state["video_title"] = "No video title found"
if "video_url" not in st.session_state:
    st.session_state["video_url"] = "No video id found"
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "No system prompt found"
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = "No user prompt found"
if "response" not in st.session_state:
    st.session_state["response"] = "No response found"
    
# initialize for generate summary
if "transcript" not in st.session_state:
    st.session_state["transcript"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = "No summary found"

# Streamlit layout
if __name__ == "__main__":
    st.set_page_config(page_title="YouTube Summarizer App", layout="wide")
    with st.sidebar:
        video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=SLwpqD8n3d0")
        
        languages = st.selectbox("Language", options=["en", "zh-TW", "zh-CN"], index=["en", "zh-TW", "zh-CN"].index(st.session_state["selected_language"]))
        st.session_state["selected_language"] = languages # save the selected language to session state
        print("Selected language:", st.session_state["selected_language"])
        
        generate_summary_button = st.button("Generate Summary")
        generate_detailed_summary_button = st.button("Generate Detailed Summary")

        # generate content of summary
        if generate_summary_button:
            if video_url:
                video_id = get_video_id(video_url)
                if video_id:
                    st.session_state["video_url"] = video_url
                    st.session_state["video_title"] = get_video_title(video_id)
                    transcript = get_transcript(video_id, st.session_state["selected_language"])
                
                    if transcript is None:
                        st.error("Failed to fetch transcript.")
                    else:
                        formatted_transcript = format_transcript(transcript['transcript'])
                        print("Formatted Transcript:", formatted_transcript)
                        st.session_state["transcript"] = "\n".join(formatted_transcript)
                        print("Transcript:", st.session_state["transcript"])
            
                
                        st.session_state["system_prompt"] = "You are a helpful assistant."
                        st.session_state["user_prompt"] = (
                            f"Please summarize the transcript from YouTube video with the selected language{st.session_state['selected_language']}. "
                            f"Transcript:\n{st.session_state['transcript']}"
                        )
                        print("User Prompt:", st.session_state["user_prompt"])
                        
                        response = llm_answers(st.session_state["system_prompt"], st.session_state["user_prompt"])
                        st.session_state["response"] = response
                        st.session_state["summary"] = response.choices[0].message.content
                else:
                    st.error("Please provide a valid YouTube URL")
            else:
                st.error("Please provide a valid YouTube URL.")
        
        # generate content of detailed summary
        if generate_detailed_summary_button:
            st.error("Detailed Summary is still under development.")

    # Display the user prompt
    with st.expander("Show Prompt", expanded=False):
        st.text(st.session_state["user_prompt"])
    
    # Display the LLM output
    with st.expander("Show LLM Output", expanded=False):
        st.write(st.session_state["response"])

    if "video_title" in st.session_state:
        st.subheader(st.session_state["video_title"])

    if "video_url" in st.session_state:
        st.write("Video URL:", st.session_state["video_url"])

    if "summary" in st.session_state:
        st.write(st.session_state["summary"])
