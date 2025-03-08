# initialize library for YouTube summarizer app
import streamlit as st
import json
import os
import requests
from openai import OpenAI
import random
import time

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

# get answers from OpenAI model
def get_html_content(video_id, num_section):
    print("Number of sections:", num_section)
    
    # Initialize the HTML content
    html_content = "<html><head></head><body>"
    html_content += f"<h1>{st.session_state['video_title']}</h1>"
                    
    for i in range(num_section):
        topic = st.session_state['summary_title'][i]
        timestamp = st.session_state['timestamp'][i]
        summary = st.session_state['section_summary'][i]
        link = f"https://www.youtube.com/watch?v={video_id}&t={timestamp}"

        html_content += f"<h2>{topic}</h2>"
        html_content += f"<p><strong>Timestamp:</strong> <a href='{link}'>{timestamp}</a></p>"
        html_content += f"<p>{summary}</p>"
        html_content += "</body></html>"

    return html_content

# generate detailed summary
def generate_detailed_summary(transcript, video_id, num_sections, languages):
    # calculate the number of sections
    sections = []
    whole_segment = len(transcript['transcript'])
    section_segment = max(whole_segment // num_sections, 1)

    # initialize current section
    current_section = []
    section_start = None

    # loop for format the text with timestamp
    for index, segment in enumerate(transcript['transcript']):
        text = segment['text']
        start_time = segment['start']

        # format the text with timestamp
        formatted_text = f"({format_time(start_time)}) {text}"
        current_section.append(formatted_text)

        # initialize section start time
        if section_start is None:
            section_start = start_time

        # check the section segment
        if (index + 1) % section_segment == 0 or index == whole_segment - 1:
            if len(sections) < num_sections:
                section = {
                    'title': f'Section {len(sections) + 1}',
                    'timestamp': f"{format_time(section_start)}",
                    'session_transcript': current_section
                }
                sections.append(section)
            
            # initialize for next section
            current_section = []  
            section_start = None 

    # Format the sections with hyperlinks
    formatted_sections = []
    base_url = f"https://www.youtube.com/watch?v={video_id}&t="

    # loop for format the sections
    for section in sections:
        url_timestamp = int(section['timestamp'][0:2]) * 3600 + int(section['timestamp'][3:5]) * 60 + int(section['timestamp'][6:8])
        section_link = f"{base_url}{url_timestamp}"
        
        formatted_summary = "\n".join(section['session_transcript'])
        formatted_sections.append(f"Section {section['title']}:\n"
                                  f"Timestamp: {section['timestamp']}\n"
                                  f"Link: {section_link}\n"
                                  f"session_transcript:\n{formatted_summary}\n")
    
    # Initialize for generate detailed summary
    detailed_summary = []
    user_prompt = []
    full_response = []
    
    # loop for generate detailed summary
    for section in formatted_sections:
        if 'Summary:'and 'Timestamp:' and 'Link:' in section:
            session_transcript = section.split("session_transcript:")[1].strip()
            section_timestamp = section.split("Timestamp:")[1].split("Link:")[0].strip()
            section_link = section.split("Link:")[1].split("session_transcript:")[0].strip()
            print(f"Session Transcript: {session_transcript}")
            print(f"Timestamp: {section_timestamp}")
            print(f"Link: {section_link}")
            
        # prompt for generate topic and summary
        system_prompt = "You are a helpful assistant."
        user_prompt_topic = f"Please generate a title only with 5 - 10 words and change the language with the selected language {languages}, based on the session transcript: {session_transcript}"
        topic_response = llm_answers(system_prompt, user_prompt_topic)
        topic_content = topic_response.choices[0].message.content.strip()
        
        user_prompt_summary = f"Please generate a 80 - 100-word summary only and change the language with the selected language {languages}, based on the session transcript: {session_transcript}."
        summary_response = llm_answers(system_prompt, user_prompt_summary)
        summary_content = summary_response.choices[0].message.content.strip()

        # Append to detailed_summary
        detailed_summary.append({
            'topic': topic_content,
            'summary': summary_content,
            'timestamp': section_timestamp,
            'session_transcript': session_transcript,
            'link': section_link
        })

        print("Detailed Summary:", detailed_summary)

        # Store the full response content
        full_response.append(topic_content)
        full_response.append(summary_content)
        
        # store user prompt of topic and summary
        user_prompt.append(user_prompt_topic)
        user_prompt.append(user_prompt_summary)

    return detailed_summary, user_prompt, full_response

# generate more detailed summary
def generate_more_detailed_summary(session_transcript, selected_languages):
    # prompt for generate detailed summary
    system_prompt = "You are a helpful assistant."
    more_detailed_prompt = f"Please generate a summary only in more detail with the given section of the transcript: {session_transcript} and please change the content with the selected language {selected_languages}."    
    
    # get response from OpenAI
    response = llm_answers(system_prompt, more_detailed_prompt)
    detailed_summary = response.choices[0].message.content
    if detailed_summary:
        return detailed_summary
    return None

# generate more concise summary
def generate_concise_summary(session_transcript, selected_languages):
    # prompt for generate concise summary
    system_prompt = "You are a helpful assistant."
    more_concise_prompt = f"Please generate a summary only in more concise with the given section of the transcript: {session_transcript} and please change the content with the selected language {selected_languages}."   
    
    # get response from OpenAI
    response = llm_answers(system_prompt, more_concise_prompt)
    concise_summary = response.choices[0].message.content
    if concise_summary:
        return concise_summary
    return None

# generate more fun summary
def generate_fun_summary(session_transcript, selected_languages): 
    # prompt for generate fun summary   
    system_prompt = "You are a helpful assistant."
    more_fun_prompt = f"Please generate a summary only in more fun and engaging with the given section of the transcript: {session_transcript} and please change the content with the selected language {selected_languages}."    
    
    # get response from OpenAI
    response = llm_answers(system_prompt, more_fun_prompt)
    fun_summary = response.choices[0].message.content
    if fun_summary:
        return fun_summary
    return None

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

# initialize for generate detailed summary
if "detailed_summaries" not in st.session_state:
    st.session_state["detailed_summaries"] = []
if "summary_title" not in st.session_state:
    st.session_state["summary_title"] = "No summary title found"
if "section_summary" not in st.session_state:
    st.session_state["section_summary"] = "No section summary found"
if "timestamp" not in st.session_state:
    st.session_state["timestamp"] = "No timestamp found"
if "link" not in st.session_state:
    st.session_state["link"] = "No link found"
if "session_transcript" not in st.session_state:
    st.session_state["session_transcript"] = "No session transcript found"
if "num_section" not in st.session_state:
    st.session_state["num_section"] = random.randint(4, 6)

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
            if video_url:
                video_id = get_video_id(video_url)
                if video_id:
                    st.session_state["video_url"] = video_url
                    st.session_state["video_title"] = get_video_title(video_id)
                    transcript = get_transcript(video_id, languages)

                    if transcript is None:
                        st.error("Failed to fetch transcript.")
                    else:
                        detailed_summaries, user_prompt, response = generate_detailed_summary(transcript, video_id, st.session_state["num_section"], st.session_state["selected_language"])
                        st.session_state["response"] = response
                        st.session_state["user_prompt"] = user_prompt
                        st.session_state["detailed_summaries"] = detailed_summaries  # Store detailed summaries
                        if detailed_summaries:
                            st.session_state["summary_title"] = [item['topic'] for item in detailed_summaries]
                            st.session_state["timestamp"] = [item['timestamp'] for item in detailed_summaries]
                            st.session_state["section_summary"] = [item['summary'] for item in detailed_summaries]
                            st.session_state["link"] = [item['link'] for item in detailed_summaries]
                            st.session_state["session_transcript"] = [item['session_transcript'].strip("\n") for item in detailed_summaries]
                else:
                    st.error("Please provide a valid YouTube URL")
            else:
                st.error("Please provide a valid YouTube URL.")

    # Display the user prompt
    with st.expander("Show Prompt", expanded=False):
        if st.session_state['detailed_summaries']:
            for i in range(st.session_state["num_section"]):
                st.text(st.session_state["user_prompt"][i])
        else:
            st.text(st.session_state["user_prompt"])
    
    # Display the LLM output
    with st.expander("Show LLM Output", expanded=False):
        if st.session_state['detailed_summaries']:
            for i in range(len(st.session_state["response"])):
                st.write(st.session_state["response"][i])
        else:
            st.write(st.session_state["response"])

    # Display the download button for HTML content
    if st.session_state["detailed_summaries"] and "video_url" in st.session_state:
        video_id = get_video_id(video_url)
        html_content = get_html_content(video_id, st.session_state["num_section"])
        st.download_button(label="Download Summary as HTML", data=html_content, file_name="summary.html", mime="text/html")

    if "video_title" in st.session_state:
        st.subheader(st.session_state["video_title"])

    if "video_url" in st.session_state:
        st.write("Video URL:", st.session_state["video_url"])

    if generate_summary_button and "summary" in st.session_state:
        st.write(st.session_state["summary"])

    # Display the detailed summaries
    if st.session_state["detailed_summaries"] and st.session_state["selected_language"]:
        print("Number of sections:", st.session_state["num_section"])
        
        for i in range(st.session_state["num_section"]):
            st.subheader(f"Section {i+1}: {st.session_state['summary_title'][i]}")
            st.write(f"Timestamp: [{st.session_state['timestamp'][i]}]({st.session_state['link'][i]})")
            with st.expander("Show Transcript", expanded=False):
                if st.session_state["session_transcript"]:
                    formatted_transcript = st.session_state["session_transcript"][i].split("\n")
                    for item in formatted_transcript:
                        st.text(item)         
            edited_summary = st.text_area("Summary: ", value=st.session_state["section_summary"][i], height=200, key=f'summary_textarea_{i}')
            
            # Display the buttons for additional functionality
            col1, col2, col3, col4 = st.columns(4)
            with col1: # Save Button
                if st.button(f"Save", key=f"save_button_{i}"):
                    if edited_summary:
                        st.session_state["section_summary"][i] = edited_summary
                        st.success(f"Section {i+1} summary is saved!")
                    else:
                        st.error("Failed to save the summary.")

            with col2: # More Details Button
                if st.button(f"More Details", key=f"detailed_button_{i}"):
                    detailed_summary = generate_more_detailed_summary(st.session_state["session_transcript"][i], st.session_state["selected_language"])
                    if detailed_summary:
                        st.session_state["section_summary"][i] = detailed_summary  # Update the section summary
                        st.success(f"Section {i+1} is regenerated with more details!")
                        time.sleep(5)
                        st.rerun()
                    else:
                        st.error("Failed to generate a detailed summary.")

            with col3: # More Concise Button
                if st.button(f"More Concise", key=f"concise_button_{i}"):
                    concise_summary = generate_concise_summary(st.session_state["session_transcript"][i], st.session_state["selected_language"])
                    if concise_summary:
                        st.session_state["section_summary"][i] = concise_summary  # Update the section summary
                        st.success(f"Section {i+1} is regenerated with more concise!")
                        time.sleep(5)
                        st.rerun()
                        
                    else:
                        st.error("Failed to generate a concise summary.")

            with col4: # More Fun Button
                if st.button(f"More Fun", key=f"fun_button_{i}"):
                    fun_summary = generate_fun_summary(st.session_state["session_transcript"][i], st.session_state["selected_language"])
                    if fun_summary:
                        st.session_state["section_summary"][i] = fun_summary
                        st.success(f"Section {i+1} is regenerated with more fun!")
                        time.sleep(5)
                        st.rerun()
                    
                    else:
                        st.error("Failed to generate a fun summary.")