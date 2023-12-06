import langchain
import streamlit as st
from langchain.llms import GooglePalm
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
import re

def main():
    st.set_page_config(page_title="Cuisine Genius",page_icon=":knife_fork_plate:")
    st.title("Cuisine Genius::knife_fork_plate:")
    st.markdown("Restaurant Idea Generator")
    cuisine = st.sidebar.text_input("Enter the cuisine type")

    # Get user's GooglePalm key
    with st.sidebar:
        google_api_key = st.text_input(label = "Google API key", placeholder="Ex sk-2twmA8tfCb8un4...",
        key ="google_api_key_input", help = "How to get a Google api key: Visit https://makersuite.google.com")

        # Container for markdown text

        with st.container():
            st.markdown("""Make sure you have entered your API key.
                        Don't have an API key yet?
                        Read this: Visit https://makersuite.google.com and login with your google account and Get your API key""")
    

    def restaurant_idea_generator():
        
        llm = GooglePalm(google_api_key=google_api_key,temperature=0.6)

        first_prompt_input = PromptTemplate(
            input_variables=['cuisine'],
            template="I want to open a restaurant for {cuisine} food. Suggest only one fancy name for this"
        )

        chain = LLMChain(llm=llm,prompt=first_prompt_input,verbose=True,output_key='name')

        second_prompt_input = PromptTemplate(
            input_variables=['name'],
            template="Suggest menu items for {name} with separate list for starters, main dishes and savouries."
        )

        chain2 = LLMChain(llm=llm,prompt=second_prompt_input,verbose=True,output_key='menu_items')

        chain = SequentialChain(
            chains = [chain,chain2],
            input_variables = ['cuisine'],
            output_variables = ['name','menu_items']
        )

        response = chain({'cuisine':cuisine})
        return response

    if cuisine:
        if not google_api_key:
            st.error("Please enter your Google API key")
        else:
            response = restaurant_idea_generator()
            name = re.search(r'[^restaurant].*',response['name'])
            st.header(name.group())
            menu_items = response['menu_items'].strip().split("\n")
            st.write("**Menu items**")
            for item in menu_items:
                st.write(item)


if __name__=="__main__":
    main()