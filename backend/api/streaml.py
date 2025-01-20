import streamlit as st
import asyncio
from ai import analyze_prompt_and_route_task, check_credentials, send_email

st.set_page_config(page_title="AI Task Assistant", layout="wide")

# Add custom CSS for better styling
st.markdown("""
    <style>
    .task-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 OpenCodeHq.Agent")
st.markdown("---")

# Sidebar with task type examples
with st.sidebar:
    st.header(" 🇳🇬 Example Prompts")
    
    st.subheader("Task Creation")
    st.markdown("""
        **Email Tasks:**
        - Send an email to kejimone@gmail.com about the project meeting tomorrow
        
        **Reminders:**
        - Remind me to prepare presentation slides by next Tuesday
        
        **Todo Lists:**
        - Create a todo list for website redesign project
        """
    )
    
    st.subheader("Real-Time Information")
    st.markdown("""
    - What's the weather like in London?
    - Show me the latest news about AI
    - What are the current stock prices for Apple?
    """)
    
    st.subheader("Web Search & Research")
    st.markdown("""
    - Who won the recent US election?
    - What are the best practices for machine learning?
    - Explain quantum computing
    """)

# Add this near the top of your app
if st.sidebar.button("Check Gmail Setup"):
    with st.sidebar:
        check_result = check_credentials()
        if check_result["status"] == "success":
            st.success(f"✅ Gmail configured for: {check_result['email']}")
        else:
            st.error(f"❌ Gmail setup error: {check_result['message']}")

# Initialize session state for email details
if 'email_details' not in st.session_state:
    st.session_state.email_details = {
        'to': '',
        'subject': '',
        'body': '',
        'sender_name': 'Your name here'
    }

# Main interface
user_input = st.text_area(
    "What would you like me to help you with?",
    placeholder="e.g., 'whats the weather in london'",
    height=100
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("🚀 Process Request", use_container_width=True)

if submit_button:
    if user_input:
        with st.spinner("🔄 Processing your request..."):
            try:
                response = asyncio.run(analyze_prompt_and_route_task(user_input))
                
                if response["status"] == "success":
                    st.success("✅ Request processed successfully!")
                    
                    if response.get("request_type") == "TASK_CREATION":
                        with st.expander("🎯 Task Interpretation", expanded=True):
                            interpreted = response['interpreted_task']
                            
                            if interpreted['task_type'].upper() == 'EMAIL':
                                st.markdown("### 📧 Email Preview")
                                
                                # Get email details from the response
                                email_content = response['task_result']['task']['email_content']
                                
                                # Update session state with new email content
                                if 'email_content' in response['task_result']['task']:
                                    st.session_state.email_details.update({
                                        'to': email_content['to'],
                                        'subject': email_content['subject'],
                                        'body': email_content['generated_content']
                                    })
                                
                                # Replace the email editing section with this:
                                with st.form(key='email_form'):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        email_to = st.text_input(
                                            "To:", 
                                            value=st.session_state.email_details['to']
                                        )
                                        email_subject = st.text_input(
                                            "Subject:", 
                                            value=st.session_state.email_details['subject']
                                        )
                                    
                                    # Add sender name field
                                    sender_name = st.text_input(
                                        "Your Name:", 
                                        value=st.session_state.email_details['sender_name']
                                    )
                                    
                                    # Editable email body
                                    email_body = st.text_area(
                                        "Email Body:", 
                                        value=st.session_state.email_details['body'],
                                        height=300
                                    )

                                    # Form submit button
                                    submit_form = st.form_submit_button("📤 Send Email", type="primary")

                                    if submit_form:
                                        # Update session state
                                        st.session_state.email_details.update({
                                            'to': email_to,
                                            'subject': email_subject,
                                            'sender_name': sender_name,
                                            'body': email_body
                                        })

                                        # Add signature
                                        if sender_name and sender_name != "Your name here":
                                            email_body = email_body.rstrip() + f"\n\nBest regards,\n{sender_name}"

                                        # Send email logic
                                        with st.spinner("Sending email..."):
                                            try:
                                                status_container = st.empty()
                                                status_container.info("Preparing to send email...")
                                                
                                                email_result = asyncio.run(
                                                    send_email(
                                                        to=email_to,
                                                        subject=email_subject,
                                                        body=email_body
                                                    )
                                                )
                                                
                                                if email_result["status"] == "success":
                                                    status_container.success(f"""
                                                    ✉️ Email sent successfully!
                                                    - To: {email_result['details']['to']}
                                                    - Subject: {email_result['details']['subject']}
                                                    - Message ID: {email_result['message_id']}
                                                    """)
                                                else:
                                                    status_container.error(f"""
                                                    ❌ Failed to send email:
                                                    Error: {email_result['message']}
                                                    """)
                                            except Exception as e:
                                                st.error(f"❌ Error sending email: {str(e)}")
                            
                            else:
                                # Handle other task types as before
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Task Type:** {interpreted['task_type'].upper()}")
                                    st.markdown(f"**Priority:** {interpreted['priority'].upper()}")
                                with col2:
                                    if 'due_date' in interpreted:
                                        st.markdown(f"**Due Date:** {interpreted['due_date']}")
                    
                    elif response.get("request_type") == "REAL_TIME_INFO":
                        # Handle real-time information response
                        with st.expander("🌐 Real-Time Information", expanded=True):
                            st.markdown("### Search Results")
                            st.markdown(response['search_result'])
                    
                    else:  # WEB_SEARCH
                        # Handle web search response
                        with st.expander("🔍 Research Results", expanded=True):
                            st.markdown("### Comprehensive Analysis")
                            st.markdown(response['search_result'])
                
                else:
                    st.error(f"❌ Error: {response['message']}")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please enter a request.")

# Footer
st.markdown("---")
st.markdown("© 2025 OpenCodeHq. All rights reserved.")
