import streamlit as st
import PyPDF2
from openai import OpenAI
import anthropic
from docx import Document
import io
import tempfile
import os
import re
import logging
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Get API keys from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CLAUDE_API_KEY = st.secrets["CLAUDE_API_KEY"]
    # Initialize clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
except KeyError as e:
    st.error(f"API key not found in secrets: {str(e)}. Please add both OPENAI_API_KEY and CLAUDE_API_KEY to your Streamlit secrets.")
    st.stop()
except Exception as e:
    st.error(f"Error initializing API clients: {str(e)}")
    st.stop()

def split_pdf_into_chunks(pdf_file, pages_per_chunk=5) -> List[Dict[str, Any]]:
    """Split PDF into chunks of specified pages and extract text."""
    logger.info("Starting PDF chunking process")
    chunks = []
    
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)
    logger.info(f"PDF has {total_pages} pages, splitting into chunks of {pages_per_chunk} pages each")
    
    for start_page in range(0, total_pages, pages_per_chunk):
        end_page = min(start_page + pages_per_chunk, total_pages)
        
        # Extract text from pages in this chunk
        chunk_text = ""
        page_texts = []
        
        for page_num in range(start_page, end_page):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            page_texts.append({
                'page_number': page_num + 1,
                'text': page_text
            })
            chunk_text += f"Page {page_num + 1}:\n{page_text}\n\n"
        
        chunks.append({
            'start_page': start_page + 1,
            'end_page': end_page,
            'text': chunk_text,
            'page_texts': page_texts
        })
        
        logger.info(f"Created chunk for pages {start_page + 1}-{end_page}")
    
    logger.info(f"PDF chunking complete. Created {len(chunks)} chunks")
    return chunks

def split_doc_into_chunks(doc_file, pages_per_chunk=5) -> List[Dict[str, Any]]:
    """Split DOC/DOCX into chunks simulating pages (like PDF)."""
    logger.info("Starting DOC/DOCX chunking process")
    chunks = []
    
    doc = Document(doc_file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    total_paragraphs = len(paragraphs)
    
    # Estimate pages: assume ~12 paragraphs per "page" for DOC files
    paragraphs_per_page = 12
    estimated_pages = max(1, total_paragraphs // paragraphs_per_page)
    
    logger.info(f"Document has {total_paragraphs} paragraphs (~{estimated_pages} estimated pages), splitting into chunks of {pages_per_chunk} pages each")
    
    # Calculate paragraphs per chunk based on pages
    paragraphs_per_chunk = pages_per_chunk * paragraphs_per_page
    
    page_counter = 1
    for start_para in range(0, total_paragraphs, paragraphs_per_chunk):
        end_para = min(start_para + paragraphs_per_chunk, total_paragraphs)
        
        # Calculate page range for this chunk
        start_page = page_counter
        pages_in_chunk = min(pages_per_chunk, (end_para - start_para + paragraphs_per_page - 1) // paragraphs_per_page)
        end_page = start_page + pages_in_chunk - 1
        
        # Extract text from paragraphs in this chunk
        chunk_paragraphs = paragraphs[start_para:end_para]
        chunk_text = "\n".join(chunk_paragraphs)
        
        chunks.append({
            'start_page': start_page,
            'end_page': end_page,
            'text': chunk_text,
            'paragraph_count': len(chunk_paragraphs)
        })
        
        logger.info(f"Created chunk for pages {start_page}-{end_page} (paragraphs {start_para + 1}-{end_para})")
        page_counter = end_page + 1
    
    logger.info(f"DOC/DOCX chunking complete. Created {len(chunks)} chunks")
    return chunks

def check_translation_openai(text: str, target_language: str, chunk_id: int) -> Dict[str, Any]:
    """Use OpenAI GPT-4 to check translation quality."""
    logger.info(f"Starting OpenAI analysis for {target_language} translation (Chunk {chunk_id})")
    
    prompt = f"""
    Please analyze the following text and check if it is properly translated to {target_language}. 
    Look for:
    1. Untranslated words or phrases
    2. Poor grammar or syntax
    3. Inconsistent terminology
    4. Mixed languages within sentences
    
    For each issue found, please specify:
    - The exact problematic text
    - What the issue is (untranslated, grammar error, etc.)
    - A suggested correction if applicable
    
    IMPORTANT: At the end of your analysis, provide a numerical quality score as a percentage (0-100%) in this exact format:
    "Quality Score: XX%"
    
    If no issues are found, respond with "No translation issues found. Quality Score: 100%"
    
    Text to analyze:
    {text}
    """
    
    try:
        logger.info(f"Sending request to OpenAI API (Chunk {chunk_id})")
        response = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"You are a professional {target_language} translator and proofreader. You have expertise in identifying translation issues and providing corrections. Always end your response with a quality score percentage."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10000,
            temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content
        
        # Extract quality score
        score_match = re.search(r'Quality Score:\s*(\d+)%', analysis_text)
        quality_score = int(score_match.group(1)) if score_match else None
        
        logger.info(f"OpenAI analysis complete (Chunk {chunk_id}). Quality score: {quality_score}%")
        
        return {
            'success': True,
            'analysis': analysis_text,
            'usage': response.usage,
            'model': 'OpenAI GPT-4',
            'quality_score': quality_score,
            'chunk_id': chunk_id
        }
    
    except Exception as e:
        logger.error(f"OpenAI API error (Chunk {chunk_id}): {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model': 'OpenAI GPT-4',
            'chunk_id': chunk_id
        }

def check_translation_claude(text: str, target_language: str, chunk_id: int) -> Dict[str, Any]:
    """Use Claude to check translation quality."""
    logger.info(f"Starting Claude analysis for {target_language} translation (Chunk {chunk_id})")
    
    if target_language == "French":
        prompt = f"""
        Please analyze the following text and check if it is properly translated to French.
        Look for:
        - Untranslated English words or phrases
        - Poor grammar or syntax in French
        - Inconsistent terminology
        - Mixed languages within sentences
        
        For each issue found, specify:
        - The exact problematic text
        - What the issue is
        - A suggested correction
        
        IMPORTANT: At the end of your analysis, provide a numerical quality score as a percentage (0-100%) in this exact format:
        "Quality Score: XX%"
        
        Text to analyze:
        {text}
        """
    else:  # Arabic
        prompt = f"""
        Role: You are an expert Arabic translator and proofreader.

        Task:
        - Detect the source text language and confirm it's translated to Arabic
        - Carefully proofread the Arabic text for accuracy and fluency
        - Identify every error, awkward phrasing, or mistranslation
        
        For each issue found:
        - Mark the problematic Arabic text clearly
        - Explain why it's incorrect
        - Provide a proper Arabic correction
        
        IMPORTANT: At the end of your analysis, provide a numerical quality score as a percentage (0-100%) in this exact format:
        "Quality Score: XX%"

        Text to analyze:
        {text}
        """
    
    try:
        logger.info(f"Sending request to Claude API (Chunk {chunk_id})")
        response = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10000,
            temperature=0.3,
            system=f"You are a professional {target_language} translator and proofreader with expertise in identifying translation issues. Always end your response with a quality score percentage.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis_text = response.content[0].text
        
        # Extract quality score
        score_match = re.search(r'Quality Score:\s*(\d+)%', analysis_text)
        quality_score = int(score_match.group(1)) if score_match else None
        
        logger.info(f"Claude analysis complete (Chunk {chunk_id}). Quality score: {quality_score}%")
        
        return {
            'success': True,
            'analysis': analysis_text,
            'usage': response.usage,
            'model': 'Claude',
            'quality_score': quality_score,
            'chunk_id': chunk_id
        }
    
    except Exception as e:
        logger.error(f"Claude API error (Chunk {chunk_id}): {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'model': 'Claude',
            'chunk_id': chunk_id
        }

def process_all_chunks_async(chunks: List[Dict[str, Any]], target_language: str) -> List[Dict[str, Any]]:
    """Process all chunks concurrently using ThreadPoolExecutor."""
    logger.info(f"Starting concurrent processing of {len(chunks)} chunks")
    
    results = []
    openai_scores = []
    claude_scores = []
    
    # Create a ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=10) as executor:  # Allow up to 10 concurrent API calls
        # Submit all OpenAI tasks
        openai_futures = []
        claude_futures = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = i + 1
            
            # Submit OpenAI task
            openai_future = executor.submit(check_translation_openai, chunk['text'], target_language, chunk_id)
            openai_futures.append((openai_future, chunk_id, chunk))
            
            # Submit Claude task
            claude_future = executor.submit(check_translation_claude, chunk['text'], target_language, chunk_id)
            claude_futures.append((claude_future, chunk_id, chunk))
        
        logger.info(f"Submitted {len(openai_futures)} OpenAI tasks and {len(claude_futures)} Claude tasks")
        
        # Collect OpenAI results
        openai_results = {}
        for future, chunk_id, chunk in openai_futures:
            try:
                result = future.result()
                openai_results[chunk_id] = result
                if result['success'] and result.get('quality_score'):
                    openai_scores.append(result['quality_score'])
                    logger.info(f"OpenAI task {chunk_id} completed with score: {result['quality_score']}%")
            except Exception as e:
                logger.error(f"OpenAI task {chunk_id} failed: {str(e)}")
                openai_results[chunk_id] = {'success': False, 'error': str(e), 'model': 'OpenAI GPT-4', 'chunk_id': chunk_id}
        
        # Collect Claude results
        claude_results = {}
        for future, chunk_id, chunk in claude_futures:
            try:
                result = future.result()
                claude_results[chunk_id] = result
                if result['success'] and result.get('quality_score'):
                    claude_scores.append(result['quality_score'])
                    logger.info(f"Claude task {chunk_id} completed with score: {result['quality_score']}%")
            except Exception as e:
                logger.error(f"Claude task {chunk_id} failed: {str(e)}")
                claude_results[chunk_id] = {'success': False, 'error': str(e), 'model': 'Claude', 'chunk_id': chunk_id}
    
    # Combine results
    for i, chunk in enumerate(chunks):
        chunk_id = i + 1
        chunk_info = f"Pages {chunk['start_page']}-{chunk['end_page']}"
        
        results.append({
            'chunk_number': chunk_id,
            'chunk_info': chunk_info,
            'openai_analysis': openai_results.get(chunk_id, {'success': False, 'error': 'No result', 'model': 'OpenAI GPT-4'}),
            'claude_analysis': claude_results.get(chunk_id, {'success': False, 'error': 'No result', 'model': 'Claude'})
        })
    
    logger.info(f"Concurrent processing complete. OpenAI scores: {openai_scores}, Claude scores: {claude_scores}")
    
    return results, openai_scores, claude_scores

def main():
    st.set_page_config(
        page_title="Patrick - The PDF Proofreader",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Patrick - The PDF Proofreader")
    st.markdown("Upload a PDF file to check translation quality using AI-powered analysis.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Language selection
        target_language = st.selectbox(
            "Select target language:",
            ["French", "Arabic"],
            help="Choose the language you want to check the translation for"
        )
        
        if target_language == "French":
            st.info("🤖 Using OpenAI GPT-4 for French translation analysis")
        elif target_language == "Arabic":
            st.info("🤖 Using Claude for Arabic translation analysis")
            
        st.success("✅ API keys configured from secrets")
        
        # Performance info
        st.info("⚡ Using concurrent processing for faster analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "doc", "docx"],
        help="Upload a PDF, DOC, or DOCX file to analyze for translation quality"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("🔍 Analyze Document Translation", type="primary"):
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                logger.info(f"Starting analysis of file: {uploaded_file.name}")
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                # Split document based on file type
                if file_extension == 'pdf':
                    status_text.text("Splitting PDF into 5-page chunks...")
                    chunks = split_pdf_into_chunks(uploaded_file)
                    progress_bar.progress(20)
                    st.info(f"PDF split into {len(chunks)} chunks of up to 5 pages each")
                elif file_extension in ['doc', 'docx']:
                    status_text.text("Splitting document into 5-page chunks...")
                    chunks = split_doc_into_chunks(uploaded_file)
                    progress_bar.progress(20)
                    st.info(f"Document split into {len(chunks)} chunks of up to 5 pages each")
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
                
                # Process all chunks concurrently
                status_text.text(f"🚀 Processing {len(chunks)} chunks concurrently with both AI models...")
                logger.info(f"Starting concurrent processing of {len(chunks)} chunks")
                
                results, openai_scores, claude_scores = process_all_chunks_async(chunks, target_language)
                
                status_text.text("Analysis complete!")
                progress_bar.progress(100)
                
                # Display aggregate scores
                st.header("🎯 Overall Quality Scores")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if openai_scores:
                        openai_avg = sum(openai_scores) / len(openai_scores)
                        st.metric("🤖 OpenAI Total Score", f"{openai_avg:.1f}%", 
                                 help=f"Average of {len(openai_scores)} chunks")
                        logger.info(f"OpenAI aggregate score: {openai_avg:.1f}%")
                    else:
                        st.metric("🤖 OpenAI Total Score", "N/A", help="No valid scores")
                        logger.warning("No valid OpenAI scores obtained")
                
                with col2:
                    if claude_scores:
                        claude_avg = sum(claude_scores) / len(claude_scores)
                        st.metric("🧠 Claude Total Score", f"{claude_avg:.1f}%", 
                                 help=f"Average of {len(claude_scores)} chunks")
                        logger.info(f"Claude aggregate score: {claude_avg:.1f}%")
                    else:
                        st.metric("🧠 Claude Total Score", "N/A", help="No valid scores")
                        logger.warning("No valid Claude scores obtained")
                
                # Display detailed results
                st.header("📊 Detailed Analysis Results")
                
                for result in results:
                    with st.expander(f"Chunk {result['chunk_number']} ({result['chunk_info']})", expanded=False):
                        
                        # Create two columns for side-by-side comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🤖 OpenAI GPT-4 Analysis")
                            if result['openai_analysis']['success']:
                                st.write(result['openai_analysis']['analysis'])
                                
                                # Show usage info
                                if 'usage' in result['openai_analysis']:
                                    usage = result['openai_analysis']['usage']
                                    if hasattr(usage, 'total_tokens'):
                                        st.caption(f"Tokens: {usage.total_tokens}")
                                
                                # Show individual score
                                if result['openai_analysis'].get('quality_score'):
                                    st.success(f"Score: {result['openai_analysis']['quality_score']}%")
                            else:
                                st.error(f"Error: {result['openai_analysis']['error']}")
                        
                        with col2:
                            st.markdown("### 🧠 Claude Analysis")
                            if result['claude_analysis']['success']:
                                st.write(result['claude_analysis']['analysis'])
                                
                                # Show usage info
                                if 'usage' in result['claude_analysis']:
                                    usage = result['claude_analysis']['usage']
                                    if hasattr(usage, 'input_tokens'):
                                        st.caption(f"Tokens: {usage.input_tokens + usage.output_tokens}")
                                
                                # Show individual score
                                if result['claude_analysis'].get('quality_score'):
                                    st.success(f"Score: {result['claude_analysis']['quality_score']}%")
                            else:
                                st.error(f"Error: {result['claude_analysis']['error']}")
                
                # Summary statistics
                st.header("📋 Analysis Summary")
                total_chunks = len(results)
                openai_success = sum(1 for r in results if r['openai_analysis']['success'])
                claude_success = sum(1 for r in results if r['claude_analysis']['success'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", total_chunks)
                with col2:
                    st.metric("OpenAI Success Rate", f"{openai_success/total_chunks*100:.1f}%")
                with col3:
                    st.metric("Claude Success Rate", f"{claude_success/total_chunks*100:.1f}%")
                
            except Exception as e:
                logger.error(f"Analysis failed: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
                status_text.text("Analysis failed!")
                progress_bar.progress(0)
    
    # Easter egg footer message
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.8em; margin-top: 2rem;'>
            🌟 <em>"Is mayonnaise an instrument?"</em> - Patrick Star 🌟<br>
            Made with ❤️ by Patrick (the PDF proofreader, not the starfish) 🧽⭐
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 