"""
PDF Information Extraction using LlamaCloud and LangChain with Structured Output

This script:
1. Extracts text from PDF using LlamaCloud parsing
2. Uses LangChain with structured output to extract community service items
3. Validates output against Pydantic models
4. Saves results in the AdapterData format
"""

import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Optional

from llama_cloud import LlamaCloud
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from schema import Item, AdapterData, Adapter, StateEnum


# Load environment variables
load_dotenv()


class ExtractedOffers(BaseModel):
    """Container for multiple extracted items from a document"""
    offers: List[Item] = Field(
        description="List of all community services, locations, or events found in the document"
    )


class PDFExtractor:
    """Handles PDF parsing and information extraction"""
    
    def __init__(
        self,
        llama_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            llama_api_key: LlamaCloud API key (defaults to LLAMA_CLOUD_API_KEY env var)
            gemini_api_key: Google Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Gemini model to use for extraction
        """
        self.llama_api_key = llama_api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.llama_api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY not found in environment")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        # Initialize LlamaCloud client
        self.llama_client = LlamaCloud(api_key=self.llama_api_key)
        
        # Initialize LangChain LLM with structured output
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=self.gemini_api_key,
            temperature=0
        )
        
    def parse_pdf(self, pdf_path: str, tier: str = "cost_effective") -> str:
        """
        Parse PDF to markdown using LlamaCloud.
        
        Args:
            pdf_path: Path to the PDF file
            tier: Parsing tier ('cost_effective' or 'agentic')
            
        Returns:
            Markdown content as string
        """
        print(f"Uploading and parsing PDF: {pdf_path}")
        
        # Upload file
        file = self.llama_client.files.create(
            file=pdf_path,
            purpose="parse"
        )
        
        # Parse the file
        result = self.llama_client.parsing.parse(
            file_id=file.id,
            tier=tier,
            version="latest",
            expand=["markdown"],
        )
        
        # Combine all pages
        markdown_content = "\n\n".join(
            page.markdown for page in result.markdown.pages
        )
        
        print(f"✓ PDF parsed successfully ({len(result.markdown.pages)} pages)")
        return markdown_content
    
    def extract_items(self, markdown_content: str) -> List[Item]:
        """
        Extract structured items from markdown content using LangChain.
        
        Args:
            markdown_content: Markdown text from PDF
            
        Returns:
            List of Item objects
        """
        print("Extracting structured information...")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured information about community services, locations, and events from documents.

Your task:
1. Identify EVERY individual service, location, facility, or event in the document
2. Extract information for EACH one separately
3. Only fill fields where information is actually present in the text
4. For multilingual fields (brief, description, etc.), use 'de' as the language key
5. Set 'state' to 'draft' for all items
6. Write descriptions in clear, accessible language without jargon

{format_instructions}"""),
            ("user", """Extract all services, locations, and events from the following document:

DOCUMENT:
{document}""")
        ])
        
        # Create structured output chain
        structured_llm = self.llm.with_structured_output(ExtractedOffers)
        
        chain = prompt | structured_llm
        
        # Run extraction
        result = chain.invoke({
            "document": markdown_content,
            "format_instructions": "Return a JSON object with an 'offers' array containing all extracted items."
        })
        
        print(f"✓ Extracted {len(result.offers)} items")
        return result.offers
    
    def create_adapter_data(
        self,
        items: List[Item],
        adapter_name: str = "pdf-import",
        source_name: str = "PDF Import",
        source_url: Optional[str] = None
    ) -> AdapterData:
        """
        Wrap items in AdapterData format.
        
        Args:
            items: List of Item objects
            adapter_name: Name of the adapter
            source_name: Human-readable source name
            source_url: Optional URL to source
            
        Returns:
            AdapterData object
        """
        # Create items record with UUIDs as keys
        items_record = {str(uuid.uuid4()): item for item in items}
        
        adapter_data = AdapterData(
            adapter=Adapter(
                name=adapter_name,
                sourceName=source_name,
                sourceUrl=source_url
            ),
            lastUpdate=int(time.time()),
            version="1.0",
            itemsRecord=items_record
        )
        
        return adapter_data
    
    def process_pdf(
        self,
        pdf_path: str,
        output_path: str,
        save_markdown: bool = True,
        adapter_name: str = "pdf-import",
        source_name: str = "PDF Import"
    ) -> AdapterData:
        """
        Complete pipeline: parse PDF, extract items, save results.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Path for output JSON file
            save_markdown: Whether to save intermediate markdown
            adapter_name: Name for the adapter
            source_name: Human-readable source name
            
        Returns:
            AdapterData object
        """
        # Step 1: Parse PDF
        markdown_content = self.parse_pdf(pdf_path)
        
        # Optionally save markdown
        if save_markdown:
            md_path = Path(output_path).with_suffix('.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"✓ Saved markdown to {md_path}")
        
        # Step 2: Extract items
        items = self.extract_items(markdown_content)
        
        # Step 3: Create adapter data
        adapter_data = self.create_adapter_data(
            items=items,
            adapter_name=adapter_name,
            source_name=source_name
        )
        
        # Step 4: Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                adapter_data.model_dump(exclude_none=True),
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"✓ Saved {len(items)} items to {output_path}")
        
        return adapter_data


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract community services information from PDF"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.json",
        help="Output JSON file path (default: output.json)"
    )
    parser.add_argument(
        "--adapter-name",
        default="pdf-import",
        help="Name for the data adapter"
    )
    parser.add_argument(
        "--source-name",
        default="PDF Import",
        help="Human-readable source name"
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Don't save intermediate markdown file"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-exp",
        help="Gemini model to use"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = PDFExtractor(model_name=args.model)
    
    # Process PDF
    try:
        adapter_data = extractor.process_pdf(
            pdf_path=args.pdf_path,
            output_path=args.output,
            save_markdown=not args.no_markdown,
            adapter_name=args.adapter_name,
            source_name=args.source_name
        )
        
        print(f"\n✓ Processing complete!")
        print(f"  Items extracted: {len(adapter_data.itemsRecord)}")
        print(f"  Output file: {args.output}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
