import os
import shutil
import requests
import lancedb as lance
from synthetic_data_kit.cli import app
from typer.testing import CliRunner

# URL of a sample PDF with images for testing
PDF_URL = "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf"
PDF_FILENAME = "sample_multimodal.pdf"
OUTPUT_DIR = "example_output"

def main():
    """Download a PDF and run multimodal ingestion."""
    # Set the API key
    os.environ["API_ENDPOINT_KEY"] = "LLM|704426635437672|3nFowHkWPXPZWYaepVCJC0Z3GMw"

    # Clean and create the output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Download the test PDF
    response = requests.get(PDF_URL)
    pdf_path = os.path.join(OUTPUT_DIR, PDF_FILENAME)
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    # Run the ingest command with the --multimodal flag
    runner = CliRunner()
    result = runner.invoke(app, [
        "ingest",
        pdf_path,
        "--output-dir",
        OUTPUT_DIR,
        "--multimodal",
    ])

    # Verify the output
    print(result.stdout)
    output_lance_path = os.path.join(OUTPUT_DIR, "sample_multimodal.lance")
    assert os.path.exists(output_lance_path)

    # Check the contents of the Lance dataset
    db = lance.connect(OUTPUT_DIR)
    table = db.open_table("sample_multimodal")
    print(f"Number of rows: {len(table)}")
    assert len(table) > 0

    # Verify schema and data
    schema = table.schema
    print(f"Schema: {schema}")
    assert "text" in schema.names
    assert "image" in schema.names

    df = table.to_arrow().to_pandas()
    text_column = df["text"]
    image_column = df["image"]

    # Check that text and image data is not null where expected
    assert all(text is not None for text in text_column)
    assert any(image is not None for image in image_column)
    print("Multimodal ingestion successful!")

    # Run the create command
    result = runner.invoke(app, [
        "create",
        output_lance_path,
        "--output-dir",
        OUTPUT_DIR,
        "--type",
        "qa",
    ])

    # Verify the output
    print(result.stdout)
    output_json_path = os.path.join(OUTPUT_DIR, "sample_multimodal_qa_pairs.json")
    assert os.path.exists(output_json_path)
    print("QA pair generation successful!")


if __name__ == "__main__":
    main()
