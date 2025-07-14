import os
import requests
import lance
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

    # Create the output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    dataset = lance.dataset(output_lance_path)
    print(f"Number of rows: {dataset.count_rows()}")
    assert dataset.count_rows() > 0

    # Verify schema and data
    schema = dataset.schema
    print(f"Schema: {schema}")
    assert "text" in schema.names
    assert "image" in schema.names

    table = dataset.to_table()
    text_column = table.column("text")
    image_column = table.column("image")

    # Check that text and image data is not null where expected
    assert all(text is not None for text in text_column)
    assert any(image is not None for image in image_column)
    print("Multimodal ingestion successful!")

if __name__ == "__main__":
    main()
