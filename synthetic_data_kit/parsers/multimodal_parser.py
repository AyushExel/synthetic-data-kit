# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image


class MultimodalParser:
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file, extracting text and images.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                 represents a page with its text and a single image.
        """
        doc = fitz.open(file_path)
        data = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            image_list = page.get_images(full=True)

            if not image_list:
                data.append({"text": text, "image": None})

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                data.append({"text": text, "image": image_bytes})

        return data
